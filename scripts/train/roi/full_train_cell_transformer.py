import os
from yaml import safe_load

import albumentations as A
from cv2 import BORDER_CONSTANT
import pandas as pd
import torch
from torch.nn import ReLU, Sequential, Upsample
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYWithCellMasks, N_CHANNELS, CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.loader import cell_mask_collate
from hpa.data.transforms import HPACompose
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.layers import RoIPool
from hpa.model.localizers import CellTransformer
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.model import get_num_params, get_num_opt_params
from hpa.utils.train import checkpoint, Logger, LRScheduler
from hpa.utils.train_roi import train_roi_epoch


def create_optimizer(model, lr_backbone, lr_transform):
    backbone_params = []
    transform_params = []
    for param_name, p in model.named_parameters():
        if 'backbone' in param_name:
            backbone_params.append(p)
        else:
            transform_params.append(p)
    return AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': transform_params, 'lr': lr_transform}
    ])


if __name__ == '__main__':
    print('Training a weakly-supervised RoI localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/roi/roi15.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    IMG_DIM = config['data']['image_size']
    CROP_SIZE = config['data']['crop']
    DOWNSIZE_SCALE = config['data']['downsize']
    BATCH_SIZE = config['data']['batch_size']

    dual_train_transform_fn = HPACompose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5, border_mode=BORDER_CONSTANT, mask_value=0),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE)
    ])

    img_transform_fn = A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255)

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'images', 'competition_1536x1536')
    SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'competition_1536x1536')
    EXTERNAL_DATA_DIR = os.path.join(ROOT_DIR, 'images', 'public_1536x1536')
    EXTERNAL_SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'public_1536x1536')
    NUM_WORKERS = 4

    data_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'complete-data-idx.csv'))
    # data_idx = data_idx.head(200)

    dataset = RGBYWithCellMasks(data_idx=data_idx,
                                data_dir=DATA_DIR,
                                seg_dir=SEG_DIR,
                                external_data_dir=EXTERNAL_DATA_DIR,
                                external_seg_dir=EXTERNAL_SEG_DIR,
                                dual_transforms=dual_train_transform_fn,
                                img_transforms=img_transform_fn,
                                downsize_scale=DOWNSIZE_SCALE,
                                img_dim=CROP_SIZE)

    dataloader = DataLoader(dataset,
                            collate_fn=cell_mask_collate,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=NUM_WORKERS)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------

    DEVICE = 'cuda'
    PRETRAINED_PATH = config['pretrained_path']

    FEATURE_ROI_METHOD = config['model']['cell_transformer']['feature_roi']
    POSITION_ENCODING = config['model']['cell_transformer']['position_encoding']
    POSITION_ENC_SHAPE = config['model']['cell_transformer']['position_encoding_shape']
    NUM_ENCODERS = config['model']['cell_transformer']['num_encoders']
    EMB_DIM = config['model']['cell_transformer']['emb_dim']
    NUM_HEADS = config['model']['cell_transformer']['num_heads']

    densenet_model = DensenetClass(in_channels=N_CHANNELS, dropout=True)

    # load the pretrained DenseNet model
    if PRETRAINED_PATH != '':
        print('Loading pre-trained model')
        pretrained_state_dict = torch.load(PRETRAINED_PATH)['state_dict']
        densenet_model.load_state_dict(pretrained_state_dict)

    densenet_encoder = Sequential(densenet_model.conv1,
                                  densenet_model.encoder2,
                                  densenet_model.encoder3,
                                  densenet_model.encoder4,
                                  densenet_model.encoder5,
                                  ReLU())

    feature_roi_pool = RoIPool(method=FEATURE_ROI_METHOD, positions=POSITION_ENCODING, tgt_shape=POSITION_ENC_SHAPE)
    upsample_fn = Upsample(scale_factor=2, mode='nearest')

    if FEATURE_ROI_METHOD == 'max_and_avg':
        cell_feature_dim = 2048
    else:
        cell_feature_dim = 1024
    if POSITION_ENCODING:
        cell_feature_dim += POSITION_ENC_SHAPE * POSITION_ENC_SHAPE
    print(f'Features extracted per cell = {cell_feature_dim}')

    model = CellTransformer(backbone=densenet_encoder,
                            feature_roi=feature_roi_pool,
                            num_encoders=NUM_ENCODERS,
                            emb_dim=EMB_DIM,
                            num_heads=NUM_HEADS,
                            upsample=upsample_fn,
                            cell_feature_dim=cell_feature_dim,
                            device=DEVICE)

    model = model.to(DEVICE)
    print(f'Total number of parameters: {get_num_params(model)}')

    # -------------------------------------------------------------------------------------------
    # Prepare the optimizer
    # -------------------------------------------------------------------------------------------

    INIT_LR = config['model']['init_lr']
    MIN_LR = config['model']['min_lr']
    LR_STEP = config['model']['lr_step']
    STEP_DELAY = config['model']['step_delay']

    lr_scheduler = LRScheduler(init_lr=INIT_LR, min_lr=MIN_LR, increment=LR_STEP, delay_start=STEP_DELAY)
    optimizer = create_optimizer(model, lr_backbone=MIN_LR, lr_transform=INIT_LR)

    n_train_params = get_num_opt_params(optimizer)
    print(f'Number of trainable parameters: {n_train_params}')

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

    N_EPOCHS = config['model']['epochs']
    N_TRAIN_BATCHES = int(len(dataset) / BATCH_SIZE)

    HEADER = [
        'epoch',
        'train_loss',
    ]
    logger = Logger(LOGGER_PATH, header=HEADER)

    criterion = FocalSymmetricLovaszHardLogLoss()

    for epoch in range(N_EPOCHS):
        train_loss = train_roi_epoch(model=model,
                                     dataloader=dataloader,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     device=DEVICE,
                                     progress=True,
                                     epoch=epoch,
                                     n_batches=N_TRAIN_BATCHES)

        lr = lr_scheduler.update()
        optimizer = create_optimizer(model, lr_backbone=MIN_LR, lr_transform=lr)

        logger.add_entry(epoch, train_loss)

        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
