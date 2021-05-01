import os
from yaml import safe_load

import albumentations as A
from albumentations.pytorch import ToTensorV2
from cv2 import BORDER_CONSTANT
import pandas as pd
import torch
from torch.nn import Conv2d, ReLU, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYDataset, N_CHANNELS, CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.transforms import HPACompose
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.layers import SqueezeAndExciteBlock
from hpa.model.localizers import PeakResponseLocalizer
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch, test_epoch, LRScheduler


def create_optimizer(prm_model, lr_value):
    param_groups = [
        {'params': prm_model.backbone[1].parameters(), 'lr': lr_value},
    ]
    return AdamW(param_groups)


if __name__ == '__main__':
    print('Training a weakly-supervised PRM localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/decomposed/prm/prm19.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    IMG_DIM = config['data']['image_size']
    CROP_SIZE = config['data']['crop']
    BATCH_SIZE = config['data']['batch_size']

    train_transform_fn = HPACompose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5, border_mode=BORDER_CONSTANT, mask_value=0),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255),
        ToTensorV2()
    ])

    val_transform_fn = HPACompose([
        A.CenterCrop(height=CROP_SIZE, width=CROP_SIZE),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255),
        ToTensorV2()
    ])

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'images', 'competition_1536x1536')
    SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'competition_1536x1536')
    EXTERNAL_DATA_DIR = os.path.join(ROOT_DIR, 'images', 'public_1536x1536')
    EXTERNAL_SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'public_1536x1536')
    NUM_WORKERS = 4

    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'joint', 'stratified', 'train-idx.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'joint', 'stratified', 'val-idx.csv'))

    # train_idx = train_idx.head(200)
    # val_idx = val_idx.head(200)

    train_data = RGBYDataset(train_idx, DATA_DIR, EXTERNAL_DATA_DIR, train_transform_fn)
    val_data = RGBYDataset(val_idx, DATA_DIR, EXTERNAL_DATA_DIR, val_transform_fn)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------
    DEVICE = 'cuda'
    PRETRAINED_PATH = config['pretrained_path']

    NUM_SE_LAYERS = config['model']['squeeze_and_excite']['num_layers']
    SE_HIDDEN_DIM = config['model']['squeeze_and_excite']['hidden_dim']
    SE_SQUEEZE_DIM = config['model']['squeeze_and_excite']['squeeze_dim']

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

    for param in densenet_encoder.parameters():
        param.requires_grad = False

    # submodules
    se_layers = Sequential(*[SqueezeAndExciteBlock(1024, SE_HIDDEN_DIM, SE_SQUEEZE_DIM) for _ in range(NUM_SE_LAYERS)])
    final_conv = Conv2d(1024, 18, 1)
    conv_layers = Sequential(se_layers, final_conv)
    backbone_cnn = Sequential(densenet_encoder, conv_layers)

    model = PeakResponseLocalizer(backbone=backbone_cnn)
    model = model.to(DEVICE)

    criterion = FocalSymmetricLovaszHardLogLoss()

    # -------------------------------------------------------------------------------------------
    # Prepare the optimizer
    # -------------------------------------------------------------------------------------------

    INIT_LR = config['model']['init_lr']
    MIN_LR = config['model']['min_lr']
    LR_STEP = config['model']['lr_step']
    STEP_DELAY = config['model']['step_delay']

    lr_scheduler = LRScheduler(init_lr=INIT_LR, min_lr=MIN_LR, increment=LR_STEP, delay_start=STEP_DELAY)
    optimizer = create_optimizer(model, INIT_LR)

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

    N_EPOCHS = config['model']['epochs']
    N_TRAIN_BATCHES = int(len(train_data) / BATCH_SIZE)
    N_VAL_BATCHES = int(len(val_data) / BATCH_SIZE)

    HEADER = [
        'epoch',
        'train_loss',
        'val_loss',
        'val_bce_loss',
        'val_focal_loss',
        'val_exact',
        'val_f1'
    ]
    logger = Logger(LOGGER_PATH, header=HEADER)

    best_loss = float('inf')
    for epoch in range(N_EPOCHS):

        train_loss = train_epoch(model=model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 device=DEVICE,
                                 progress=True,
                                 epoch=epoch,
                                 n_batches=N_TRAIN_BATCHES)

        val_results = test_epoch(model=model,
                                 dataloader=val_loader,
                                 criterion=criterion,
                                 device=DEVICE,
                                 calc_bce=True,
                                 calc_focal=True,
                                 progress=True,
                                 epoch=epoch,
                                 n_batches=N_VAL_BATCHES)

        lr = lr_scheduler.update()
        optimizer = create_optimizer(model, lr)

        logger.add_entry(epoch, train_loss, *val_results)

        # checkpoint all epochs for now
        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
