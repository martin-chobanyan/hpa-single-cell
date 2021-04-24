import os
from yaml import safe_load

import albumentations as A
import pandas as pd
import torch
from torch.nn import Conv2d, ReLU, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYWithCellMasks, N_CHANNELS, CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.loader import cell_mask_collate
from hpa.data.transforms import HPACompose
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.layers import ConvBlock
from hpa.model.localizers import PeakStimClassRoI
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger
from hpa.utils.train_roi import train_peak_roi_epoch, test_peak_roi_epoch

if __name__ == '__main__':
    print('Training a weakly-supervised RoI localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/roi/roi2.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    IMG_DIM = config['data']['image_size']
    CROP_SIZE = config['data']['crop']
    BATCH_SIZE = config['data']['batch_size']

    dual_train_transform_fn = HPACompose([
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE)
    ])

    dual_val_transform_fn = HPACompose([
        A.CenterCrop(height=CROP_SIZE, width=CROP_SIZE),
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

    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'joint', 'stratified', 'train-idx.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'joint', 'stratified', 'val-idx.csv'))

    train_idx = train_idx.head(200)
    val_idx = val_idx.head(200)

    train_data = RGBYWithCellMasks(data_idx=train_idx,
                                   data_dir=DATA_DIR,
                                   seg_dir=SEG_DIR,
                                   external_data_dir=EXTERNAL_DATA_DIR,
                                   external_seg_dir=EXTERNAL_SEG_DIR,
                                   dual_transforms=dual_train_transform_fn,
                                   img_transforms=img_transform_fn,
                                   img_dim=CROP_SIZE)

    val_data = RGBYWithCellMasks(data_idx=val_idx,
                                 data_dir=DATA_DIR,
                                 seg_dir=SEG_DIR,
                                 external_data_dir=EXTERNAL_DATA_DIR,
                                 external_seg_dir=EXTERNAL_SEG_DIR,
                                 dual_transforms=dual_val_transform_fn,
                                 img_transforms=img_transform_fn,
                                 img_dim=CROP_SIZE)

    train_loader = DataLoader(train_data,
                              collate_fn=cell_mask_collate,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_data,
                            collate_fn=cell_mask_collate,
                            batch_size=BATCH_SIZE,
                            pin_memory=True,
                            num_workers=NUM_WORKERS)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------
    DEVICE = 'cuda'
    LR = config['model']['lr']
    PRETRAINED_PATH = config['pretrained_path']

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
    backbone_cnn = Sequential(densenet_encoder, ConvBlock(1024, 1024, kernel_size=3))
    final_conv = Conv2d(1024, 18, 1)

    model = PeakStimClassRoI(backbone=backbone_cnn, final_conv=final_conv)
    model = model.to(DEVICE)

    criterion = FocalSymmetricLovaszHardLogLoss()
    optimizer = AdamW([
        {'params': model.backbone[1].parameters(), 'lr': LR},
        {'params': model.final_conv.parameters(), 'lr': LR}
    ])

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

    W_ROI = config['model']['w_roi']
    W_PEAK = config['model']['w_peak']

    N_EPOCHS = config['model']['epochs']
    N_TRAIN_BATCHES = int(len(train_data) / BATCH_SIZE)
    N_VAL_BATCHES = int(len(val_data) / BATCH_SIZE)

    HEADER = [
        'epoch',
        'train_loss',
        'train_roi_loss',
        'train_peak_loss',
        'val_loss',
        'val_roi_loss',
        'val_peak_loss',
        'val_bce_loss',
        'val_focal_loss',
        'val_exact',
        'val_f1'
    ]
    logger = Logger(LOGGER_PATH, header=HEADER)

    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        train_results = train_peak_roi_epoch(model=model,
                                             dataloader=train_loader,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             device=DEVICE,
                                             w_roi=W_ROI,
                                             w_peak=W_PEAK,
                                             clip_grad_value=1,
                                             progress=True,
                                             epoch=epoch,
                                             n_batches=N_TRAIN_BATCHES)

        val_results = test_peak_roi_epoch(model=model,
                                          dataloader=val_loader,
                                          criterion=criterion,
                                          device=DEVICE,
                                          w_roi=W_ROI,
                                          w_peak=W_PEAK,
                                          calc_bce=True,
                                          calc_focal=True,
                                          progress=True,
                                          epoch=epoch,
                                          n_batches=N_VAL_BATCHES)

        logger.add_entry(epoch, *train_results, *val_results)

        # checkpoint all epochs for now
        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
