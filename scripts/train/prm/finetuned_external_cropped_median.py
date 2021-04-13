import os
from yaml import safe_load

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import torch
from torch.nn import Conv2d, ReLU, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYDataset, N_CHANNELS, CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.transforms import ChannelSpecificAug, HPACompose
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.layers import ConvBlock
from hpa.model.localizers import PeakResponseLocalizer
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch, test_epoch

if __name__ == '__main__':
    print('Training a weakly-supervised PRM localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/decomposed/prm/prm12.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    IMG_DIM = config['data']['image_size']
    CROP_SIZE = config['data']['crop']
    BATCH_SIZE = config['data']['batch_size']

    # ref_color_aug = A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0)
    # tgt_color_aug = A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.1, 0.3), p=1.0)

    train_transform_fn = HPACompose([
        A.Resize(IMG_DIM, IMG_DIM),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        # ChannelSpecificAug(aug=ref_color_aug, channels=[0, 3, 2], p=0.5),
        # ChannelSpecificAug(aug=tgt_color_aug, channels=[1], p=0.5),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255),
        ToTensorV2()
    ])

    val_transform_fn = HPACompose([
        A.Resize(IMG_DIM, IMG_DIM),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255),
        ToTensorV2()
    ])

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'train')
    EXTERNAL_DATA_DIR = os.path.join(ROOT_DIR, 'misc', 'public-hpa', 'data2')
    NUM_WORKERS = 3

    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'joint', 'stratified', 'train-idx.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'splits', 'joint', 'stratified', 'val-idx.csv'))

    # train_idx = train_idx.head(64)
    # val_idx = val_idx.head(64)

    train_data = RGBYDataset(train_idx, DATA_DIR, EXTERNAL_DATA_DIR, train_transform_fn)
    val_data = RGBYDataset(val_idx, DATA_DIR, EXTERNAL_DATA_DIR, val_transform_fn)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------
    DEVICE = 'cuda'
    LR = config['model']['lr']
    MIN_LR = config['model']['min_lr']
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

    backbone_cnn = Sequential(densenet_encoder, ConvBlock(1024, 1024, kernel_size=3), Conv2d(1024, 18, 1))

    model = PeakResponseLocalizer(cnn=backbone_cnn, return_maps=False, return_peaks=False)
    model = model.to(DEVICE)

    criterion = FocalSymmetricLovaszHardLogLoss()
    optimizer = AdamW([
        {'params': model.cnn[0].parameters(), 'lr': MIN_LR},
        {'params': model.cnn[1].parameters(), 'lr': LR},
        {'params': model.cnn[2].parameters(), 'lr': LR}
    ])

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LR_STEP = config['model']['lr_step']
    STEP_DELAY = config['model']['step_delay']
    N_EPOCHS = config['model']['epochs']
    ACCUM_GRAD = config['model']['accum_grad']

    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

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

        if epoch >= STEP_DELAY:
            LR = max(LR - LR_STEP, MIN_LR)
            print(f'Lowering learning rate: {LR}')
            optimizer = AdamW([
                {'params': model.cnn[0].parameters(), 'lr': MIN_LR},
                {'params': model.cnn[1].parameters(), 'lr': LR},
                {'params': model.cnn[2].parameters(), 'lr': LR}
            ])

        train_loss = train_epoch(model=model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 device=DEVICE,
                                 accum_grad=ACCUM_GRAD,
                                 clip_grad_value=1,
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

        logger.add_entry(epoch, train_loss, *val_results)

        # checkpoint all epochs for now
        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
