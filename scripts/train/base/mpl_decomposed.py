import os
from yaml import safe_load

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYDataset, N_CHANNELS, CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.transforms import HPACompose
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.localizers import DecomposedDensenet, PooledLocalizer
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch, test_epoch

if __name__ == '__main__':
    print('Training a weakly-supervised max-pooled localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/decomposed/base/decomposed0.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    img_dim = config['data']['image_size']

    train_transforms = HPACompose([
        A.Resize(img_dim, img_dim),
        A.Flip(p=0.5),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255),
        ToTensorV2()
    ])

    val_transforms = HPACompose([
        A.Resize(img_dim, img_dim),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255),
        ToTensorV2()
    ])

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'train')
    EXTERNAL_DATA_DIR = os.path.join(ROOT_DIR, 'misc', 'public-hpa', 'data2')
    NUM_WORKERS = 12

    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'train-index.csv'))
    # train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'full-train-index.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'val-index.csv'))

    # train_idx = train_idx.head(64)
    # val_idx = val_idx.head(64)

    train_data = RGBYDataset(train_idx, DATA_DIR, transforms=train_transforms)
    # train_data = RGBYDataset(train_idx, DATA_DIR, external_data_dir=EXTERNAL_DATA_DIR, transforms=train_transforms)
    val_data = RGBYDataset(val_idx, DATA_DIR, transforms=val_transforms)

    BATCH_SIZE = config['data']['batch_size']
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------
    DEVICE = 'cuda'
    LR = config['model']['lr']
    N_EPOCHS = config['model']['epochs']
    PRETRAINED_PATH = config['pretrained_path']

    densenet_model = DensenetClass(in_channels=N_CHANNELS, dropout=True)

    # load the pretrained DenseNet model
    if PRETRAINED_PATH != '':
        print('Loading pre-trained model')
        pretrained_state_dict = torch.load(PRETRAINED_PATH)['state_dict']
        densenet_model.load_state_dict(pretrained_state_dict)

    # decompose the model
    decomposed_model = DecomposedDensenet(densenet_model=densenet_model, map_classes=True, max_classes=False)

    # define the localizer model
    model = PooledLocalizer(cnn=decomposed_model, pool='avg', return_maps=False)
    model = model.to(DEVICE)

    criterion = FocalSymmetricLovaszHardLogLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

    N_TRAIN_BATCHES = int(len(train_data) / BATCH_SIZE)
    N_VAL_BATCHES = int(len(val_data) / BATCH_SIZE)

    HEADER = ['epoch', 'train_loss', 'val_loss', 'val_bce_loss', 'val_focal_loss', 'val_exact', 'val_f1']
    logger = Logger(LOGGER_PATH, header=HEADER)

    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model=model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 device=DEVICE,
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
