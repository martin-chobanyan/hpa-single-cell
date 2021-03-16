import os
from yaml import safe_load

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import torch
from torch.nn import ReLU, Sequential, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYDataset, CroppedRGBYDataset, N_CHANNELS, N_CLASSES
from hpa.data.transforms import AdjustableCropCompose, HPACompose
from hpa.data.loader import AlternatingDataLoader, SingleLabelBatchSampler
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.localizers import MaxPooledLocalizer
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch, test_epoch

if __name__ == '__main__':
    print('Training a weakly-supervised max-pooled localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/max-pooled-sigmoid/finetuned-10.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    img_dim = config['data']['image_size']

    base_transform_fn = HPACompose([
        A.Resize(img_dim, img_dim, p=1.0),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(
            mean=[0.074598, 0.050630, 0.050891, 0.076287],
            std=[0.122813, 0.085745, 0.129882, 0.119411],
            max_pixel_value=255
        ),
        ToTensorV2()
    ])

    crop_transform_fn = AdjustableCropCompose([
        A.Resize(img_dim, img_dim, p=1.0),
        A.Flip(p=0.5),
        A.RandomCrop(512, 512),  # this will be adjusted per batch (i.e. not that important)
        A.Normalize(
            mean=[0.074598, 0.050630, 0.050891, 0.076287],
            std=[0.122813, 0.085745, 0.129882, 0.119411],
            max_pixel_value=255
        ),
        ToTensorV2()
    ])

    val_transform_fn = HPACompose([
        A.Resize(img_dim, img_dim, p=1.0),
        A.Normalize(
            mean=[0.074598, 0.050630, 0.050891, 0.076287],
            std=[0.122813, 0.085745, 0.129882, 0.119411],
            max_pixel_value=255
        ),
        ToTensorV2()
    ])

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'train')
    EXTERNAL_DATA_DIR = os.path.join(ROOT_DIR, 'misc', 'public-hpa', 'data2')

    # train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'full-train-index.csv'))
    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'train-index.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'val-index.csv'))

    base_train_data = RGBYDataset(train_idx, DATA_DIR, transforms=base_transform_fn)
    balanced_train_data = CroppedRGBYDataset(train_idx, DATA_DIR, transforms=crop_transform_fn)
    # base_train_data = RGBYDataset(train_idx, DATA_DIR, external_data_dir=EXTERNAL_DATA_DIR, transforms=base_transform_fn)
    # balanced_train_data = CroppedRGBYDataset(train_idx, DATA_DIR, external_data_dir=EXTERNAL_DATA_DIR, transforms=crop_transform_fn)
    val_data = RGBYDataset(val_idx, DATA_DIR, transforms=val_transform_fn)

    BATCH_SIZE = config['data']['batch_size']
    N_TRAIN_BATCHES = int(len(base_train_data) / BATCH_SIZE)
    N_VAL_BATCHES = int(len(val_data) / BATCH_SIZE)
    NUM_WORKERS = 4

    balanced_sampler = SingleLabelBatchSampler(balanced_train_data,
                                               batch_size=BATCH_SIZE,
                                               num_batches=int(N_TRAIN_BATCHES / 2))

    base_train_loader = DataLoader(base_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    balanced_loader = DataLoader(balanced_train_data, batch_sampler=balanced_sampler, num_workers=NUM_WORKERS)

    train_loader = AlternatingDataLoader([base_train_loader, balanced_loader])
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

    # isolate the CNN encoder
    densenet_encoder = Sequential(densenet_model.conv1,
                                  densenet_model.encoder2,
                                  densenet_model.encoder3,
                                  densenet_model.encoder4,
                                  densenet_model.encoder5,
                                  ReLU())

    # define the localizer model
    model = MaxPooledLocalizer(densenet_encoder, n_classes=N_CLASSES - 1, n_hidden_filters=1024)
    model = model.to(DEVICE)

    criterion1 = FocalSymmetricLovaszHardLogLoss()
    criterion2 = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

    best_loss = float('inf')
    logger = Logger(LOGGER_PATH, header=['epoch', 'train_loss', 'val_loss', 'val_bce_loss', 'val_focal_loss'])
    for epoch in range(N_EPOCHS):

        if epoch % 2 == 0:
            criterion = criterion1
        else:
            criterion = criterion2

        train_loss = train_epoch(model,
                                 train_loader,
                                 criterion,
                                 optimizer,
                                 DEVICE,
                                 clip_grad_value=1,
                                 progress=True,
                                 epoch=epoch,
                                 n_batches=N_TRAIN_BATCHES)

        val_loss, val_bce_loss, val_focal_loss = test_epoch(model,
                                                            val_loader,
                                                            criterion1,
                                                            DEVICE,
                                                            calc_bce=True,
                                                            calc_focal=True,
                                                            progress=True,
                                                            epoch=epoch,
                                                            n_batches=N_VAL_BATCHES)

        logger.add_entry(epoch, train_loss, val_loss, val_bce_loss, val_focal_loss)

        # checkpoint all epochs for now
        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
