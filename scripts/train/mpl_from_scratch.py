import os
from yaml import safe_load

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from torch.nn import BCEWithLogitsLoss, Conv2d, Sequential
from torch.optim import Adam
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from hpa.data import HPADataset, N_CLASSES
from hpa.data.transforms import HPACompose
from hpa.model.localizers import MaxPooledLocalizer
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch, test_epoch

if __name__ == '__main__':
    print('Training a weakly-supervised classifier from scratch')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/max-pooled-sigmoid/from-scratch-0.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    img_dim = config['data']['image_size']
    transform_fn = HPACompose([
        A.Resize(img_dim, img_dim),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=(0,), std=(1,), max_pixel_value=255),
        ToTensorV2()
    ])

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'train')
    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'train-index.csv'))
    # train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'train-sample.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'val-index.csv'))
    # val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'val-sample.csv'))

    train_data = HPADataset(train_idx, DATA_DIR, transforms=transform_fn)
    val_data = HPADataset(val_idx, DATA_DIR, transforms=transform_fn)

    BATCH_SIZE = config['data']['batch_size']
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------
    DEVICE = 'cuda'
    LR = config['model']['lr']
    N_EPOCHS = config['model']['epochs']

    resnet_model = resnet50()
    resnet_model.conv1 = Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base_model = Sequential(*list(resnet_model.children())[:-2])
    model = MaxPooledLocalizer(base_model, n_classes=N_CLASSES - 1)
    model = model.to(DEVICE)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # -------------------------------------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------------------------------------
    LOGGER_PATH = config['logger_path']
    CHECKPOINT_DIR = config['checkpoint_dir']
    create_folder(os.path.dirname(LOGGER_PATH))
    create_folder(os.path.dirname(CHECKPOINT_DIR))

    N_TRAIN_BATCHES = int(len(train_data) / BATCH_SIZE)
    N_VAL_BATCHES = int(len(val_data) / BATCH_SIZE)

    best_loss = float('inf')
    logger = Logger(LOGGER_PATH, header=['epoch', 'train_loss', 'val_loss'])
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model,
                                 train_loader,
                                 criterion,
                                 optimizer,
                                 DEVICE,
                                 progress=True,
                                 epoch=epoch,
                                 n_batches=N_TRAIN_BATCHES)

        val_loss = test_epoch(model,
                              val_loader,
                              criterion,
                              DEVICE,
                              progress=True,
                              epoch=epoch,
                              n_batches=N_VAL_BATCHES)

        logger.add_entry(epoch, train_loss, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            filepath = os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth')
            checkpoint(model, filepath)
