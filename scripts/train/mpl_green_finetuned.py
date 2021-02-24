import os
from yaml import safe_load

import albumentations as A
import pandas as pd
import torch
from torch.nn import BCELoss, ReLU, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYWithGreenTarget, N_CHANNELS, N_CLASSES
from hpa.data.transforms import HPACompose
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.localizers import MaxPooledLocalizer
from hpa.model.loss import FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch_with_segmentation, test_epoch_with_segmentation

if __name__ == '__main__':
    print('Training a weakly-supervised max-pooled localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/max-pooled-sigmoid/finetuned-seg-1.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    img_dim = config['data']['image_size']

    dual_train_transform_fn = HPACompose([
        A.Resize(img_dim, img_dim),
        A.Flip(p=0.5),
    ])

    dual_val_transform_fn = A.Resize(img_dim, img_dim)

    img_transform_fn = A.Normalize(
        mean=[0.074598, 0.050630, 0.050891, 0.076287],
        std=[0.122813, 0.085745, 0.129882, 0.119411],
        max_pixel_value=255
    )

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'train')
    SEG_DIR = config['data']['seg_dir']

    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'train-index.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'val-index.csv'))

    train_data = RGBYWithGreenTarget(train_idx,
                                     DATA_DIR,
                                     SEG_DIR,
                                     dual_train_transform_fn,
                                     img_transform_fn,
                                     tensorize=True)

    val_data = RGBYWithGreenTarget(val_idx,
                                   DATA_DIR,
                                   SEG_DIR,
                                   dual_val_transform_fn,
                                   img_transform_fn,
                                   tensorize=True)

    BATCH_SIZE = config['data']['batch_size']
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # -------------------------------------------------------------------------------------------
    # Prepare the model
    # -------------------------------------------------------------------------------------------
    DEVICE = 'cuda'
    LR = config['model']['lr']
    N_EPOCHS = config['model']['epochs']
    PRETRAINED_PATH = config['pretrained_path']

    # load the pretrained DenseNet model
    pretrained_state_dict = torch.load(PRETRAINED_PATH)['state_dict']
    densenet_model = DensenetClass(in_channels=N_CHANNELS, dropout=True)
    densenet_model.load_state_dict(pretrained_state_dict)

    # isolate the CNN encoder
    densenet_encoder = Sequential(densenet_model.conv1,
                                  densenet_model.encoder2,
                                  densenet_model.encoder3,
                                  densenet_model.encoder4,
                                  densenet_model.encoder5,
                                  ReLU())

    # define the localizer model
    model = MaxPooledLocalizer(densenet_encoder,
                               n_classes=N_CLASSES - 1,
                               n_hidden_filters=1024,
                               merge_classes=True,
                               seg_shape=(img_dim, img_dim))
    model = model.to(DEVICE)

    classify_criterion = FocalSymmetricLovaszHardLogLoss()
    segment_criterion = BCELoss()
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

    W_CLASSIFY = config['model']['classify_weight']
    W_SEGMENT = config['model']['segment_weight']

    header = [
        'epoch',
        'train_loss',
        'train_classify_loss',
        'train_segment_loss',
        'val_loss',
        'val_classify_loss',
        'val_segment_loss',
        'val_bce_loss',
        'val_focal_loss'
    ]
    logger = Logger(LOGGER_PATH, header=header)

    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        train_results = train_epoch_with_segmentation(model,
                                                      train_loader,
                                                      classify_criterion,
                                                      segment_criterion,
                                                      optimizer,
                                                      DEVICE,
                                                      W_CLASSIFY,
                                                      W_SEGMENT,
                                                      clip_grad_value=1,
                                                      progress=True,
                                                      epoch=epoch,
                                                      n_batches=N_TRAIN_BATCHES)

        val_results = test_epoch_with_segmentation(model,
                                                   val_loader,
                                                   classify_criterion,
                                                   segment_criterion,
                                                   DEVICE,
                                                   W_CLASSIFY,
                                                   W_SEGMENT,
                                                   calc_bce=True,
                                                   calc_focal=True,
                                                   progress=True,
                                                   epoch=epoch,
                                                   n_batches=N_VAL_BATCHES)

        logger.add_entry(epoch, *train_results, *val_results)

        # checkpoint all epochs for now
        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
