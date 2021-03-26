import os
from yaml import safe_load

import albumentations as A
import pandas as pd
import torch
from torch.nn import L1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hpa.data import RGBYWithSegmentation, N_CHANNELS, CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.transforms import HPACompose, ToBinaryCellSegmentation
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.localizers import DecomposedDensenet, PooledLocalizer
from hpa.model.loss import ClassHeatmapLoss, FocalSymmetricLovaszHardLogLoss
from hpa.utils import create_folder
from hpa.utils.train import checkpoint, Logger, train_epoch_with_seg, test_epoch_with_seg

if __name__ == '__main__':
    print('Training a weakly-supervised max-pooled localizer with pretrained encoder')

    # -------------------------------------------------------------------------------------------
    # Read in the config
    # -------------------------------------------------------------------------------------------
    CONFIG_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/configs/decomposed/seg/decomposed0.yaml'
    with open(CONFIG_PATH, 'r') as file:
        config = safe_load(file)

    # -------------------------------------------------------------------------------------------
    # Prepare the augmentations
    # -------------------------------------------------------------------------------------------
    IMG_DIM = config['data']['image_size']
    MIN_BLUR = config['data']['min_blur']
    MAX_BLUR = config['data']['max_blur']
    HEATMAP_SCALE = config['data']['heatmap_scale']
    HEATMAP_DIM = int(IMG_DIM / HEATMAP_SCALE)

    dual_train_transform_fn = HPACompose([
        A.Flip(p=0.5)
    ])

    img_transform_fn = HPACompose([
        A.Resize(IMG_DIM, IMG_DIM),
        A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255)
    ])

    seg_transform_fn = HPACompose([
        ToBinaryCellSegmentation(),
        A.Blur(blur_limit=(MIN_BLUR, MAX_BLUR), p=1.0),
        A.Resize(HEATMAP_DIM, HEATMAP_DIM)
    ])

    # -------------------------------------------------------------------------------------------
    # Prepare the data
    # -------------------------------------------------------------------------------------------
    ROOT_DIR = config['data']['root_dir']
    DATA_DIR = os.path.join(ROOT_DIR, 'train')
    SEG_DIR = config['data']['seg_dir']
    NUM_WORKERS = 12

    train_idx = pd.read_csv(os.path.join(ROOT_DIR, 'train-index.csv'))
    val_idx = pd.read_csv(os.path.join(ROOT_DIR, 'val-index.csv'))

    # train_idx = train_idx.head(64)
    # val_idx = val_idx.head(64)

    train_data = RGBYWithSegmentation(data_idx=train_idx,
                                      data_dir=DATA_DIR,
                                      seg_dir=SEG_DIR,
                                      dual_transforms=dual_train_transform_fn,
                                      img_transforms=img_transform_fn,
                                      seg_transforms=seg_transform_fn,
                                      tensorize=True)

    val_data = RGBYWithSegmentation(data_idx=val_idx,
                                    data_dir=DATA_DIR,
                                    seg_dir=SEG_DIR,
                                    img_transforms=img_transform_fn,
                                    seg_transforms=seg_transform_fn,
                                    tensorize=True)

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
    decomposed_model = DecomposedDensenet(densenet_model=densenet_model, max_classes=True)

    # define the localizer model
    model = PooledLocalizer(cnn=decomposed_model, pool='max', return_maps=True)
    model = model.to(DEVICE)

    classify_criterion = FocalSymmetricLovaszHardLogLoss()
    segment_criterion = ClassHeatmapLoss(heatmap_loss_fn=L1Loss())
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

    # starting hyperparameter weights (full classify, no segment)
    w_classify = 1.0
    w_segment = 0.0

    HEADER = [
        'epoch',
        'train_loss',
        'train_classify_loss',
        'train_segment_loss',
        'val_loss',
        'val_classify_loss',
        'val_segment_loss',
        'val_bce_loss',
        'val_focal_loss',
        'val_exact',
        'val_f1'
    ]
    logger = Logger(LOGGER_PATH, header=HEADER)

    best_loss = float('inf')
    for epoch in range(N_EPOCHS):

        if epoch == 5:
            print('Changing hyperparameters')
            w_classify = 0.75
            w_segment = 0.25

        if epoch == 10:
            print('Changing hyperparameters')
            w_classify = 0.5
            w_segment = 0.5

        train_results = train_epoch_with_seg(model=model,
                                             dataloader=train_loader,
                                             classify_criterion=classify_criterion,
                                             segment_criterion=segment_criterion,
                                             optimizer=optimizer,
                                             device=DEVICE,
                                             w_classify=w_classify,
                                             w_segment=w_segment,
                                             clip_grad_value=1,
                                             progress=True,
                                             epoch=epoch,
                                             n_batches=N_TRAIN_BATCHES)

        val_results = test_epoch_with_seg(model=model,
                                          dataloader=val_loader,
                                          classify_criterion=classify_criterion,
                                          segment_criterion=segment_criterion,
                                          device=DEVICE,
                                          w_classify=w_classify,
                                          w_segment=w_segment,
                                          calc_bce=True,
                                          calc_focal=True,
                                          progress=True,
                                          epoch=epoch,
                                          n_batches=N_VAL_BATCHES)

        logger.add_entry(epoch, *train_results, *val_results)

        # checkpoint all epochs for now
        checkpoint(model, os.path.join(CHECKPOINT_DIR, f'model{epoch}.pth'))
