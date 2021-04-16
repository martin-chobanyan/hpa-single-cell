"""Create a new dataset with a different size and new segmentations"""
import os

from cv2 import resize, INTER_NEAREST
import numpy as np
import pandas as pd
from PIL import Image
from torch import no_grad
from tqdm import tqdm

from hpa.data import load_channels
from hpa.segment import HPACellSegmenter
from hpa.utils import create_folder


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------------------------------------------------------
    IMG_DIM = 1536
    ROOT_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/'
    INDEX_PATH = os.path.join(ROOT_DIR, 'splits', 'complete-data-idx.csv')

    NUCLEI_PATH = os.path.join(ROOT_DIR, 'misc', 'hpa-cell-seg-weights', 'dpn_unet_nuclei_v1.pth')
    CELL_PATH = os.path.join(ROOT_DIR, 'misc', 'hpa-cell-seg-weights', 'dpn_unet_cell_3ch_v1.pth')

    COMPETITION_IMG_DIR = os.path.join(ROOT_DIR, 'train')
    PUBLIC_IMG_DIR = os.path.join(ROOT_DIR, 'misc', 'public-hpa', 'data2')

    COMPETITION_OUT_DIR = os.path.join(ROOT_DIR, 'images', 'competition_1536x1536')
    PUBLIC_OUT_DIR = os.path.join(ROOT_DIR, 'images', 'public_1536x1536')

    COMPETITION_SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'competition_1536x1536')
    PUBLIC_SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'public_1536x1536')

    create_folder(COMPETITION_OUT_DIR)
    create_folder(PUBLIC_OUT_DIR)

    create_folder(COMPETITION_SEG_DIR)
    create_folder(PUBLIC_SEG_DIR)

    # ------------------------------------------------------------------------------------------------------------------
    # Resize and segment the images
    # ------------------------------------------------------------------------------------------------------------------
    data_idx = pd.read_csv(INDEX_PATH)
    segmenter = HPACellSegmenter(NUCLEI_PATH, CELL_PATH, device='cuda')

    print(f'Resizing images to {IMG_DIM}x{IMG_DIM}')
    for _, (img_id, _, src) in tqdm(data_idx.iterrows(), total=len(data_idx)):
        if src == 'competition':
            img_dir = COMPETITION_IMG_DIR
            out_dir = COMPETITION_OUT_DIR
            seg_dir = COMPETITION_SEG_DIR
        elif src == 'external':
            img_dir = PUBLIC_IMG_DIR
            out_dir = PUBLIC_OUT_DIR
            seg_dir = PUBLIC_SEG_DIR
        else:
            raise ValueError(f'Unknown image source: {src}')

        # resize each channel and store them in their respective output directory
        channels = load_channels(img_id, img_dir)
        for color, channel in channels.items():
            resized_channel = resize(channel, (IMG_DIM, IMG_DIM))
            resized_channel = Image.fromarray(resized_channel)
            resized_channel.save(os.path.join(out_dir, f'{img_id}_{color}.png'))

        # create the segmentation
        with no_grad():
            cell_seg = segmenter(channels['red'], channels['yellow'], channels['blue'])
        cell_seg = resize(cell_seg, (IMG_DIM, IMG_DIM), interpolation=INTER_NEAREST)
        np.savez_compressed(os.path.join(seg_dir, img_id), cell_seg)

    print('Done!\n')
    print(f'Resized competition images can be found in:\n{COMPETITION_OUT_DIR}\n')
    print(f'Resized public HPA images can be found in:\n{PUBLIC_OUT_DIR}\n')
    print(f'Competition segmentations can be found in:\n{COMPETITION_OUT_DIR}\n')
    print(f'Public HPA segmentations can be found in:\n{PUBLIC_OUT_DIR}\n')
