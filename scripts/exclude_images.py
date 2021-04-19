from multiprocessing import Pool
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_segmentation(filepath):
    return np.load(filepath)['arr_0']


def check_seg(filepath):
    seg = read_segmentation(filepath)
    if seg.max() == 0:
        return filepath
    return None


if __name__ == '__main__':
    ROOT_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/'
    COMPETITION_SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'competition_1536x1536')
    EXTERNAL_SEG_DIR = os.path.join(ROOT_DIR, 'segmentation', 'public_1536x1536')
    DATA_IDX_PATH = os.path.join(ROOT_DIR, 'splits', 'complete-data-idx.csv')
    NUM_CORES = 8

    data_idx = pd.read_csv(DATA_IDX_PATH)

    seg_filepaths = []
    for _, (img_id, _, src) in data_idx.iterrows():
        seg_dir = COMPETITION_SEG_DIR if src == 'competition' else EXTERNAL_SEG_DIR
        seg_filepaths.append(os.path.join(seg_dir, f'{img_id}.npz'))

    empty_seg_filepaths = []
    with Pool(NUM_CORES) as pool, tqdm(total=len(seg_filepaths)) as pbar:
        for result in pool.imap_unordered(check_seg, seg_filepaths):
            if result is not None:
                empty_seg_filepaths.append(result)
            pbar.update()

    data_idx['include'] = True
    img_id_to_idx = {img_id: i for i, img_id in data_idx['ID'].items()}
    for filepath in empty_seg_filepaths:
        img_id, _ = os.path.splitext(os.path.basename(filepath))
        i = img_id_to_idx[img_id]
        data_idx.loc[i, 'include'] = False

    data_idx.to_csv(DATA_IDX_PATH, index=False)
