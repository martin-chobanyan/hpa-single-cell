import os

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # --------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------
    ROOT_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/'
    INDEX_PATH = os.path.join(ROOT_DIR, 'train.csv')
    TRAIN_OUTPUT_PATH = os.path.join(ROOT_DIR, 'train-index.csv')
    VAL_OUTPUT_PATH = os.path.join(ROOT_DIR, 'val-index.csv')

    # --------------------------------------------------------------
    # Split the data
    # --------------------------------------------------------------
    data_idx = pd.read_csv(INDEX_PATH)
    train_idx, val_idx = train_test_split(data_idx, test_size=0.2, shuffle=True, random_state=0)
    train_idx.to_csv(TRAIN_OUTPUT_PATH, index=False)
    val_idx.to_csv(VAL_OUTPUT_PATH, index=False)
