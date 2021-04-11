"""Stratified Multi-Label Split of the Data"""

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from hpa.data.dataset import get_label_vector

if __name__ == '__main__':
    # Constants
    RANDOM_SEED = 0
    P_VALIDATION = 0.2

    # Filepaths
    FULL_DATA_IDX_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/splits/complete-data-idx.csv'
    TRAIN_OUTPUT_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/splits/joint/stratified/train-idx.csv'
    VAL_OUTPUT_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/splits/joint/stratified/val-idx.csv'

    # Seed, load, and shuffle the data
    np.random.seed(RANDOM_SEED)
    complete_df = pd.read_csv(FULL_DATA_IDX_PATH)
    complete_df = complete_df.sample(frac=1).reset_index(drop=True)

    # Create the multi-label vectors
    label_vectors = []
    for label in complete_df['Label']:
        label_vec = get_label_vector(label)
        label_vectors.append(label_vec)
    label_vectors = np.stack(label_vectors).astype(np.int64)

    # stratified multi-label split
    idx = complete_df.index.values.reshape((-1, 1))
    train_idx, y_train, val_idx, y_val = iterative_train_test_split(idx, label_vectors, test_size=P_VALIDATION)

    # retrieve the train and validation DataFrames
    train_df = complete_df.loc[train_idx.squeeze()]
    val_df = complete_df.loc[val_idx.squeeze()]

    # save the results
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_csv(VAL_OUTPUT_PATH, index=False)
