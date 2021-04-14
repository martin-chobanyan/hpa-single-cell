from argparse import ArgumentParser

import albumentations as A
from albumentations.pytorch import ToTensorV2
from cv2 import resize
import numpy as np
import pandas as pd
from tqdm import tqdm

from hpa.data import CHANNEL_MEANS, CHANNEL_STDS
from hpa.data.dataset import load_channels, get_label_vector
from hpa.data.misc import parse_string_label
from hpa.model.bestfitting.densenet import DensenetClass
from hpa.model.localizers import *


def map_to_new_classes(probs):
    return torch.cat([
        probs[:, :8],
        probs[:, [11]],
        probs[:, [12, 13]].max(dim=1, keepdim=True).values,
        probs[:, [14, 17, 19]],
        probs[:, [21, 22]].max(dim=1, keepdim=True).values,
        probs[:, [23, 24, 25]],
        probs[:, [8, 9, 10, 20, 26]].max(dim=1, keepdim=True).values
    ], dim=-1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    MODEL_PATH = args.model
    OUTPUT_PATH = args.out
    DATA_IDX_PATH = '/home/mchobanyan/data/kaggle/hpa-single-cell/splits/complete-data-idx.csv'

    DEVICE = 'cuda'
    IMG_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/train/'
    IMG_DIM = 1536
    BATCH_SIZE = 12

    model_state = torch.load(MODEL_PATH, map_location=DEVICE)['state_dict']
    densenet_model = DensenetClass(in_channels=4, dropout=True)
    densenet_model.load_state_dict(model_state)
    model = densenet_model.to(DEVICE)
    model = model.eval()

    df = pd.read_csv(DATA_IDX_PATH)

    img_ids = []
    img_labels = []
    img_sources = []
    probs_list = []

    batch = []
    normalize_fn = A.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS, max_pixel_value=255)
    for _, (img_id, labels, src) in tqdm(df.iterrows(), total=len(df)):

        # prepare the labels
        label_vec = get_label_vector(labels)
        label_vec = torch.from_numpy(label_vec)
        label_vec = label_vec.unsqueeze(0)
        label_vec = label_vec.to(DEVICE)
        labels = sorted(parse_string_label(labels))

        # load and prepare the image
        channels = load_channels(img_id, IMG_DIR)
        img_full = np.dstack([channels['red'], channels['green'], channels['blue'], channels['yellow']])
        img_full = img_full.astype(np.float32)
        img_reduced = resize(img_full, (IMG_DIM, IMG_DIM))
        x = normalize_fn(image=img_reduced)['image']
        x = ToTensorV2()(image=x)['image']
        batch.append(x)

        if len(batch) == BATCH_SIZE:
            batch = torch.stack(batch)
            batch = batch.to(DEVICE)
            with torch.no_grad():
                logits = model(batch)
                class_probs = torch.sigmoid(logits)
                class_probs = map_to_new_classes(class_probs)
                class_probs = class_probs.cpu().tolist()
            probs_list += class_probs
            batch = []

        img_ids.append(img_id)
        img_labels.append(','.join([str(l) for l in labels]))
        img_sources.append(src)

    # take care of any stragglers
    if len(batch) > 0:
        batch = torch.stack(batch)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
            class_probs = torch.sigmoid(logits)
            class_probs = map_to_new_classes(class_probs)
            class_probs = class_probs.cpu().tolist()
        probs_list += class_probs

    metrics_df = pd.DataFrame({
        'ID': img_ids,
        'Label': img_labels,
        'Source': img_sources
    })

    # group probabilities by their class
    probs_list = list(zip(*probs_list))
    for label_id, probs in enumerate(probs_list):
        metrics_df[f'label{label_id}'] = probs

    metrics_df.to_csv(OUTPUT_PATH, sep='\t', index=False)
