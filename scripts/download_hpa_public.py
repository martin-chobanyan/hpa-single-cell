import io
import os
import requests
import pathlib
import gzip
import shutil

import imageio
import pandas as pd
from tqdm import tqdm


def download_jpg(url, target_path):
    r = requests.get(url, stream=True)
    r.raw.decode_content = True
    with open(target_path, 'wb') as file:
        shutil.copyfileobj(r.raw, file)


def tif_gzip_to_png(tif_path):
    """Function to convert .tif.gz to .png and put it in the same folder"""
    png_path = pathlib.Path(tif_path.replace('.tif.gz', '.png'))
    tf = gzip.open(tif_path).read()
    img = imageio.imread(tf, 'tiff')
    imageio.imwrite(png_path, img)


def download_and_convert_tifgzip_to_png(url, target_path):
    """Function to convert .tif.gz to .png and put it in the same folder"""
    r = requests.get(url)
    f = io.BytesIO(r.content)
    tf = gzip.open(f).read()
    img = imageio.imread(tf, 'tiff')
    imageio.imwrite(target_path, img)


if __name__ == '__main__':

    SAVE_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/misc/public-hpa/data2/'
    COLORS = ['blue', 'red', 'green', 'yellow']

    CLASS_NAMES = [
        'Nucleoplasm',
        'Nuclear membrane',
        'Nucleoli',
        'Nucleoli fibrillar center',
        'Nuclear speckles',
        'Nuclear bodies',
        'Endoplasmic reticulum',
        'Golgi apparatus',
        'Intermediate filaments',
        'Actin filaments ',
        'Microtubules',
        'Mitotic spindle',
        'Centrosome',
        'Plasma membrane',
        'Mitochondria',
        'Aggresome',
        'Cytosol',
        'Vesicles and punctate cytosolic patterns',
        'Negative'
    ]
    CLASS_NAMES = [name.lower().strip() for name in CLASS_NAMES]
    class_to_label_id = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

    data_idx = pd.read_csv('/home/mchobanyan/data/kaggle/hpa-single-cell/misc/public-hpa/kaggle_2021.csv')
    data_idx['Label'] = data_idx['Label'].str.lower()
    print(f'Total images: {len(data_idx)}')

    celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30',
                 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
    data_idx = data_idx[data_idx['Cellline'].isin(celllines)]
    print(f'Images with appropriate cell-lines: {len(data_idx)}')

    # group/standardize the class names
    data_idx['Label'] = data_idx['Label'].str.replace('focal adhesion sites', 'actin filaments')
    data_idx['Label'] = data_idx['Label'].str.replace('cell junctions', 'plasma membrane')
    data_idx['Label'] = data_idx['Label'].str.replace('centriolar satellite', 'centrosome')
    data_idx['Label'] = data_idx['Label'].str.replace('no staining', 'negative')

    data_idx['Label'] = data_idx['Label'].str.replace('peroxisomes', 'vesicles')
    data_idx['Label'] = data_idx['Label'].str.replace('endosomes', 'vesicles')
    data_idx['Label'] = data_idx['Label'].str.replace('lysosomes', 'vesicles')
    data_idx['Label'] = data_idx['Label'].str.replace('lipid droplets', 'vesicles')
    data_idx['Label'] = data_idx['Label'].str.replace('cytoplasmic bodies', 'vesicles')
    data_idx['Label'] = data_idx['Label'].str.replace('vesicles', 'vesicles and punctate cytosolic patterns')

    # find class names not included
    extra_classes = set()
    for label in data_idx['Label']:
        tokens = label.split(',')
        for token in tokens:
            if token not in class_to_label_id:
                extra_classes.add(token)

    # throw away all rows containing the extra classes
    extra_mask = None
    for class_name in extra_classes:
        mask = data_idx['Label'].str.contains(class_name)
        if extra_mask is None:
            extra_mask = mask
        else:
            extra_mask = extra_mask | mask
    data_idx = data_idx.loc[~extra_mask]
    data_idx = data_idx.reset_index(drop=True)
    print(f'Dropped {extra_mask.sum()} rows due to extra classes')
    print(f'Images remaining: {len(data_idx)}')

    # transform the label column to label ids
    numeric_labels = []
    for label in data_idx['Label']:
        label_ids = set()
        tokens = label.split(',')
        for token in tokens:
            label_ids.add(class_to_label_id[token])
        label_ids = sorted(label_ids)
        label_ids = [str(i) for i in label_ids]
        numeric_labels.append('|'.join(label_ids))
    data_idx['Label'] = numeric_labels

    # no more than 5000 samples per unique label combo
    data_idx = data_idx.groupby('Label').head(5000)
    print(f'Removing examples from abundant classes...')

    # keep only 1000 samples for labels 0 and 16 (the most popular ones)
    drop_idx = data_idx.loc[data_idx['Label'] == '16'].sample(4000).index
    data_idx = data_idx.drop(drop_idx, axis='index')

    drop_idx = data_idx.loc[data_idx['Label'] == '0'].sample(4000).index
    data_idx = data_idx.drop(drop_idx, axis='index')

    # keep only 1000 samples for label 0|16
    drop_idx = data_idx.loc[data_idx['Label'] == '0|16'].sample(4000).index
    data_idx = data_idx.drop(drop_idx, axis='index')
    print(f'Images remaining: {len(data_idx)}')

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for base_url in tqdm(data_idx['Image'], desc='Downloading data'):
        try:
            for color in COLORS:
                img_url = f'{base_url}_{color}.tif.gz'
                save_path = os.path.join(SAVE_DIR, f'{os.path.basename(base_url)}_{color}.png')
                download_and_convert_tifgzip_to_png(img_url, save_path)
        except:
            print(f'failed to download: {base_url}')
