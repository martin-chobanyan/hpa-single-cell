import os

from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset

from .misc import parse_string_label

ImageFile.LOAD_TRUNCATED_IMAGES = True
N_CLASSES = 19
NEGATIVE_LABEL = N_CLASSES - 1


def load_channels(img_id, img_dir):
    imgs = dict()
    for color in ('blue', 'green', 'red', 'yellow'):
        filename = f'{img_id}_{color}.png'
        filepath = os.path.join(img_dir, filename)
        imgs[color] = np.array(Image.open(filepath))
    return imgs


class BaseDataset(Dataset):
    def __init__(self, train_idx, data_dir):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        """
        super().__init__()
        self.data_idx = train_idx
        self.data_dir = data_dir
        self.n_samples = len(train_idx)

    def __getitem__(self, item):
        """Retrieve a multichannel image

        Parameters
        ----------
        item: int

        Returns
        -------
        An array or tensor of the four image filters
        """
        image_id, labels = self.data_idx.loc[item, ['ID', 'Label']]

        # load and stack the images
        channels = load_channels(image_id, self.data_dir)

        # define a binary vector for the labels
        label_vec = np.zeros(N_CLASSES - 1, dtype=np.float32)
        labels = parse_string_label(labels)
        for i in labels:
            if i != 18:
                label_vec[i] = 1

        return channels, label_vec

    def __len__(self):
        return self.n_samples


class RGBYDataset(BaseDataset):
    def __init__(self, train_idx, data_dir, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.HPACompose
        """
        super().__init__(train_idx, data_dir)
        self.transforms = transforms

    def __getitem__(self, item):
        """Retrieve a multichannel image

        Parameters
        ----------
        item: int

        Returns
        -------
        An array or tensor of the four image filters
        """
        channels, label_vec = super().__getitem__(item)

        # stack the channels as RGBY
        img = np.dstack([channels['red'], channels['green'], channels['blue'], channels['yellow']])

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        # convert the data types to floats for all channels
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
        elif isinstance(img, torch.Tensor):
            img = img.float()

        return img, label_vec


class IsolatedTargetDataset(BaseDataset):
    def __init__(self, train_idx, data_dir, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.HPACompose
        """
        super().__init__(train_idx, data_dir)
        self.transforms = transforms

    def __getitem__(self, item):
        channels, label_vec = super().__getitem__(item)

        # prepare the target protein image
        tgt_img = channels['green']

        # prepare the reference image (microtuble + er + nuclei)
        ref_img = np.dstack([channels['red'], channels['yellow'], channels['blue']])

        if self.transforms is not None:
            aug_dict = self.transforms(image=tgt_img, ref=ref_img)
            tgt_img = aug_dict['image']
            ref_img = aug_dict['ref']

        # convert the data types to floats for all channels
        if isinstance(tgt_img, np.ndarray):
            tgt_img = tgt_img.astype(np.float32)
            ref_img = ref_img.astype(np.float32)
        elif isinstance(tgt_img, torch.Tensor):
            tgt_img = tgt_img.float()
            ref_img = ref_img.float()

        return tgt_img, ref_img, label_vec


class HPADataset(IsolatedTargetDataset):
    def __getitem__(self, item):
        """Retrieve a multichannel image

        Parameters
        ----------
        item: int

        Returns
        -------
        An array or tensor of the four image filters
        """
        tgt_img, ref_img, label_vec = super().__getitem__(item)
        if isinstance(tgt_img, np.ndarray):
            img = np.concatenate([tgt_img, ref_img], axis=0)
        else:
            img = torch.cat([tgt_img, ref_img], dim=0)
        return img, label_vec
