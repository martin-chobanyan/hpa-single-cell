import os

from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
N_CLASSES = 19


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
        label_vec = np.zeros(N_CLASSES - 1, dtype=np.int64)
        labels = [int(i) for i in labels.split('|')]
        for i in labels:
            if i != 18:
                label_vec[i] = 1

        return channels, label_vec

    def __len__(self):
        return self.n_samples


class HPADataset(BaseDataset):
    def __init__(self, train_idx, data_dir, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        transforms: callable
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
        image = np.stack([channels['green'], channels['red'], channels['yellow'], channels['blue']])
        image = image.transpose((1, 2, 0))
        image = image.astype(np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label_vec


class IsolatedTargetDataset(BaseDataset):
    def __init__(self, train_idx, data_dir, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        transforms: callable
        """
        super().__init__(train_idx, data_dir)
        self.transforms = transforms

    def __getitem__(self, item):
        channels, label_vec = super().__getitem__(item)

        # prepare the target protein image
        tgt_img = channels['green']
        tgt_img = tgt_img.astype(np.float32)
        tgt_img = np.expand_dims(tgt_img, axis=0)

        # prepare the reference image (microtuble, er, nuclei)
        ref_img = np.stack([channels['red'], channels['yellow'], channels['blue']])
        ref_img = ref_img.astype(np.float32)

        if self.transforms is not None:
            tgt_img = self.transforms(tgt_img)
            ref_img = self.transforms(ref_img)
        return tgt_img, ref_img

    def __len__(self):
        return
