import os

from imageio import imread
import numpy as np
from torch.utils.data import Dataset

N_CLASSES = 19


def load_channels(img_id, img_dir):
    imgs = dict()
    for color in ('blue', 'green', 'red', 'yellow'):
        filename = f'{img_id}_{color}.png'
        filepath = os.path.join(img_dir, filename)
        imgs[color] = imread(filepath)
    return imgs


class HPADataset(Dataset):
    def __init__(self, train_idx, data_dir, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        """
        super().__init__()
        self.train_idx = train_idx
        self.data_dir = data_dir
        self.transforms = transforms
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
        image_id, labels = self.train_idx.loc[item, ['ID', 'Label']]

        # load and stack the images
        channels = load_channels(image_id, self.data_dir)
        image = np.stack([channels['green'], channels['blue'], channels['red'], channels['yellow']])
        image = image.astype(np.float32)

        # define a binary vector for the labels
        labels = labels.split('|')
        label_vec = np.zeros(N_CLASSES - 1, dtype=np.int64)
        for i in labels:
            label_vec[int(i)] = 1

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label_vec

    def __len__(self):
        return self.n_samples
