"""DataLoader and Sampler utilities"""

from itertools import cycle
from random import randint

import numpy as np
import torch
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from torch.utils.data.dataloader import default_collate

from .misc import get_single_label_subset
from .dataset import N_CLASSES


class AlternatingDataLoader:
    """Cycle between different DataLoader instances for each epoch"""

    def __init__(self, loaders):
        """Initialization
        Parameters
        ----------
        loaders: list[torch.utils.data.DataLoader]
        """
        self.loaders = cycle(loaders)

    def __iter__(self):
        return iter(next(self.loaders))


class SingleLabelBatchSampler:
    def __init__(self, dataset, batch_size, num_batches, crop_range=(256, 768)):
        """Initialization

        Parameters
        ----------
        dataset: hpa.data.dataset.CroppedRGBYDataset
        batch_size: int
        num_batches: int
        crop_range: tuple[int], optional
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.crop_range = crop_range

        self.unique_label_ids = list(str(i) for i in range(N_CLASSES))
        # self.unique_label_ids = [str(i) for i in [0, 5, 10, 8, 12]]
        self.single_label_idx = get_single_label_subset(self.dataset.data_idx)

    def sample_label_id(self):
        return np.random.choice(self.unique_label_ids)

    def sample_idx_for_label(self, label_id):
        label_df = self.single_label_idx.loc[self.single_label_idx['Label'] == label_id]
        return np.random.choice(label_df.index.values, self.batch_size, replace=True)

    def sample_crop_size(self):
        return randint(*self.crop_range)

    def set_random_crop_size(self):
        crop_size = self.sample_crop_size()
        self.dataset.set_crop_size(crop_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            self.set_random_crop_size()
            label_id = self.sample_label_id()
            yield self.sample_idx_for_label(label_id)

    def __len__(self):
        return self.num_batches


class CropDataSampler(BatchSampler):
    def __init__(self, crop_dataset, batch_size, crop_range=(512, 768), shuffle=True, drop_last=False):
        """Initialization

        Parameters
        ----------
        crop_dataset: hpa.data.dataset.CroppedRGBYDataset
        batch_size: int
        crop_range: tuple[int], optional
        shuffle: bool, optional
        drop_last: bool, optional
        """
        if shuffle:
            sampler = RandomSampler(crop_dataset)
        else:
            sampler = SequentialSampler(crop_dataset)

        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        self.crop_range = crop_range
        self.dataset = crop_dataset

    def sample_crop_size(self):
        return randint(*self.crop_range)

    def set_random_crop_size(self):
        crop_size = self.sample_crop_size()
        self.dataset.set_crop_size(crop_size)

    def __iter__(self):
        self.set_random_crop_size()
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                self.set_random_crop_size()
        if len(batch) > 0 and not self.drop_last:
            yield batch


def cell_mask_collate(batch):
    """Collate a batch containing a list of tuples with the (image, cell masks, cell count, and label vector)

    Parameters
    ----------
    batch: list[tuple]

    Returns
    -------
    tuple[torch.Tensor]
    """
    # isolate each component in the batch
    images, cell_masks, cell_counts, label_vectors = list(zip(*batch))

    # apply default collate function to the images, cell counts per image, and label vectors
    batch_img, batch_cell_counts, batch_label = default_collate(list(zip(images, cell_counts, label_vectors)))

    # concatenate the non-null cell masks along the mask (first) dimension
    batch_cell_masks = []
    for mask in cell_masks:
        if mask is not None:
            batch_cell_masks.append(mask)
    try:
        batch_cell_masks = np.concatenate(batch_cell_masks, axis=0)
        batch_cell_masks = torch.as_tensor(batch_cell_masks)
    except ValueError:
        batch_cell_masks = torch.zeros((0, 0, 0))
    return batch_img, batch_cell_masks, batch_cell_counts, batch_label
