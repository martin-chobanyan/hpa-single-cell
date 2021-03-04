from random import randint

from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler


class CropDataSampler(BatchSampler):
    def __init__(self, crop_dataset, batch_size, crop_range=(512, 768), shuffle=True, drop_last=False):
        """

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
