import os

from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset

from .misc import parse_string_label

ImageFile.LOAD_TRUNCATED_IMAGES = True
N_CHANNELS = 4
N_CLASSES = 19
NEGATIVE_LABEL = N_CLASSES - 1


# TODO: move this to utils/misc
def load_channels(img_id, img_dir):
    imgs = dict()
    for color in ('blue', 'green', 'red', 'yellow'):
        filename = f'{img_id}_{color}.png'
        filepath = os.path.join(img_dir, filename)
        channel = np.array(Image.open(filepath))
        if channel.dtype == np.int32:
            channel = 255 * (channel / 65535)
            channel = channel.astype(np.uint8)
        imgs[color] = channel
    return imgs


# TODO: move this to utils/misc
def get_label_vector(labels):
    label_vec = np.zeros(N_CLASSES - 1, dtype=np.float32)
    labels = parse_string_label(labels)
    for i in labels:
        if i != 18:
            label_vec[i] = 1
    return label_vec


# TODO: move this to utils/misc
def get_single_label_subset(df):
    return df.loc[~df['Label'].str.contains('\|')]


class BaseDataset(Dataset):
    def __init__(self, train_idx, data_dir, external_data_dir=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        """
        super().__init__()
        self.data_idx = train_idx
        self.data_dir = data_dir
        self.external_data_dir = external_data_dir
        self.n_samples = len(train_idx)

    def get_img_id_and_label(self, item):
        return self.data_idx.loc[item, ['ID', 'Label']]

    def __getitem__(self, item):
        """Retrieve a multichannel image

        Parameters
        ----------
        item: int

        Returns
        -------
        An array or tensor of the four image filters
        """
        image_id, labels = self.get_img_id_and_label(item)

        if (self.external_data_dir is not None) and (self.data_idx.at[item, 'Source'] == 'external'):
            data_dir = self.external_data_dir
        else:
            data_dir = self.data_dir

        channels = load_channels(image_id, data_dir)
        label_vec = get_label_vector(labels)
        return image_id, channels, label_vec

    def __len__(self):
        return self.n_samples


class RGBYDataset(BaseDataset):
    def __init__(self, train_idx, data_dir, external_data_dir=None, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.HPACompose
        """
        super().__init__(train_idx, data_dir, external_data_dir)
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
        _, channels, label_vec = super().__getitem__(item)

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


# TODO: move this to loader.py module
class SingleLabelBatchSampler:
    def __init__(self, dataset, batch_size, num_batches):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

        # self.unique_label_ids = list(str(i) for i in range(N_CLASSES))
        self.unique_label_ids = list(str(i) for i in range(3))
        self.single_label_idx = get_single_label_subset(self.dataset.data_idx)

        # TODO: change this to be sampled before each batch
        self.crop_size = (224, 224)

    def sample_label_id(self):
        return np.random.choice(self.unique_label_ids)

    def sample_idx_for_label(self, label_id):
        label_df = self.single_label_idx.loc[self.single_label_idx['Label'] == label_id]
        return np.random.choice(label_df.index.values, self.batch_size, replace=False)

    def __iter__(self):
        for _ in range(self.num_batches):
            label_id = self.sample_label_id()
            yield self.sample_idx_for_label(label_id)
        
    def __len__(self):
        return self.num_batches


class RGBYWithSegmentation(BaseDataset):
    def __init__(self,
                 train_idx,
                 data_dir,
                 seg_dir,
                 external_data_dir=None,
                 dual_transforms=None,
                 img_transforms=None,
                 seg_transforms=None,
                 tensorize=True):
        super().__init__(train_idx, data_dir, external_data_dir)
        self.seg_dir = seg_dir

        self.dual_transforms = dual_transforms
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms

        if tensorize:
            self.tensorize = ToTensorV2()
        else:
            self.tensorize = None

    def __getitem__(self, item):
        img_id, channels, label_vec = super().__getitem__(item)

        # stack the channels as RGBY
        img = np.dstack([channels['red'], channels['green'], channels['blue'], channels['yellow']])

        # load the segmentation map
        seg = np.load(os.path.join(self.seg_dir, f'{img_id}.npz'))['arr_0']

        if self.dual_transforms is not None:
            aug_result = self.dual_transforms(image=img, mask=seg)
            img = aug_result['image']
            seg = aug_result['mask']

        if self.img_transforms is not None:
            img = self.img_transforms(image=img)['image']

        if self.seg_transforms is not None:
            seg = self.seg_transforms(image=seg)['image']

        # convert the data types to floats for all channels
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
            seg = seg.astype(np.float32)
        elif isinstance(img, torch.Tensor):
            img = img.float()
            seg = seg.float()

        if self.tensorize is not None:
            img = self.tensorize(image=img)['image']
            seg = self.tensorize(image=seg)['image']

        return img, seg, label_vec


class RGBYWithGreenTarget(BaseDataset):
    def __init__(self,
                 train_idx,
                 data_dir,
                 external_data_dir=None,
                 dual_transforms=None,
                 img_transforms=None,
                 tensorize=True):
        super().__init__(train_idx, data_dir, external_data_dir)
        self.dual_transforms = dual_transforms
        self.img_transforms = img_transforms
        if tensorize:
            self.tensorize = ToTensorV2()
        else:
            self.tensorize = None

    def __getitem__(self, item):
        img_id, channels, label_vec = super().__getitem__(item)

        # stack the channels as RGBY
        img = np.dstack([channels['red'], channels['green'], channels['blue'], channels['yellow']])
        green = img[..., 1]

        if self.dual_transforms is not None:
            aug_result = self.dual_transforms(image=img, extra=green)
            img = aug_result['image']
            green = aug_result['extra']

        if self.img_transforms is not None:
            img = self.img_transforms(image=img)['image']

        # normalize the green channel to be in range [0, 1]
        green = green.astype(np.float32)
        green /= 255

        # convert the data types to floats for all channels
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
        elif isinstance(img, torch.Tensor):
            img = img.float()

        if self.tensorize is not None:
            img = self.tensorize(image=img)['image']
            green = self.tensorize(image=green)['image']
        return img, green, label_vec


# ----------------------------------------------------------------------------------------------------------------------
# OLD STUFF
# ----------------------------------------------------------------------------------------------------------------------


class IsolatedTargetDataset(BaseDataset):
    def __init__(self, train_idx, data_dir, external_data_dir=None, transforms=None):
        """Initialization

        Parameters
        ----------
        train_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.HPACompose
        """
        super().__init__(train_idx, data_dir, external_data_dir)
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
