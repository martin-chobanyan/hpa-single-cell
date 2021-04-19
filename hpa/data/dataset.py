import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from cv2 import INTER_LINEAR, INTER_NEAREST
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset

from .misc import parse_string_label, remove_empty_masks
from .transforms import ToCellMasks

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


class BaseDataset(Dataset):
    def __init__(self, data_idx, data_dir, external_data_dir=None):
        """Initialization

        Parameters
        ----------
        data_idx: pandas.DataFrame
        data_dir: str
        """
        super().__init__()
        self.data_idx = data_idx
        self.data_dir = data_dir
        self.external_data_dir = external_data_dir
        self.n_samples = len(data_idx)

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
    def __init__(self, data_idx, data_dir, external_data_dir=None, transforms=None):
        """Initialization

        Parameters
        ----------
        data_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.HPACompose
        """
        super().__init__(data_idx, data_dir, external_data_dir)
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


class CroppedRGBYDataset(RGBYDataset):
    def __init__(self, data_idx, data_dir, transforms, external_data_dir=None):
        """Initialization
        Parameters
        ----------
        data_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.AdjustableCropCompose
        external_data_dir: str
        """
        super().__init__(data_idx, data_dir, external_data_dir, transforms=transforms)

    def set_crop_size(self, size):
        self.transforms.set_crop_size((size, size))


class RGBYWithCellMasks(BaseDataset):
    def __init__(self,
                 data_idx,
                 data_dir,
                 seg_dir,
                 external_data_dir=None,
                 external_seg_dir=None,
                 dual_transforms=None,
                 img_transforms=None,
                 img_dim=1536,
                 downsize_scale=32,
                 tensorize=True):
        super().__init__(data_idx, data_dir, external_data_dir)
        self.seg_dir = seg_dir
        self.external_seg_dir = external_seg_dir

        self.dual_transforms = dual_transforms
        self.img_transforms = img_transforms

        self.img_dim = img_dim
        self.downsize_scale = downsize_scale
        self.feature_map_dim = int(img_dim / downsize_scale)
        self.seg_transforms = A.Compose([
            A.Resize(height=self.feature_map_dim, width=self.feature_map_dim, interpolation=INTER_NEAREST),
            ToCellMasks()
        ])

        if tensorize:
            self.tensorize = ToTensorV2()
        else:
            self.tensorize = None

    def __getitem__(self, item):
        img_id, channels, label_vec = super().__getitem__(item)

        # stack the channels as RGBY
        img = np.dstack([channels['red'], channels['green'], channels['blue'], channels['yellow']])

        # load the segmentation map
        if (self.external_seg_dir is not None) and (self.data_idx.at[item, 'Source'] == 'external'):
            seg_dir = self.external_seg_dir
        else:
            seg_dir = self.seg_dir
        seg = np.load(os.path.join(seg_dir, f'{img_id}.npz'))['arr_0']

        # transform both image and seg simultaneously
        if self.dual_transforms is not None:
            aug_result = self.dual_transforms(image=img, mask=seg)
            img = aug_result['image']
            seg = aug_result['mask']

        # only image transform
        if self.img_transforms is not None:
            img = self.img_transforms(image=img)['image']

        # get the individual cell masks and their count
        cell_masks = self.seg_transforms(image=seg)['image']
        if cell_masks is not None:
            cell_masks = remove_empty_masks(cell_masks)
            num_cells = len(cell_masks)
        else:
            num_cells = 0

        # convert the image to a float
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)

        if self.tensorize is not None:
            img = self.tensorize(image=img)['image']
            cell_masks = torch.from_numpy(cell_masks)
        return img, cell_masks, num_cells, label_vec


class RGBYWithSegmentation(BaseDataset):
    def __init__(self,
                 data_idx,
                 data_dir,
                 seg_dir,
                 external_data_dir=None,
                 external_seg_dir=None,
                 dual_transforms=None,
                 img_transforms=None,
                 seg_transforms=None,
                 tensorize=True):
        super().__init__(data_idx, data_dir, external_data_dir)
        self.seg_dir = seg_dir
        self.external_seg_dir = external_seg_dir

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

        if (self.external_seg_dir is not None) and (self.data_idx.at[item, 'Source'] == 'external'):
            seg_dir = self.external_seg_dir
        else:
            seg_dir = self.seg_dir

        # load the segmentation map
        seg = np.load(os.path.join(seg_dir, f'{img_id}.npz'))['arr_0']

        # if the label is 18 ("other"), zero out the segmentation map
        if label_vec.sum() == 0:
            seg = np.zeros(seg.shape)

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
                 data_idx,
                 data_dir,
                 external_data_dir=None,
                 dual_transforms=None,
                 img_transforms=None,
                 tensorize=True):
        super().__init__(data_idx, data_dir, external_data_dir)
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
    def __init__(self, data_idx, data_dir, external_data_dir=None, transforms=None):
        """Initialization

        Parameters
        ----------
        data_idx: pandas.DataFrame
        data_dir: str
        transforms: hpa.data.transforms.HPACompose
        """
        super().__init__(data_idx, data_dir, external_data_dir)
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
