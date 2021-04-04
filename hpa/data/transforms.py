import albumentations as A
import numpy as np


class HPACompose(A.Compose):
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(transforms, additional_targets={'extra': 'image'}, *args, **kwargs)


class AdjustableCropCompose(HPACompose):
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(transforms, *args, **kwargs)
        self.crop_transform = None
        for transform in transforms:
            if isinstance(transform, A.RandomCrop):
                self.crop_transform = transform
        if self.crop_transform is None:
            raise ValueError('You must provide a RandomCrop transformation!')

    def set_crop_size(self, shape):
        self.crop_transform.height = shape[0]
        self.crop_transform.width = shape[1]


class RandomCropCycle(A.RandomCrop):
    def __init__(self, min_dim, max_dim, cycle_size, dual=False):
        # initialize the size using the maximum dimensions
        super().__init__(height=max_dim, width=max_dim)
        self.min_dim = min_dim
        self.max_dim = max_dim

        self.dual = dual
        if self.dual:
            self.cycle_size = 2 * cycle_size
        else:
            self.cycle_size = cycle_size
        self.count = 0

    def sample_new_crop_size(self):
        crop_size = np.random.randint(self.min_dim, self.max_dim + 1)
        self.height = crop_size
        self.width = crop_size

    def apply(self, img, **params):
        if self.count % self.cycle_size == 0:
            self.sample_new_crop_size()
            self.count = 0
        self.count += 1
        return super().apply(img, **params)


class ToBinaryCellSegmentation(A.ImageOnlyTransform):
    """Transform a cell segmentation array (with values 0, 1, ..., num_cells) to a binary cell segmentation array"""

    def __init__(self, dtype=np.float32):
        super().__init__(p=1.0)
        self.dtype = dtype

    def apply(self, img, **params):
        # 1 if cell is present, 0 otherwise
        img = ~(img == 0)
        img = img.astype(self.dtype)
        return img

    @property
    def targets(self):
        return {"image": self.apply}

    def get_params_dependent_on_targets(self, params):
        return ()

    def get_transform_init_args_names(self):
        return ()
