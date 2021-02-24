import albumentations as A
import numpy as np


class HPACompose(A.Compose):
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(transforms, additional_targets={'extra': 'image'}, *args, **kwargs)


class ToBinaryCellSegmentation(A.ImageOnlyTransform):
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
