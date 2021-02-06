import albumentations as A


class HPACompose(A.Compose):
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(transforms, additional_targets={'ref': 'image'}, *args, **kwargs)
