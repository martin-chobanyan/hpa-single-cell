from albumentations.augmentations.transforms import Flip
from cv2 import INTER_AREA, resize


class ResizeImage:
    def __init__(self, size):
        """Initialization

        Parameters
        ----------
        size: tuple
            The (width, height) of the image
        """
        self.size = size

    def __call__(self, img):
        """Main method

        Parameters
        ----------
        img: numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return resize(img, dsize=self.size, interpolation=INTER_AREA)


class RandomFlip:
    def __init__(self, p):
        self.flip = Flip(p=p)

    def __call__(self, img):
        return self.flip(image=img)['image']
