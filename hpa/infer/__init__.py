from copy import deepcopy
import base64
import zlib

import numpy as np
from pycocotools import _mask as coco_mask


def encode_binary_mask(mask):
    if mask.dtype != np.bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")
    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


class Cell:
    """Helper class for dealing with cells in images"""

    def __init__(self, cell_id, full_size_mask, reduced_mask=None):
        """Initialization

        Parameters
        ----------
        cell_id: int
            The integer ID of the cell. This cannot be zero (the background value).
        full_size_mask: np.ndarray
            A boolean mask identifying the pixels containing the target cell.
            The mask should occupy the full size of the target image.
            The `rle_encoding` attribute corresponds to the full size dimension.
        reduced_mask: np.ndarray, optional
            A boolean mask identifying the pixels containing the target cell.
            The mask should occupy the reduced image size for classification purposes.
            This argument is optional.
        """
        self.cell_id = cell_id
        self.full_size_mask = full_size_mask
        self.reduced_mask = reduced_mask
        self.rle_encoding = encode_binary_mask(full_size_mask)
        self.preds = []

    @staticmethod
    def __isolate_cell(img, mask, background_value):
        if len(img.shape) == 2:
            background_mask = ~mask
        else:
            background_mask = ~np.expand_dims(mask, axis=-1)
        img_copy = deepcopy(img)
        img_copy[background_mask] = background_value
        return img_copy

    def isolate_full_cell(self, img, background_value=0):
        return self.__isolate_cell(img, self.full_size_mask, background_value)

    def isolate_reduced_cell(self, img, background_value=0):
        if self.reduced_mask is None:
            raise ValueError('Cannot isolate cell when `reduced_mask` is None')
        else:
            return self.__isolate_cell(img, self.reduced_mask, background_value)

    def add_prediction(self, label):
        self.preds.append(label)
