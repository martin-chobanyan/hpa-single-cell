import torch

from .base import HPACellSegmenter


def get_cell_bbox(cell_mask):
    """Input has shape (height, width)"""
    y_idx, x_idx = torch.where(cell_mask)
    return (y_idx.min(), y_idx.max() + 1), (x_idx.min(), x_idx.max() + 1)
