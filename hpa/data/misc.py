import numpy as np


def parse_string_label(label_str):
    return [int(i) for i in label_str.split('|')]


def get_single_label_subset(df):
    return df.loc[~df['Label'].str.contains('\|')]


def get_cell_masks(cell_seg):
    """Extract individual cell masks from a complete cell segmentation image

    Parameters
    ----------
    cell_seg: numpy.ndarray
        The cell segmentation image with the background as 0 and each cell has a unique positive integer ID

    Returns
    -------
    list[numpy.ndarray]
        A list of individual cell masks, each having the same shape as the original cell segmentation image
    """
    seg_values = np.unique(cell_seg)
    cell_ids = seg_values[np.nonzero(seg_values)]

    cell_masks = []
    for cell_id in cell_ids:
        cell_mask = (cell_seg == cell_id)
        cell_masks.append(cell_mask)
    return cell_masks


def remove_empty_masks(masks):
    num_masks = len(masks)
    max_per_mask = masks.reshape((num_masks, -1)).max(axis=-1)
    keep_mask = max_per_mask > 0
    return masks[keep_mask]
