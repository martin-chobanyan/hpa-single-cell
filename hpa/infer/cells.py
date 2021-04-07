import numpy as np

from .misc import encode_binary_mask


class Cell:
    """Helper class for dealing with cells in images"""

    def __init__(self, cell_id, cell_mask):
        """Initialization

        Parameters
        ----------
        cell_id: int
            The integer ID of the cell. This cannot be zero (the background value).
        cell_mask: np.ndarray
        """
        self.cell_id = cell_id
        self.cell_mask = cell_mask
        self.total_pxls = cell_mask.sum()
        self.rle_encoding = encode_binary_mask(cell_mask).decode("utf-8")
        self.preds = set()

    def calc_intersect(self, seg_mask):
        intersect_pxls = seg_mask[self.cell_mask]
        return intersect_pxls[np.nonzero(intersect_pxls)].size / self.total_pxls

    def calc_confidence(self, heatmap):
        intersect_pxls = heatmap[self.cell_mask]
        return intersect_pxls[np.nonzero(intersect_pxls)].mean()

    def add_prediction(self, label, confidence):
        self.preds.add((label, confidence))

    def get_prediction_string(self):
        pred_strings = []
        for pred_label, confidence in self.preds:
            pred_strings.append(f'{pred_label} {confidence} {self.rle_encoding}')
        return ' '.join(pred_strings)


def get_cells(cell_segmentation):
    """Extract cells from a cell segmentation map

    Parameters
    ----------
    cell_segmentation: numpy.ndarray

    Returns
    -------
    list[Cell]
    """
    cells = []
    num_cells = np.max(cell_segmentation)
    for cell_id in range(1, num_cells + 1):
        cell_mask = (cell_segmentation == cell_id)
        cell = Cell(cell_id, cell_mask)
        cells.append(cell)
    return cells
