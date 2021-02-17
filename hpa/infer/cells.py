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
        intersect_pxls = seg_mask[self.cell_mask].sum()
        return intersect_pxls / self.total_pxls

    def add_prediction(self, label):
        self.preds.add(label)

    def get_prediction_string(self):
        pred_strings = []
        confidence = 1  # constant for now
        for pred_label in self.preds:
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
