import numpy as np
from shapely.geos import TopologicalError

from .misc import encode_binary_mask
# from ..segment.shapes import mask_to_polygons


class Cell:
    """Helper class for dealing with cells in images"""

    def __init__(self, cell_id, cell_mask, extract_geom=False):
        """Initialization

        Parameters
        ----------
        cell_id: int
            The integer ID of the cell. This cannot be zero (the background value).
        cell_mask: np.ndarray
        extract_geom: bool, optional
        """
        self.cell_id = cell_id
        self.cell_mask = cell_mask
        self.total_pxls = cell_mask.sum()
        self.preds = set()

        # create the RLE encoding for this cell
        self.rle_encoding = encode_binary_mask(cell_mask).decode("utf-8")

        # extract the shapely Polygon for the cell segmentation
        self.geom = None
        # if extract_geom:
        #     geom_list = mask_to_polygons(cell_mask)
        #     if len(geom_list) > 1:
        #         print(f'Warning: more than one polygon extracted for cell #{cell_id}')
        #     self.geom = geom_list[0].buffer(0)

    def get_shapely_intersection(self, other_geom):
        """Find the intersection of this cell with a shapely Polygon

        Parameters
        ----------
        other_geom: shapely.geometry.Polygon

        Returns
        -------
        shapely.geometry.Polygon
        """
        if self.geom is None:
            raise ValueError('Cell geometry not yet extracted!')
        try:
            intersection = self.geom.intersection(other_geom)
        except TopologicalError:
            self.geom = self.geom.buffer(0)
            intersection = self.geom.intersection(other_geom)
        return intersection

    def get_intersection(self, seg_mask):
        return seg_mask[self.cell_mask]

    def calc_intersect(self, seg_mask):
        intersect_pxls = self.get_intersection(seg_mask)
        return intersect_pxls[np.nonzero(intersect_pxls)].size / self.total_pxls

    def calc_confidence(self, heatmap):
        intersect_pxls = self.get_intersection(heatmap)
        return intersect_pxls[np.nonzero(intersect_pxls)].mean()

    def add_prediction(self, label, confidence):
        self.preds.add((label, confidence))

    def get_prediction_string(self):
        pred_strings = []
        for pred_label, confidence in self.preds:
            pred_strings.append(f'{pred_label} {confidence} {self.rle_encoding}')
        return ' '.join(pred_strings)


class Segmentation:
    def __init__(self, geom, label, confidence=None):
        self.geom = geom
        self.label = label
        self.confidence = confidence


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
