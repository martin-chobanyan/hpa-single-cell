"""This module contains a wrapper for the baseline cell segmentation model"""

import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell


class HPACellSegmenter:
    def __init__(self, nuclei_model_path, cell_model_path, scale_factor=0.25, device='cpu'):
        """Initialization

        Parameters
        ----------
        nuclei_model_path: str
        cell_model_path: str
        scale_factor: float, optional
        device: str, optional
        """
        self.nuclei_model_path = nuclei_model_path
        self.cell_model_path = cell_model_path
        self.scale_factor = scale_factor
        self.device = device

        self.segmenter = cellsegmentator.CellSegmentator(
            self.nuclei_model_path,
            self.cell_model_path,
            scale_factor=scale_factor,
            device=device,
            padding=False,
            multi_channel_model=True
        )

    def __call__(self, microtuble_img, er_img, nuclei_img):
        """Segment an image of cells given three filters

        Parameters
        ----------
        microtuble_img: numpy.ndarray
        er_img: numpy.ndarray
        nuclei_img: numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        input_img = [[microtuble_img], [er_img], [nuclei_img]]
        cell_segmentations = self.segmenter.pred_cells(input_img)
        nuclei_segmentations = self.segmenter.pred_nuclei([nuclei_img])
        nuclei_mask, cell_mask = label_cell(nuclei_segmentations[0], cell_segmentations[0])
        return cell_mask
