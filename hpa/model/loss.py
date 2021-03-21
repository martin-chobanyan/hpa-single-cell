"""Module containing useful loss functions"""

import torch
from torch.nn import Module, MSELoss

from .bestfitting.layers_loss import FocalSymmetricLovaszHardLogLoss, FocalLoss


class ClassHeatmapLoss(Module):
    def __init__(self, heatmap_loss_fn=None):
        super().__init__()
        self.heatmap_loss_fn = heatmap_loss_fn
        if self.heatmap_loss_fn is None:
            self.heatmap_loss_fn = MSELoss()

    def forward(self, class_heatmaps, target_heatmap, label_vectors):
        """Forward Propagation

        Parameters
        ----------
        class_heatmaps: torch.Tensor
            The logit feature maps for all of the classes. Has shape (batch, num_classes, height, width)
        target_heatmap: torch.Tensor
            The target (cell-segmentation) heatmap with values in [0, 1].
            Has shape (batch, 1, height, width) where height and width match the shape of `class_heatmaps`.
        label_vectors: torch.Tensor
            The target label vectors for each instance in the batch.
            These vectors will indicate which classes we use to construct the final heatmap prediction.
            Has shape (batch, num_classes)

        Returns
        -------
        torch.Tensor
        """
        pred_heatmap = []
        class_heatmaps = torch.sigmoid(class_heatmaps)
        for batch_idx, label_vec in enumerate(label_vectors):
            label_idx = torch.where(label_vec)[0]
            if len(label_idx) > 0:
                tgt_maps = class_heatmaps[batch_idx, label_idx]
            else:
                tgt_maps = class_heatmaps[batch_idx, :]
            heatmap = tgt_maps.max(dim=0).values
            pred_heatmap.append(heatmap)
        pred_heatmap = torch.stack(pred_heatmap)
        pred_heatmap = pred_heatmap.unsqueeze(1)
        return self.heatmap_loss_fn(pred_heatmap, target_heatmap)
