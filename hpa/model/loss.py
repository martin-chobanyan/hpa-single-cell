"""Module containing useful loss functions"""

import torch
from torch.nn import L1Loss, Module
from torch.nn.functional import interpolate

from .bestfitting.layers_loss import FocalSymmetricLovaszHardLogLoss, FocalLoss


class ClassHeatmapLoss(Module):
    def __init__(self, heatmap_loss_fn=None):
        super().__init__()
        self.heatmap_loss_fn = heatmap_loss_fn
        if self.heatmap_loss_fn is None:
            self.heatmap_loss_fn = L1Loss()

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


# class BackgroundLoss(Module):
#     def forward(self, class_maps, batch_seg):
#         # upsample the class response maps to the same dimension as the cell segmentations
#         pred_seg = interpolate(torch.sigmoid(class_maps), size=batch_seg.shape[-2:], mode='bilinear', align_corners=False)
#
#         # transform the cell segmentations into background masks
#         bckgrnd_mask = ~batch_seg.bool()
#         bckgrnd_mask = bckgrnd_mask.repeat_interleave(pred_seg.shape[1], dim=1)
#
#         # average the activations in the background
#         return pred_seg[bckgrnd_mask].mean()


class BackgroundLoss(Module):
    def forward(self, class_maps, batch_seg, batch_label):
        # upsample the class response maps to the same dimension as the cell segmentations
        pred_seg = interpolate(class_maps, size=batch_seg.shape[-2:], mode='bilinear', align_corners=False)

        bckgrnd_loss = 0
        for batch_idx, (seg, label_vec) in enumerate(zip(batch_seg, batch_label)):
            # isolate the class response maps for the target labels
            label_idx = torch.where(label_vec)[0]
            if len(label_idx) > 0:
                tgt_maps = pred_seg[batch_idx, label_idx]
            else:
                tgt_maps = pred_seg[batch_idx]

            # define the background mask
            bckgrnd_mask = ~seg.bool()
            bckgrnd_mask = bckgrnd_mask.repeat_interleave(len(tgt_maps), dim=0)

            # isolate and average the background
            bckgrnd_loss += tgt_maps[bckgrnd_mask].mean()
        return torch.sigmoid(bckgrnd_loss / len(batch_label))


class PuzzleRegLoss(Module):
    def __init__(self, logits=False, criterion=None):
        super().__init__()
        self.logits = logits
        self.criterion = criterion

        if self.criterion is None:
            self.criterion = L1Loss()

    def forward(self, full_cam, tiled_cam):
        if self.logits:
            full_cam = torch.sigmoid(full_cam)
            tiled_cam = torch.sigmoid(tiled_cam)
        return self.criterion(full_cam, tiled_cam)
