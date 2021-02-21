import torch
from torch.nn import AdaptiveMaxPool2d, Conv2d, Flatten, Module, Upsample


def get_num_output_features(cnn):
    final_conv = None
    for m in cnn.modules():
        if isinstance(m, Conv2d):
            final_conv = m
    if final_conv is None:
        raise ValueError('The input model has no Conv2d layers!')
    return final_conv.out_channels


class MergeClassHeatmaps(Module):
    def __init__(self, seg_shape):
        super().__init__()
        self.seg_shape = seg_shape
        self.upsample = Upsample(seg_shape, mode='bilinear', align_corners=False)

    def forward(self, class_heatmaps):
        merged_heatmap = class_heatmaps.max(dim=1, keepdim=True).values
        merged_heatmap = torch.sigmoid(merged_heatmap)
        merged_heatmap = self.upsample(merged_heatmap)
        return merged_heatmap


class MaxPooledLocalizer(Module):
    def __init__(self, base_cnn, n_classes, n_hidden_filters=None, merge_classes=False, seg_shape=None):
        super().__init__()
        self.base_cnn = base_cnn
        self.n_hidden_filters = n_hidden_filters
        if n_hidden_filters is None:
            self.n_hidden_filters = get_num_output_features(base_cnn)
        self.final_conv = Conv2d(self.n_hidden_filters, n_classes, kernel_size=(1, 1), bias=False)
        self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.flatten = Flatten()

        self.merge_classes = merge_classes
        self.seg_shape = seg_shape
        self.merge_nn = None
        if merge_classes and seg_shape is not None:
            self.merge_nn = MergeClassHeatmaps(seg_shape)

    def forward(self, x):
        feature_maps = self.base_cnn(x)
        class_maps = self.final_conv(feature_maps)
        class_scores = self.max_pool(class_maps)
        class_scores = self.flatten(class_scores)

        if self.merge_nn is not None:
            segmentation = self.merge_nn(class_maps)
            return class_scores, segmentation
        return class_scores
