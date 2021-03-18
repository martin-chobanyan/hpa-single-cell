"""Neural networks for localization"""

import torch
from torch.nn import AdaptiveMaxPool2d, BatchNorm2d, Conv2d, Flatten, Module, ReLU, Sequential, Upsample

from hpa.utils.model import merge_tiles, tile_image_batch


def get_num_output_features(cnn):
    final_conv = None
    for m in cnn.modules():
        if isinstance(m, Conv2d):
            final_conv = m
    if final_conv is None:
        raise ValueError('The input model has no Conv2d layers!')
    return final_conv.out_channels


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, bnorm=True, relu=True, bias=True):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias=bias)

        self.bnorm = None
        if bnorm:
            self.bnorm = BatchNorm2d(out_channels)

        self.relu = None
        if relu:
            self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bnorm is not None:
            x = self.bnorm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


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
    def __init__(self,
                 base_cnn,
                 n_classes,
                 n_hidden_filters=None,
                 deep_final_conv=False,
                 final_conv_bias=True,
                 merge_classes=False,
                 seg_shape=None):
        super().__init__()
        self.base_cnn = base_cnn
        self.n_hidden_filters = n_hidden_filters
        if n_hidden_filters is None:
            self.n_hidden_filters = get_num_output_features(base_cnn)

        if deep_final_conv:
            self.final_conv_block = Sequential(
                ConvBlock(self.n_hidden_filters, self.n_hidden_filters, kernel_size=1),
                Conv2d(self.n_hidden_filters, n_classes, kernel_size=1, bias=final_conv_bias)
            )
        else:
            self.final_conv_block = Conv2d(self.n_hidden_filters, n_classes, kernel_size=(1, 1), bias=final_conv_bias)

        self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.flatten = Flatten()

        self.merge_classes = merge_classes
        self.seg_shape = seg_shape
        self.merge_nn = None
        if merge_classes and seg_shape is not None:
            self.merge_nn = MergeClassHeatmaps(seg_shape)

    def forward(self, x):
        feature_maps = self.base_cnn(x)
        class_maps = self.final_conv_block(feature_maps)
        class_scores = self.max_pool(class_maps)
        class_scores = self.flatten(class_scores)

        if self.merge_nn is not None:
            segmentation = self.merge_nn(class_maps)
            return class_scores, segmentation
        return class_scores


class PuzzleCAM(Module):
    def __init__(self,
                 base_cnn,
                 n_classes,
                 tile_size=(2, 2),
                 n_hidden_filters=None,
                 deep_final_conv=False,
                 final_conv_bias=True):

        super().__init__()
        self.base_cnn = base_cnn
        self.tile_size = tile_size
        self.n_hidden_filters = n_hidden_filters

        if n_hidden_filters is None:
            self.n_hidden_filters = get_num_output_features(base_cnn)

        if deep_final_conv:
            self.final_conv_block = Sequential(
                ConvBlock(self.n_hidden_filters, self.n_hidden_filters, kernel_size=1),
                Conv2d(self.n_hidden_filters, n_classes, kernel_size=1, bias=final_conv_bias)
            )
        else:
            self.final_conv_block = Conv2d(self.n_hidden_filters,
                                           n_classes,
                                           kernel_size=(1, 1),
                                           bias=final_conv_bias)

        self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.flatten = Flatten()
        self.use_tiles = True

    def base_branch(self, x):
        # calculate feature maps using full image
        feature_maps = self.base_cnn(x)
        class_maps = self.final_conv_block(feature_maps)

        # calculate class scores using full image
        class_scores = self.max_pool(class_maps)
        class_scores = self.flatten(class_scores)
        return class_maps, class_scores

    def tiled_branch(self, x):
        # tile the image batch
        tiles = tile_image_batch(x, *self.tile_size)

        # calculate the feature maps for each tile
        feature_maps = self.base_cnn(tiles)
        class_maps = self.final_conv_block(feature_maps)

        # merge the tiled feature maps into a full image again
        class_maps = merge_tiles(class_maps, *self.tile_size)

        # calculate class scores using the merged features maps of the tiled images
        class_scores = self.max_pool(class_maps)
        class_scores = self.flatten(class_scores)
        return class_maps, class_scores

    def forward(self, x):
        full_class_maps, full_class_scores = self.base_branch(x)
        if self.use_tiles:
            tile_class_maps, tile_class_scores = self.tiled_branch(x)
            return full_class_maps, full_class_scores, tile_class_maps, tile_class_scores
        else:
            return full_class_scores
