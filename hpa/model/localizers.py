"""Neural networks for localization"""
import torch
import torch.nn.functional as F
from torch.nn import (AdaptiveAvgPool2d, AdaptiveMaxPool2d, BatchNorm1d, BatchNorm2d, Conv2d,
                      Flatten, Module, ReLU, Sequential, Linear, Parameter, Upsample)

from .layers import ConvBlock
from .prm import median_filter, peak_stimulation
from ..utils.model import get_num_output_features, merge_tiles, tile_image_batch


class MaxPooledLocalizer(Module):
    def __init__(self,
                 base_cnn,
                 n_classes,
                 n_hidden_filters=None,
                 deep_final_conv=False,
                 final_conv_bias=True,
                 return_maps=False):
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
        self.return_maps = return_maps

    def forward(self, x):
        feature_maps = self.base_cnn(x)
        class_maps = self.final_conv_block(feature_maps)
        class_scores = self.max_pool(class_maps)
        class_scores = self.flatten(class_scores)
        if self.return_maps:
            return class_maps, class_scores
        return class_scores


class DecomposedDensenet(Module):
    def __init__(self, densenet_model, map_classes=False, max_classes=False):
        """Initialization

        Parameters
        ----------
        densenet_model: hpa.model.bestfitting.densenet.DensenetClass
        map_classes: bool, optional
        """
        super().__init__()

        # define the CNN encoder
        self.densenet_encoder = Sequential(densenet_model.conv1,
                                           densenet_model.encoder2,
                                           densenet_model.encoder3,
                                           densenet_model.encoder4,
                                           densenet_model.encoder5,
                                           ReLU())

        # TODO: extract the `num_features` variable for the encoder output
        self.num_features = 1024
        self.num_classes = densenet_model.logit.out_features

        self.max_pool = AdaptiveMaxPool2d(1)

        # split the BatchNorm1d `bn1` into two separate modules: `bn1_avg` and `bn1_max`
        # `bn1_avg` is a BatchNorm2d module which normalizes the feature maps
        # `bn1_max` is a BatchNorm1d module which normalizes the global-max-pooled features
        self.bn1_avg = self.__prepare_bn1_avg(densenet_model)
        self.bn1_max = self.__prepare_bn1_max(densenet_model)

        # transform fully-connected layer `fc1` into a 1x1 Conv2d used on the feature map branch
        # and another fully-connected layer used on the max-pooled features
        self.fc1_avg = self.__prepare_fc1_avg(densenet_model)
        self.fc1_max = self.__prepare_fc1_max(densenet_model)
        self.fc1_bias = self.__prepare_fc1_bias(densenet_model)

        # transform `bn2` of the original model into a BatchNorm2d module
        self.bn2 = self.__prepare_bn2(densenet_model)

        # transform the `logit` fully-connected layer into a 1x1 Conv2d
        self.fc2 = self.__prepare_fc2(densenet_model)

        if max_classes and map_classes:
            raise ValueError('Cannot have both max_classes and map_classes be True!')
        self.max_classes = max_classes
        self.map_classes = map_classes

        self.final_conv = None
        if map_classes:
            self.final_conv = Conv2d(self.num_classes, 18, kernel_size=1)

    def __prepare_bn1_avg(self, densenet_model):
        bn_avg = BatchNorm2d(num_features=self.num_features)
        bn_avg.weight.data = densenet_model.bn1.weight.data[:self.num_features]
        bn_avg.bias.data = densenet_model.bn1.bias.data[:self.num_features]
        bn_avg.running_mean = densenet_model.bn1.running_mean[:self.num_features]
        bn_avg.running_var = densenet_model.bn1.running_var[:self.num_features]
        return bn_avg

    def __prepare_bn1_max(self, densenet_model):
        bn_max = BatchNorm1d(num_features=self.num_features)
        bn_max.weight.data = densenet_model.bn1.weight.data[self.num_features:]
        bn_max.bias.data = densenet_model.bn1.bias.data[self.num_features:]
        bn_max.running_mean = densenet_model.bn1.running_mean[self.num_features:]
        bn_max.running_var = densenet_model.bn1.running_var[self.num_features:]
        return bn_max

    def __prepare_fc1_avg(self, densenet_model):
        fc1_avg = Conv2d(self.num_features, self.num_features, 1, bias=False)
        fc1_avg.weight.data[..., 0, 0] = densenet_model.fc1.weight.data[:, :self.num_features]
        return fc1_avg

    def __prepare_fc1_max(self, densenet_model):
        fc1_max = Linear(self.num_features, self.num_features, bias=False)
        fc1_max.weight.data = densenet_model.fc1.weight.data[:, self.num_features:]
        return fc1_max

    def __prepare_fc1_bias(self, densenet_model):
        fc1_bias = Parameter(torch.zeros(1, self.num_features, 1, 1))
        fc1_bias.data[0, :, 0, 0] = densenet_model.fc1.bias.data
        return fc1_bias

    def __prepare_bn2(self, densenet_model):
        bn2 = BatchNorm2d(self.num_features)
        bn2.weight.data = densenet_model.bn2.weight.data
        bn2.bias.data = densenet_model.bn2.bias.data
        bn2.running_mean = densenet_model.bn2.running_mean
        bn2.running_var = densenet_model.bn2.running_var
        return bn2

    def __prepare_fc2(self, densenet_model):
        fc2_conv = Conv2d(self.num_features, self.num_classes, kernel_size=1)
        fc2_conv.weight.data[..., 0, 0] = densenet_model.logit.weight.data
        fc2_conv.bias.data = densenet_model.logit.bias.data
        return fc2_conv

    @staticmethod
    def map_old_classes_to_new(class_maps):
        channel_groups = [
            class_maps[:, :8, ...],
            class_maps[:, [11], ...],
            class_maps[:, [12, 13], ...].max(dim=1, keepdim=True).values,
            class_maps[:, [14, 17, 19], ...],
            class_maps[:, [21, 22], ...].max(dim=1, keepdim=True).values,
            class_maps[:, [23, 24, 25], ...],
            class_maps[:, [8, 9, 10, 20, 26], ...].max(dim=1, keepdim=True).values
        ]
        return torch.cat(channel_groups, dim=1)

    def forward(self, x):
        batch_size, *_ = x.shape
        feature_maps = self.densenet_encoder(x)

        # (disabled) avg pool branch
        act_avg = self.bn1_avg(feature_maps)

        # max pool branch
        max_features = self.max_pool(feature_maps)
        max_features = max_features.view(batch_size, -1)
        act_max = self.bn1_max(max_features)

        # apply dropout on both branches
        # act_avg = F.dropout2d(act_avg, p=0.5, training=self.training)
        # act_max = F.dropout(act_max, p=0.5, training=self.training)

        # pass each branch through their respective 1x1 Conv2d or Linear layers
        act_fc_avg = self.fc1_avg(act_avg)
        act_fc_max = self.fc1_max(act_max)

        # join the two branches together by adding the "max" branch to the "avg" branch
        # add their shared bias
        act_fc = act_fc_avg + act_fc_max.view(*act_fc_max.shape, 1, 1)
        act_fc += self.fc1_bias
        act_fc = F.relu(act_fc)

        # apply second round of batch normalization and dropout
        act_bn2 = self.bn2(act_fc)
        # act_bn2 = F.dropout2d(act_bn2, p=0.5, training=self.training)

        # linearly map the features to the class map (using the old 28 labels)
        class_maps = self.fc2(act_bn2)

        # linearly map the classes to the newer class labels if requested
        if self.map_classes:
            class_maps = self.final_conv(class_maps)

        # condense the class maps to the new label set
        if self.max_classes:
            class_maps = self.map_old_classes_to_new(class_maps)

        return class_maps


class PooledLocalizer(Module):
    """A more general version of the `MaxPooledLocalizer`"""

    def __init__(self, cnn, pool='max', return_maps=True):
        super().__init__()
        self.cnn = cnn
        self.pool_type = pool
        self.return_maps = return_maps

        if self.pool_type == 'max':
            self.pool_fn = AdaptiveMaxPool2d((1, 1))
        elif self.pool_type == 'avg':
            self.pool_fn = AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f'No supported pool type: "{self.pool_type}"')
        self.flatten = Flatten()

    def forward(self, x):
        class_maps = self.cnn(x)
        class_logits = self.pool_fn(class_maps)
        class_logits = self.flatten(class_logits)
        if self.return_maps:
            return class_maps, class_logits
        return class_logits


class PeakResponseLocalizer(Module):
    def __init__(self, cnn, return_maps=True, return_peaks=False, window_size=3, scale_factor=1):
        super().__init__()
        self.cnn = cnn
        self.return_maps = return_maps
        self.return_peaks = return_peaks
        self.window_size = window_size
        self.scale_factor = scale_factor

        self.upsample = None
        if self.scale_factor > 1:
            self.upsample = Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        class_maps = self.cnn(x)
        if self.upsample is not None:
            class_maps = self.upsample(class_maps)

        peak_list, class_logits = peak_stimulation(input=class_maps,
                                                   return_aggregation=True,
                                                   win_size=self.window_size,
                                                   peak_filter=median_filter)
        result = []
        if self.return_maps:
            result.append(class_maps)
        if self.return_peaks:
            result.append(peak_list)
        if len(result) > 0:
            return tuple(result + [class_logits])
        return class_logits


class Densenet121Pyramid(Module):
    def __init__(self, densenet_model):
        super().__init__()


    def forward(self, x):
        return


class PuzzleCAM(PooledLocalizer):
    def __init__(self, cnn, pool='max', return_maps=True, tile_size=(2, 2)):
        super().__init__(cnn, pool, return_maps)
        self.tile_size = tile_size
        self.use_tiles = True

    def base_branch(self, x):
        class_maps = self.cnn(x)
        class_scores = self.pool_fn(class_maps)
        class_scores = self.flatten(class_scores)
        return class_maps, class_scores

    def tiled_branch(self, x):
        # tile the image batch
        tiles = tile_image_batch(x, *self.tile_size)

        # calculate the feature maps for each tile
        class_maps = self.cnn(tiles)

        # merge the tiled feature maps into a full image again
        class_maps = merge_tiles(class_maps, *self.tile_size)

        # calculate class scores using the merged features maps of the tiled images
        class_scores = self.pool_fn(class_maps)
        class_scores = self.flatten(class_scores)
        return class_maps, class_scores

    def forward(self, x):
        full_class_maps, full_class_scores = self.base_branch(x)
        if self.use_tiles:
            tile_class_maps, tile_class_scores = self.tiled_branch(x)
            return full_class_maps, full_class_scores, tile_class_maps, tile_class_scores
        elif self.return_maps:
            return full_class_maps, full_class_scores
        else:
            return full_class_scores
