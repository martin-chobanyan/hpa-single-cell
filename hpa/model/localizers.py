"""Neural networks for localization"""

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, BatchNorm2d, Conv2d, Flatten, Module, ReLU, Sequential

from hpa.utils.model import get_num_output_features, merge_tiles, tile_image_batch


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
