from torch.nn import AdaptiveMaxPool2d, Conv2d, Flatten, Module


def get_num_output_features(cnn):
    final_conv = None
    for m in cnn.modules():
        if isinstance(m, Conv2d):
            final_conv = m
    if final_conv is None:
        raise ValueError('The input model has no Conv2d layers!')
    return final_conv.out_channels


class MaxPooledLocalizer(Module):
    def __init__(self, base_cnn, n_classes):
        super().__init__()
        self.base_cnn = base_cnn
        self.n_base_filters = get_num_output_features(base_cnn)
        self.final_conv = Conv2d(self.n_base_filters, n_classes, kernel_size=(1, 1), bias=False)
        self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.flatten = Flatten()

    def forward(self, x):
        feature_maps = self.base_cnn(x)
        class_maps = self.final_conv(feature_maps)
        class_scores = self.max_pool(class_maps)
        class_scores = self.flatten(class_scores)
        return class_scores
