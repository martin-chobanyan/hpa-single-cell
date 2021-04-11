from torch.nn import BatchNorm2d, Conv2d, Module, ReLU


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, bnorm=True, relu=True, bias=True):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=int(kernel_size/2))

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
