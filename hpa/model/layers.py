import torch
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, bnorm=True, relu=True, bias=True):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=int(kernel_size / 2))
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


class LogSumExp(Module):
    """Log-Sum-Exponential averaging layer for 2D matrices(interpolates between maximum and average)"""

    def __init__(self, r=5, dim=0, keepdim=True):
        """Initialization

        Parameters
        ----------
        r: float, optional
            The interpolation parameter (r = 1 ~ average, r >> 1 ~ maximum; default = 5)
        dim: int, optional
            The dimension to sum across (default = 0)
        keepdim: bool, optional
            Whether or not to keep the reduced dimension (default = True)
        """
        super().__init__()
        self.r = r
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        assert len(x.shape) == 2, 'Input must be a 2D matrix tensor'
        n, _ = x.shape
        lse = torch.logsumexp(self.r * x, dim=self.dim, keepdim=self.keepdim)
        return (1 / self.r) * (lse + torch.log(torch.tensor(1 / n)))
