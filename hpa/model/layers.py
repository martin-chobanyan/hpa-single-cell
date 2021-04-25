import torch
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU, MultiheadAttention, Dropout, Sequential, Linear, LayerNorm


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


class RoIPool(Module):
    """Cell / Region of Interest Pooling

    Extracts a feature vector for each cell in an image. The feature vector for each cell is calculated
    by pooling the pixels in the feature map which have been keep after the cell mask filter.
    """

    def __init__(self, method='max'):
        super().__init__()
        if method == 'max':
            self.pool_fn = self.roi_maxpool
        elif method == 'avg':
            self.pool_fn = self.roi_avgpool
        elif method == 'max_and_avg':
            self.pool_fn = self.roi_max_and_avg
        else:
            raise ValueError(f'Unknown RoI pooling method: {method}')

    @staticmethod
    def roi_maxpool(x):
        return x.max(dim=1).values

    @staticmethod
    def roi_avgpool(x):
        return x.mean(dim=1)

    def roi_max_and_avg(self, x):
        x_max = self.roi_maxpool(x)
        x_avg = self.roi_avgpool(x)
        return torch.cat([x_max, x_avg])

    def forward(self, feature_maps, cell_masks, cell_counts):
        """Forward call

        Parameters
        ----------
        feature_maps: torch.Tensor
            CNN feature maps with shape (batch, num_features, height, width)
        cell_masks: torch.Tensor
            Boolean mask for each cell with shape (batch * cell_per_batch, height, width)
        cell_counts: torch.Tensor
            Sequence of cell counts per image in the batch. Has shape (batch,)

        Returns
        -------
        torch.Tensor
            A tensor with shape (batch * cell_per_batch, num_features) where each row is a feature vector for a cell.
        """
        i = 0
        cell_vectors = []
        for batch_idx, cell_count in enumerate(cell_counts):
            for cell_mask in cell_masks[i:i+cell_count]:
                roi = feature_maps[batch_idx, :, cell_mask]
                feature_vec = self.pool_fn(roi)
                cell_vectors.append(feature_vec)
            i += cell_count
        return torch.stack(cell_vectors)


class TransformerEncoderLayer(Module):
    def __init__(self, emb_dim=512, num_heads=4, dropout=0.1, fc_hidden_dim=2048):
        super().__init__()

        # multihead attention section
        self.multihead_attn = MultiheadAttention(emb_dim, num_heads)
        self.dropout1 = Dropout(p=dropout)
        self.layer_norm1 = LayerNorm(emb_dim)

        # feedforward section
        self.linear = Sequential(Linear(emb_dim, fc_hidden_dim), ReLU(), Linear(fc_hidden_dim, emb_dim))
        self.dropout2 = Dropout(p=dropout)
        self.layer_norm2 = LayerNorm(emb_dim)

    # input shape: (num_cells, 1, emb_dim)
    def forward(self, cell_seq):
        # section 1: multi-head attention
        attn_out, _ = self.multihead_attn(cell_seq, cell_seq, cell_seq)
        attn_out = self.dropout1(attn_out)
        attn_out += cell_seq
        attn_out = self.layer_norm1(attn_out)

        # section 2: feedforward
        linear_out = self.linear(attn_out)
        linear_out = self.dropout2(linear_out)
        linear_out += attn_out
        out = self.layer_norm2(linear_out)

        return out


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
        if self.dim == 0:
            n, _ = x.shape
        else:
            _, n = x.shape
        lse = torch.logsumexp(self.r * x, dim=self.dim, keepdim=self.keepdim)
        return (1 / self.r) * (lse + torch.log(torch.tensor(1 / n)))
