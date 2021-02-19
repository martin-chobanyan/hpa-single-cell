import torch
from torch.nn import Module


class DPNEncoder(Module):
    def __init__(self, encoder_stages):
        super().__init__()
        self.stages = encoder_stages

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
            if isinstance(x, tuple):
                x = torch.cat(x, dim=1)
        return x


class GreenAndReferenceEncoder(Module):
    def __init__(self, ref_encoder, n_green_hidden_channels, n_output_channels):
        super().__init__()
        self.n_green_hidden_channels = n_green_hidden_channels
        self.n_output_channels = n_output_channels

        self.green_encoder = None
        self.ref_encoder = ref_encoder

    def forward(self, x):
        return x
