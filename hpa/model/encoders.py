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
