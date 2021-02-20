"""Module containing useful loss functions"""

from torch.nn import Module
from torch.nn.functional import logsigmoid


# this class is copied over from bestfitting's code
# https://github.com/CellProfiling/HPA-competition-solutions/blob/master/bestfitting/src/layers/loss.py#L8
class FocalLoss(Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()
