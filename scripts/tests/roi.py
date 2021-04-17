import torch

from hpa.model.layers import RoIPool


fmaps = torch.zeros(1, 3, 5, 5)
fmaps[0, 0, :2, -2:] = 1
fmaps[0, 1, 2, :] = 10
fmaps[0, 1, :, 2] = 10
fmaps[0, 2, 0, 0] = 100
fmaps[0, 2, -1, 0] = 100
fmaps[0, 2, 0, -1] = 100
fmaps[0, 2, -1, -1] = 100


masks = torch.zeros(4, 5, 5, dtype=torch.bool)
masks[0, :3, 2:] = True
masks[1, :2, 3:] = True
masks[2, 1:4, 1:4] = True
masks[3, 1:3, 1:3] = True
masks[3, 2:4, 2:4] = True
masks[3, 3:, -1] = True

method = 'max'
roi_fn = RoIPool(method)
result = roi_fn(fmaps, masks, torch.LongTensor([len(masks)]))
target = torch.Tensor([
    [1, 10, 100],
    [1, 0, 100],
    [1, 10, 0],
    [0, 10, 100]
])
assert (result == target).all(), 'RoI with max fails assert'

method = 'avg'
roi_fn = RoIPool(method)
result = roi_fn(fmaps, masks, torch.LongTensor([len(masks)]))
target = torch.Tensor([
    [4/9, 50/9, 100/9],
    [1, 0, 25],
    [1/9, 50/9, 0],
    [0, 50/9, 100/9]
])
assert (result == target).all(), 'RoI with avg fails assert'
