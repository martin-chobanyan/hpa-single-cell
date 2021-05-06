import torch
from torch.nn import Conv2d


def get_num_params(model):
    num_total_params = 0
    for p in model.parameters():
        num_total_params += p.numel()
    return num_total_params


def get_num_opt_params(optimizer):
    n_params = 0
    for param_dict in optimizer.param_groups:
        for p in param_dict['params']:
            n_params += p.numel()
    return n_params


def get_num_output_features(cnn):
    final_conv = None
    for m in cnn.modules():
        if isinstance(m, Conv2d):
            final_conv = m
    if final_conv is None:
        raise ValueError('The input model has no Conv2d layers!')
    return final_conv.out_channels


def tile_image_batch(x, n_vertical=2, n_horizontal=2):
    """PUZZLE-CAM utility function

    Tile a batch of images

    Parameters
    ----------
    x: torch.Tensor
        A pytorch tensor of an image batch with shape (batch, channel, height, width)
    n_vertical: int
        The number of tiles along the vertical dimension
    n_horizontal: int
        The number of tiles along the horizontal dimension

    Returns
    -------
    torch.Tensor
        A 4D pytorch tensor with shape:
        dim0: batch * n_vertical * n_horizontal
        dim1: channels
        dim2: height / n_vertical
        dim3: width / n_horizontal
    """
    n_batch, _, img_dim_v, img_dim_h = x.shape
    tile_dim_v = int(img_dim_v / n_vertical)
    tile_dim_h = int(img_dim_h / n_horizontal)

    vertical_cutoffs = list(range(0, img_dim_v, tile_dim_v))
    horizontal_cutoffs = list(range(0, img_dim_h, tile_dim_h))

    tiles = []
    for b in range(n_batch):
        for v in vertical_cutoffs:
            for h in horizontal_cutoffs:
                tile = x[b, :, v:v + tile_dim_v, h:h + tile_dim_h]
                tiles.append(tile)
    return torch.stack(tiles)


def merge_tiles(tiles, n_vertical=2, n_horizontal=2):
    """PUZZLE-CAM utility function

    Parameters
    ----------
    tiles: torch.Tensor
        A pytorch tensor of an image batch with shape (tiles, channel, height, width)
    n_vertical: int
        The number of tiles along the vertical dimension
    n_horizontal: int
        The number of tiles along the horizontal dimension

    Returns
    -------
    torch.Tensor
        A 4D pytorch tensor with shape:
        dim0: batch
        dim1: channels
        dim2: n_vertical * height
        dim3: n_horizontal * width
    """
    n_tiles, n_channels, tile_dim_v, tile_dim_h = tiles.shape
    n_batch = int(n_tiles / (n_vertical * n_horizontal))

    # introduce two new dimensions for both directions of the tiles
    tiles = tiles.view(n_batch, n_vertical, n_horizontal, n_channels, tile_dim_v, tile_dim_h)

    # move the tile direction dimensions before their respective tile dimension
    # shape: (n_batch, n_channels, n_vertical, tile_dim_v, n_horizontal, tile_dim_h)
    tiles = tiles.permute(0, 3, 1, 4, 2, 5)
    tiles = tiles.contiguous()

    # group the last four dimensions based on the direction pairs
    return tiles.view(n_batch, n_channels, (n_vertical * tile_dim_v), (n_horizontal * tile_dim_h))
