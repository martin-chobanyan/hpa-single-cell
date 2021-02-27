import numpy as np


def stitch_four_tile_predictions(top_left, bottom_left, top_right, bottom_right, img_dim):
    """Take the average of four tiled predictions for each of the four corners

    Assumes that the tiles are shaped (num_classes, tile_dim, tile_dim)

    Parameters
    ----------
    top_left: numpy.ndarray
    bottom_left: numpy.ndarray
    top_right: numpy.ndarray
    bottom_right: numpy.ndarray
    img_dim: int
        The dimension for the full sized image

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (num_classes, img_dim, img_dim) where the tiles are stitched together and
        averaged over their overlapping regions
    """
    assert top_left.shape == bottom_left.shape == top_right.shape == bottom_right.shape, 'Tile shapes do not match!'
    num_classes, tile_dim, _ = top_left.shape

    # pad the tiles so that they match the image dimension and stack them together
    tiles = np.full((4, num_classes, img_dim, img_dim), np.nan, dtype=np.float32)
    tiles[0, :, :tile_dim, :tile_dim] = top_left
    tiles[1, :, -tile_dim:, :tile_dim] = bottom_left
    tiles[2, :, :tile_dim, -tile_dim:] = top_right
    tiles[3, :, -tile_dim:, -tile_dim:] = bottom_right
    return np.nanmean(tiles, axis=0)
