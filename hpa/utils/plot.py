import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_sample(imgs, figsize=(10, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ((ax_blue, ax_green), (ax_red, ax_yellow)) = axes

    ax_blue.imshow(imgs['blue'])
    ax_green.imshow(imgs['green'])
    ax_red.imshow(imgs['red'])
    ax_yellow.imshow(imgs['yellow'])

    ax_blue.set_title('Blue (Nuclei)')
    ax_green.set_title('Green (Target)')
    ax_red.set_title('Red (Microtubles)')
    ax_yellow.set_title('Yellow (ER)')

    for _, ax in np.ndenumerate(axes):
        ax.axis('off')
    plt.show()


def visualize_blend(imgs, size=None):
    r_channel = imgs['red'] + imgs['yellow']
    g_channel = imgs['green'] + imgs['yellow'] / 2
    b_channel = imgs['blue']

    img = np.stack([r_channel, g_channel, b_channel])
    img = np.clip(img, a_min=0, a_max=255)
    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))

    img = Image.fromarray(img)
    if size is not None:
        img = img.resize(size)
    return img
