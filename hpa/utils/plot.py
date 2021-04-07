from cv2 import resize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..data.transforms import ToBinaryCellSegmentation


def plot_example(cell_img, green_img, seg, figsize=(18, 6)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.imshow(cell_img)
    ax1.set_title('Reference Image')
    ax1.axis('off')

    ax2.imshow(green_img)
    ax2.set_title('Target Proteins')
    ax2.axis('off')

    ax3.imshow(seg)
    ax3.set_title('Cell Segmentation')
    ax3.axis('off')
    plt.show()


def plot_predicted_probs(probs, tgt_class_idx, figsize=(15, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    bar_idx = np.arange(len(probs))
    ax.bar(bar_idx, probs, color='gray', alpha=0.8, label='')
    if len(tgt_class_idx) > 0:
        ax.bar(tgt_class_idx, probs[tgt_class_idx], color='blue', alpha=0.8, label='Selected Classes')
    ax.set_xticks(bar_idx)
    ax.set_title('Class Predictions')
    plt.legend()
    return ax


def overlay_heatmaps(cell_img, heatmaps, pred_segs, alpha1=0.75, alpha2=0.5):
    """Overlay the predicted heatmaps and segmentation maps over the input image

    Parameters
    ----------
    cell_img: PIL.Image
    heatmaps: dict[int, numpy.ndarray]
    pred_segs: dict[int, numpy.ndarray]
    alpha1: float, optional
    alpha2: float, optional
    """
    img_dim = cell_img.size[0]
    for label_id, pred in pred_segs.items():
        heatmap = resize(heatmaps[label_id], (img_dim, img_dim))
        pred = pred_segs[label_id]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        colors = 'gray'
        ax1.imshow(cell_img)
        ax1.imshow(heatmap, cmap=colors, vmin=0, vmax=1, alpha=alpha1)
        ax1.set_title(f'Label: {label_id} with heatmap')
        ax1.axis('off')

        colors = 'seismic'
        ax2.imshow(heatmap, cmap=colors, vmin=0, vmax=1)
        ax2.set_title('Heatmap')
        ax2.axis('off')

        colors = 'inferno'
        ax3.imshow(cell_img.resize(pred.shape))
        ax3.imshow(pred, cmap=colors, alpha=alpha2)
        ax3.set_title(f'Label: {label_id} with filtered heatmap')
        ax3.axis('off')
        plt.show()


def overlay_peaks(cell_img, seg, peaks, alpha1=0.6, alpha2=0.8):
    """Overlay the class response peaks

    Parameters
    ----------
    cell_img: PIL.Image
    seg: numpy.ndarray
    peaks: dict[int, numpy.ndarray]
    alpha1: float, optional
    alpha2: float, optional
    """
    img_dim = cell_img.size[0]
    seg_img = ToBinaryCellSegmentation()(image=seg)['image']
    seg_img = resize(seg_img, (img_dim, img_dim))
    for label_id, channel_peaks in peaks.items():
        heatmap = resize(channel_peaks, (img_dim, img_dim))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        colors = 'gray'
        ax1.imshow(cell_img)
        ax1.imshow(heatmap, cmap=colors, vmin=0, vmax=1, alpha=alpha1)
        ax1.set_title(f'Label: {label_id} with heatmap')
        ax1.axis('off')

        colors = 'seismic'
        ax2.imshow(heatmap, cmap=colors, vmin=0, vmax=1)
        ax2.set_title('Heatmap')
        ax2.axis('off')

        colors = 'gray'
        ax3.imshow(seg_img)
        ax3.imshow(heatmap, cmap=colors, vmin=0, vmax=1, alpha=alpha2)
        ax3.set_title(f'Label: {label_id} with filtered heatmap')
        ax3.axis('off')
        plt.show()


def overlay_cell_assignments(cells, seg, cmap='PRGn', figsize=(10, 10)):
    """Plot the cell segmentation with the predicted cell labels

    Note: Slight distortion in label placement due to cell center calculation

    Parameters
    ----------
    cells: list[hpa.infer.cells.Cell]
    seg: numpy.ndarray
    cmap: str
    figsize: tuple[int]
    """
    fig, ax = plt.subplots(figsize=figsize)
    to_binary = ToBinaryCellSegmentation()
    ax.imshow(to_binary(image=seg)['image'], cmap=cmap)
    ax.axis('off')

    num_negative = 0
    for cell in cells:
        pred_labels = sorted([str(label_id) for label_id, _ in cell.preds])
        pred_labels = '|'.join(pred_labels)
        num_negative += '18' in pred_labels
        ys, xs = np.where(cell.cell_mask)
        ax.annotate(f'{pred_labels}', (np.mean(xs), np.mean(ys)), color='white', weight='bold')
    ax.set_title(f'Predictions ({num_negative} negative cells)')
    plt.show()


def plot_sample(imgs, figsize=(10, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ((ax_green, ax_blue), (ax_red, ax_yellow)) = axes

    ax_blue.imshow(imgs['blue'], cmap='Blues')
    ax_green.imshow(imgs['green'], cmap='Greens')
    ax_red.imshow(imgs['red'], cmap='Reds')
    ax_yellow.imshow(imgs['yellow'], cmap='YlOrBr')

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


def visualize_cell_mask(mask, imgs):
    binary_mask = mask.copy()
    binary_mask[binary_mask != 0] = 255
    binary_mask = binary_mask.astype(np.uint8)

    blend = visualize_blend(imgs)
    seg_mask = Image.fromarray(binary_mask)
    blend.putalpha(seg_mask)
    return blend
