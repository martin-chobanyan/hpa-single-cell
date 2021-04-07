from collections import defaultdict

import numpy as np

from hpa.data import NEGATIVE_LABEL, N_CLASSES

DEFAULT_CONFIDENCE = 0.5


def assign_cell_labels_v1(cells, pred_map, intersect_cutoff, confidence_map=None):
    # apply a default confidence to all class labels if the mapping is not provided
    if confidence_map is None:
        confidence_map = {label_id: DEFAULT_CONFIDENCE for label_id in range(N_CLASSES)}
    for cell in cells:
        negative = True
        for label_id, seg_mask in pred_map.items():
            p_intersect = cell.calc_intersect(seg_mask)
            if p_intersect > intersect_cutoff:
                confidence = confidence_map[label_id]
                cell.add_prediction(label_id, confidence)
                negative = False
        if negative:
            cell.add_prediction(NEGATIVE_LABEL, DEFAULT_CONFIDENCE)
    return cells


# not successful...
def assign_cell_labels_v2(cells, pred_map, intersect_cutoff, class_probs):
    for cell in cells:
        assigned = False
        for label_id, seg_mask in pred_map.items():
            p_intersect = cell.calc_intersect(seg_mask)
            if p_intersect > intersect_cutoff:
                cell.add_prediction(label_id, class_probs[label_id])
                assigned = True

        # if the current cell has not been assigned a class yet
        if not assigned:
            if len(pred_map) > 0:
                # then give it all of the classes which are above the probability cutoff
                for label_id in pred_map:
                    cell.add_prediction(label_id, class_probs[label_id])
            else:
                # and if no classes are above the cutoff, then simply give it the class with the highest score
                label_id = np.argmax(class_probs)
                cell.add_prediction(label_id, class_probs[label_id])
    return cells


def assign_cell_labels_v3(cells, peak_heatmaps, intersect_cutoff):
    for cell in cells:
        negative = True
        for label_id, heatmap in peak_heatmaps.items():
            p_intersect = cell.calc_intersect(heatmap)
            if p_intersect > intersect_cutoff:
                confidence = cell.calc_confidence(heatmap)
                cell.add_prediction(label_id, confidence)
                negative = False
        if negative:
            cell.add_prediction(NEGATIVE_LABEL, DEFAULT_CONFIDENCE)
    return cells


def assign_cell_labels_v4(cells, peaks, peak_recall_cutoff):
    """Keep channel peak segmentations if the recall of their cell intersection is greater than a threshold

    Parameters
    ----------
    cells: list[hpa.infer.cells.Cell]
    peaks: list[hpa.infer.cells.Segmentation]
    peak_recall_cutoff: float

    Returns
    -------
    list[hpa.infer.cells.Cell]
    """
    cell_results = defaultdict(list)
    for peak in peaks:
        for cell in cells:
            peak_intersection = cell.get_shapely_intersection(peak.geom)
            recall = peak_intersection.area / peak.geom.area
            if recall > peak_recall_cutoff:
                cell_results[cell.cell_id].append(peak)
    for cell in cells:
        max_peaks = defaultdict(int)
        for peak in cell_results[cell.cell_id]:
            max_peaks[peak.label] = max(peak.confidence, max_peaks[peak.label])
        for label, confidence in max_peaks.items():
            cell.add_prediction(label, confidence)
        if len(max_peaks) == 0:
            cell.add_prediction(NEGATIVE_LABEL, DEFAULT_CONFIDENCE)
    return cells


def get_percent_labeled_cells(cells, verbose=True):
    count = 0
    for cell in cells:
        if list(cell.preds)[0][0] != 18:
            count += 1
    if verbose:
        print(f'{round(100 * count / len(cells), 2)}% of cells labeled')
    return count


def print_assigned_labels(cells):
    for cell in cells:
        labels, confidences = zip(*sorted(list(cell.preds)))
        print(f'Cell {cell.cell_id}')
        print(f'Labels: {labels}')
        print(f'Confidences: {confidences}\n')
