from hpa.data import NEGATIVE_LABEL, N_CLASSES

DEFAULT_CONFIDENCE = 0.5


def assign_cell_labels(cells, pred_map, intersect_cutoff, confidence_map=None):
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


def get_percent_labeled_cells(cells, verbose=True):
    count = 0
    for cell in cells:
        if sorted(list(cell.preds))[0] != 18:
            count += 1
    if verbose:
        print(f'{round(100 * count / len(cells), 2)}% of cells labeled')
    return count


def print_assigned_labels(cells):
    for cell in cells:
        print(f'Cell {cell.cell_id}: {sorted(list(cell.preds))}')
