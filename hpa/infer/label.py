from hpa.data import NEGATIVE_LABEL


def assign_cell_labels(cells, pred_map, intersect_cutoff):
    for cell in cells:
        negative = True
        for label_id, seg_mask in pred_map.items():
            p_intersect = cell.calc_intersect(seg_mask)
            if p_intersect > intersect_cutoff:
                cell.add_prediction(label_id)
                negative = False
        if negative:
            cell.add_prediction(NEGATIVE_LABEL)
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
