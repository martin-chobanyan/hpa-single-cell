from copy import deepcopy


class Cell:
    def __init__(self, cell_id, mask):
        self.cell_id = cell_id
        self.mask = mask
        self.preds = []

    def isolate_cell(self, img, background=0):
        img_copy = deepcopy(img)
        img_copy[~self.mask] = background
        return img_copy

    def add_prediction(self, label):
        self.preds.append(label)
