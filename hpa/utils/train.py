import csv
from statistics import mean

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hpa.model.loss import FocalLoss


# default value to use for selecting classes from probability outputs
DEFAULT_PROB_CUTOFF = 0.4


class Logger:
    """A CSV logger
    Parameters
    ----------
    filepath: str
        The filepath where the logger will be created.
    header: list[str]
        The columns for the CSV file as a list of strings
    """

    def __init__(self, filepath, header):
        self.filepath = filepath
        self.header = header
        with open(filepath, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def add_entry(self, *args):
        """Append a row to the CSV file
        The arguments for this function must match the length and order of the initialized headers.
        """
        if len(args) != len(self.header):
            raise ValueError('Entry length must match the header length!')
        with open(self.filepath, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(args)


def exact_matchs(probs, labels, prob_cutoff=DEFAULT_PROB_CUTOFF):
    # cast to numpy if necessary
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # threshold probabilities to one-hot predictions
    preds = np.zeros(probs.shape)
    preds[probs > prob_cutoff] = 1.0
    return (preds == labels).all(axis=1)


def f1_scores(probs, labels, prob_cutoff=DEFAULT_PROB_CUTOFF, eps=1e-9):
    # cast to numpy if necessary
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # threshold probabilities to one-hot predictions
    preds = np.zeros(probs.shape)
    preds[probs > prob_cutoff] = 1.0

    true_pos = ((preds == labels) & (preds != 0) & (labels != 0)).sum(axis=1)
    total_preds = preds.sum(axis=1)
    total_labels = labels.sum(axis=1)

    precs = true_pos / total_preds
    recall = true_pos / total_labels
    f1_score = 2 * (precs * recall) / (precs + recall + eps)
    return f1_score


def train_epoch(model,
                dataloader,
                criterion,
                optimizer,
                device,
                clip_grad_value=None,
                progress=False,
                epoch=None,
                n_batches=None):
    """Train the model for an epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: str or torch.device
    clip_grad_value: float, optional
    progress: bool, optional
    epoch: int, optional
    n_batches: int, optional

    Returns
    -------
    float
        The average loss
    """
    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    avg_loss = []
    model.train()
    for batch_image, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)
        optimizer.zero_grad()
        output = model(batch_image)
        loss = criterion(output, batch_label)
        loss.backward()
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()
        avg_loss.append(loss.item())
    return mean(avg_loss)


def test_epoch(model,
               dataloader,
               criterion,
               device,
               calc_bce=False,
               calc_focal=False,
               progress=False,
               epoch=None,
               n_batches=None):
    """Run the model for a test epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    device: str or torch.device
    progress: bool, optional
    epoch: int, optional
    n_batches: int, optional

    Returns
    -------
    float, float
        The average loss and the average accuracy
    """
    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (testing)', total=n_batches)
    else:
        generator = dataloader

    if calc_bce:
        bce_fn = BCEWithLogitsLoss()
        avg_bce_loss = []
    if calc_focal:
        focal_fn = FocalLoss()
        avg_focal_loss = []

    avg_loss = []
    model.eval()
    with torch.no_grad():
        for batch_image, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_image)
            loss = criterion(output, batch_label)
            avg_loss.append(loss.item())

            if calc_bce:
                bce_loss = bce_fn(output, batch_label)
                avg_bce_loss.append(bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(output, batch_label)
                avg_focal_loss.append(focal_loss.item())

    # package the results and return
    result = [mean(avg_loss)]
    if calc_bce:
        result.append(mean(avg_bce_loss))
    if calc_focal:
        result.append(mean(avg_focal_loss))
    if len(result) == 1:
        return result[0]
    return tuple(result)


def train_puzzlecam_epoch(model,
                          dataloader,
                          criterion,
                          reg_criterion,
                          optimizer,
                          device,
                          reg_alpha=1.0,
                          clip_grad_value=None,
                          progress=False,
                          epoch=None,
                          n_batches=None):
    """Train the model for an epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    reg_criterion: callable loss function
    optimizer: pytorch optimizer
    device: str or torch.device
    reg_alpha: float, optional
    clip_grad_value: float, optional
    progress: bool, optional
    epoch: int, optional
    n_batches: int, optional

    Returns
    -------
    tuple[float]
        The average losses
    """
    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    avg_loss = []
    avg_full_loss = []
    avg_tile_loss = []
    avg_reg_loss = []
    model.train()
    for batch_image, batch_label in generator:
        # copy the data onto the device
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)

        # run the batch through the model
        optimizer.zero_grad()
        full_class_maps, full_class_scores, tile_class_maps, tile_class_scores = model(batch_image)

        # calculate (1) full class loss, (2) tile class loss, (3) regularization term
        loss_full = criterion(full_class_scores, batch_label)
        loss_tile = criterion(tile_class_scores, batch_label)
        loss_reg = reg_criterion(full_class_maps, tile_class_maps)

        # combine the losses and backward propagate
        loss = loss_full + loss_tile + reg_alpha * loss_reg
        loss.backward()

        # clip the gradient and then descend
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()

        # store the losses
        avg_loss.append(loss.item())
        avg_full_loss.append(loss_full.item())
        avg_tile_loss.append(loss_tile.item())
        avg_reg_loss.append(loss_reg.item())
    return mean(avg_loss), mean(avg_full_loss), mean(avg_tile_loss), mean(avg_reg_loss)


def train_epoch_with_segmentation(model,
                                  dataloader,
                                  classify_criterion,
                                  segment_criterion,
                                  optimizer,
                                  device,
                                  w_classify=0.5,
                                  w_segment=0.5,
                                  clip_grad_value=None,
                                  progress=False,
                                  epoch=None,
                                  n_batches=None):
    """Train the model for an epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    classify_criterion: callable loss function
    optimizer: pytorch optimizer
    device: str or torch.device
    clip_grad_value: float, optional
    progress: bool, optional
    epoch: int, optional
    n_batches: int, optional

    Returns
    -------
    tuple
    """
    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    avg_loss = []
    avg_classify_loss = []
    avg_segment_loss = []
    model.train()
    for batch_image, batch_seg, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_seg = batch_seg.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()
        class_scores, pred_seg = model(batch_image)

        classify_loss = classify_criterion(class_scores, batch_label)
        segment_loss = segment_criterion(pred_seg, batch_seg)
        loss = w_classify * classify_loss + w_segment * segment_loss

        loss.backward()
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()

        avg_loss.append(loss.item())
        avg_classify_loss.append(classify_loss.item())
        avg_segment_loss.append(segment_loss.item())
    return mean(avg_loss), mean(avg_classify_loss), mean(avg_segment_loss)


def test_epoch_with_segmentation(model,
                                 dataloader,
                                 classify_criterion,
                                 segment_criterion,
                                 device,
                                 w_classify=0.5,
                                 w_segment=0.5,
                                 calc_bce=False,
                                 calc_focal=False,
                                 progress=False,
                                 epoch=None,
                                 n_batches=None):
    """Run the model for a test epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    device: str or torch.device
    progress: bool, optional
    epoch: int, optional
    n_batches: int, optional

    Returns
    -------
    tuple
    """
    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (testing)', total=n_batches)
    else:
        generator = dataloader

    if calc_bce:
        bce_fn = BCEWithLogitsLoss()
        avg_bce_loss = []
    if calc_focal:
        focal_fn = FocalLoss()
        avg_focal_loss = []

    avg_loss = []
    avg_classify_loss = []
    avg_segment_loss = []
    model.eval()
    with torch.no_grad():
        for batch_image, batch_seg, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)

            class_scores, pred_seg = model(batch_image)

            classify_loss = classify_criterion(class_scores, batch_label)
            segment_loss = segment_criterion(pred_seg, batch_seg)
            loss = w_classify * classify_loss + w_segment * segment_loss

            avg_loss.append(loss.item())
            avg_classify_loss.append(classify_loss.item())
            avg_segment_loss.append(segment_loss.item())

            if calc_bce:
                bce_loss = bce_fn(class_scores, batch_label)
                avg_bce_loss.append(bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(class_scores, batch_label)
                avg_focal_loss.append(focal_loss.item())

    # package the results and return
    result = [mean(avg_loss), mean(avg_classify_loss), mean(avg_segment_loss)]
    if calc_bce:
        result.append(mean(avg_bce_loss))
    if calc_focal:
        result.append(mean(avg_focal_loss))
    return tuple(result)


def checkpoint(model, filepath):
    """Save the state of the model
    To restore the model do the following:
    >> the_model = TheModelClass(*args, **kwargs)
    >> the_model.load_state_dict(torch.load(PATH))
    Parameters
    ----------
    model: nn.Module
        The pytorch model to be saved
    filepath: str
        The filepath of the pickle
    """
    torch.save(model.state_dict(), filepath)
