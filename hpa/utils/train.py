import csv
from statistics import mean

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hpa.model.loss import FocalLoss
from hpa.utils.metrics import Metrics


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
    model.train()
    metrics = Metrics('loss')

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

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
        metrics.insert('loss', loss.item())
    return metrics.average()


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
    model.eval()

    metric_names = ['loss']
    if calc_bce:
        bce_fn = BCEWithLogitsLoss()
        metric_names.append('bce_loss')
    if calc_focal:
        focal_fn = FocalLoss()
        metric_names.append('focal_loss')
    metrics = Metrics(metric_names)

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (testing)', total=n_batches)
    else:
        generator = dataloader

    with torch.no_grad():
        for batch_image, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_image)
            loss = criterion(output, batch_label)
            metrics.insert('loss', loss.item())
            if calc_bce:
                bce_loss = bce_fn(output, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(output, batch_label)
                metrics.insert('focal_loss', focal_loss.item())
    return metrics.average()


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
    model.train()
    metrics = Metrics(['loss', 'full_loss', 'tile_loss', 'reg_loss'])

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    for batch_image, batch_label in generator:
        # copy the data onto the device
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)

        # run the batch through the model
        optimizer.zero_grad()
        full_class_maps, full_class_scores, tile_class_maps, tile_class_scores = model(batch_image)

        # calculate (1) full class loss, (2) tile class loss, (3) regularization term
        full_loss = criterion(full_class_scores, batch_label)
        tile_loss = criterion(tile_class_scores, batch_label)
        reg_loss = reg_criterion(full_class_maps, tile_class_maps)

        # combine the losses and backward propagate
        loss = full_loss + tile_loss + reg_alpha * reg_loss
        loss.backward()

        # clip the gradient and then descend
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()

        # store the losses
        metrics.insert('loss', loss.item())
        metrics.insert('full_loss', full_loss.item())
        metrics.insert('tile_loss', tile_loss.item())
        metrics.insert('reg_loss', reg_loss.item())
    return metrics.average()


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
    model.train()
    metrics = Metrics(['loss', 'classify_loss', 'segment_loss'])

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

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

        metrics.insert('loss', loss.item())
        metrics.insert('classify_loss', classify_loss.item())
        metrics.insert('segment_loss', segment_loss.item())
    return metrics.average()


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
    model.eval()

    metric_names = ['loss', 'classify_loss', 'segment_loss']
    if calc_bce:
        bce_fn = BCEWithLogitsLoss()
        metric_names.append('bce_loss')
    if calc_focal:
        focal_fn = FocalLoss()
        metric_names.append('focal_loss')
    metrics = Metrics(metric_names)

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (testing)', total=n_batches)
    else:
        generator = dataloader

    with torch.no_grad():
        for batch_image, batch_seg, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)

            class_scores, pred_seg = model(batch_image)

            classify_loss = classify_criterion(class_scores, batch_label)
            segment_loss = segment_criterion(pred_seg, batch_seg)
            loss = w_classify * classify_loss + w_segment * segment_loss

            metrics.insert('loss', loss.item())
            metrics.insert('classify_loss', classify_loss.item())
            metrics.insert('segment_loss', segment_loss.item())

            if calc_bce:
                bce_loss = bce_fn(class_scores, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(class_scores, batch_label)
                metrics.insert('focal_loss', focal_loss.item())
    return metrics.average()


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
