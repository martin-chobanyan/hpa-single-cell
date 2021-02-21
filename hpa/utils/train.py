import csv

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hpa.model.loss import FocalLoss


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
    return sum(avg_loss) / len(avg_loss)


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
                avg_bce_loss.append(bce_loss)
            if calc_focal:
                focal_loss = focal_fn(output, batch_label)
                avg_focal_loss.append(focal_loss)

    # package the results and return
    result = [sum(avg_loss) / len(avg_loss)]
    if calc_bce:
        result.append(sum(avg_bce_loss) / len(avg_bce_loss))
    if calc_focal:
        result.append(sum(avg_focal_loss) / len(avg_focal_loss))
    if len(result) == 1:
        return result[0]
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
