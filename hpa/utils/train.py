import csv

import torch
from tqdm import tqdm


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


def train_epoch(model, dataloader, criterion, optimizer, device, progress=False, epoch=None, n_batches=None):
    """Train the model for an epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: str or torch.device
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
        optimizer.step()
        avg_loss.append(loss.item())
    return sum(avg_loss) / len(avg_loss)


def test_epoch(model, dataloader, criterion, device, progress=False, epoch=None, n_batches=None):
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

    avg_loss = []
    model.eval()
    with torch.no_grad():
        for batch_image, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_image)
            loss = criterion(output, batch_label)
            avg_loss.append(loss.item())
    return sum(avg_loss) / len(avg_loss)


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
