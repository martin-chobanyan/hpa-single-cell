import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hpa.model.loss import FocalLoss
from hpa.utils.metrics import calc_exact_matchs, calc_f1_scores, Metrics


def train_roi_epoch(model,
                    dataloader,
                    criterion,
                    optimizer,
                    device,
                    accum_grad=0,
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
    accum_grad: int, optional
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

    batch_count = 0
    for batch_image, batch_mask, batch_counts, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_mask = batch_mask.to(device)
        batch_counts = batch_counts.to(device)
        batch_label = batch_label.to(device)

        logits = model(batch_image, batch_mask, batch_counts)
        loss = criterion(logits, batch_label)
        loss.backward()

        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        if batch_count == accum_grad:
            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0
        else:
            batch_count += 1
        metrics.insert('loss', loss.item())

    # handle any left-over gradients waiting to be descended
    if batch_count != 0:
        optimizer.step()
        optimizer.zero_grad()
    return metrics.average()


def test_roi_epoch(model,
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
    tuple[float]
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
    metric_names += ['exact', 'f1']
    metrics = Metrics(metric_names)

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (testing)', total=n_batches)
    else:
        generator = dataloader

    with torch.no_grad():
        for batch_image, batch_mask, batch_counts, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_mask = batch_mask.to(device)
            batch_counts = batch_counts.to(device)
            batch_label = batch_label.to(device)

            logits = model(batch_image, batch_mask, batch_counts)
            loss = criterion(logits, batch_label)
            metrics.insert('loss', loss.item())

            if calc_bce:
                bce_loss = bce_fn(logits, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(logits, batch_label)
                metrics.insert('focal_loss', focal_loss.item())

            exact_hits = calc_exact_matchs(logits, batch_label)
            f1_values = calc_f1_scores(logits, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
    return metrics.average()


def train_peak_roi_epoch(model,
                         dataloader,
                         criterion,
                         optimizer,
                         device,
                         accum_grad=0,
                         w_roi=0.5,
                         w_peak=0.5,
                         clip_grad_value=None,
                         progress=False,
                         epoch=None,
                         n_batches=None):
    model.train()
    metrics = Metrics(['loss', 'roi_loss', 'peak_loss'])

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    batch_count = 0
    for batch_image, batch_mask, batch_counts, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_mask = batch_mask.to(device)
        batch_counts = batch_counts.to(device)
        batch_label = batch_label.to(device)

        roi_logits, peak_logits = model(batch_image, batch_mask, batch_counts)
        roi_loss = criterion(roi_logits, batch_label)
        peak_loss = criterion(peak_logits, batch_label)

        loss = w_roi * roi_loss + w_peak * peak_loss
        loss.backward()

        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        if batch_count == accum_grad:
            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0
        else:
            batch_count += 1
        metrics.insert('loss', loss.item())
        metrics.insert('roi_loss', roi_loss.item())
        metrics.insert('peak_loss', peak_loss.item())

    # handle any left-over gradients waiting to be descended
    if batch_count != 0:
        optimizer.step()
        optimizer.zero_grad()
    return metrics.average()


def test_peak_roi_epoch(model,
                        dataloader,
                        criterion,
                        device,
                        w_roi=0.5,
                        w_peak=0.5,
                        calc_bce=False,
                        calc_focal=False,
                        progress=False,
                        epoch=None,
                        n_batches=None):
    model.eval()

    metric_names = ['loss', 'roi_loss', 'peak_loss']
    if calc_bce:
        bce_fn = BCEWithLogitsLoss()
        metric_names.append('bce_loss')
    if calc_focal:
        focal_fn = FocalLoss()
        metric_names.append('focal_loss')
    metric_names += ['exact', 'f1']
    metrics = Metrics(metric_names)

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (testing)', total=n_batches)
    else:
        generator = dataloader

    with torch.no_grad():
        for batch_image, batch_mask, batch_counts, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_mask = batch_mask.to(device)
            batch_counts = batch_counts.to(device)
            batch_label = batch_label.to(device)

            roi_logits, peak_logits = model(batch_image, batch_mask, batch_counts)
            roi_loss = criterion(roi_logits, batch_label)
            peak_loss = criterion(peak_logits, batch_label)
            loss = w_roi * roi_loss + w_peak * peak_loss

            metrics.insert('loss', loss.item())
            metrics.insert('roi_loss', roi_loss.item())
            metrics.insert('peak_loss', peak_loss.item())

            if calc_bce:
                bce_loss = bce_fn(roi_logits, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(roi_logits, batch_label)
                metrics.insert('focal_loss', focal_loss.item())

            exact_hits = calc_exact_matchs(roi_logits, batch_label)
            f1_values = calc_f1_scores(roi_logits, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
    return metrics.average()
