import csv

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import interpolate
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hpa.model.loss import BackgroundLoss, ClassHeatmapLoss, FocalLoss
from hpa.utils.metrics import calc_exact_matchs, calc_f1_scores, Metrics


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


class LRScheduler:
    def __init__(self, init_lr=0.00005, min_lr=0.00001, increment=0.00001, delay_start=0):
        self.lr = init_lr
        self.epoch = 0
        self.min_lr = min_lr
        self.increment = increment
        self.delay_start = delay_start

    def update(self):
        if self.epoch >= self.delay_start:
            update_val = max(self.lr - self.increment, self.min_lr)
            if update_val != self.lr:
                print(f'Changing learning rate: {update_val}')
            self.lr = update_val
        self.epoch += 1
        return min(self.lr, 1.0)


def train_epoch(model,
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
    for batch_image, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)

        output = model(batch_image)
        loss = criterion(output, batch_label)
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

            exact_hits = calc_exact_matchs(output, batch_label)
            f1_values = calc_f1_scores(output, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
    return metrics.average()


def train_epoch_with_seg(model,
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
    model: torch.nn.Module
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
        class_maps, class_scores = model(batch_image)

        classify_loss = classify_criterion(class_scores, batch_label)
        segment_loss = segment_criterion(class_maps, batch_seg, batch_label)
        loss = w_classify * classify_loss + w_segment * segment_loss

        loss.backward()
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()

        metrics.insert('loss', loss.item())
        metrics.insert('classify_loss', classify_loss.item())
        metrics.insert('segment_loss', segment_loss.item())
    return metrics.average()


def test_epoch_with_seg(model,
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
    model: torch.nn.Module
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
    metric_names += ['exact', 'f1']
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

            class_maps, class_scores = model(batch_image)

            classify_loss = classify_criterion(class_scores, batch_label)
            segment_loss = segment_criterion(class_maps, batch_seg, batch_label)
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

            exact_hits = calc_exact_matchs(class_scores, batch_label)
            f1_values = calc_f1_scores(class_scores, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
    return metrics.average()


def train_puzzlecam_epoch(model,
                          dataloader,
                          criterion,
                          reg_criterion,
                          seg_criterion,
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
    model: hpa.model.localizers.PuzzleCAM
    dataloader: DataLoader
    criterion: callable loss function
    reg_criterion: callable loss function
    seg_criterion: torch.nn.Module
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
    model.use_tiles = True

    metric_names = ['loss', 'full_loss', 'tile_loss', 'reg_loss', 'seg_loss']
    metrics = Metrics(metric_names)

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    for batch_image, batch_seg, batch_label in generator:
        # copy the data onto the device
        batch_image = batch_image.to(device)
        batch_seg = batch_seg.to(device)
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

        # calculate the segmentation loss (only for metrics, no backpropagation)
        seg_loss = seg_criterion(full_class_maps, batch_seg, batch_label)

        # store the losses
        metrics.insert('loss', loss.item())
        metrics.insert('full_loss', full_loss.item())
        metrics.insert('tile_loss', tile_loss.item())
        metrics.insert('reg_loss', reg_loss.item())
        metrics.insert('seg_loss', seg_loss.item())
    return metrics.average()


def test_puzzlecam_epoch(model,
                         dataloader,
                         criterion,
                         reg_criterion,
                         seg_criterion,
                         device,
                         reg_alpha=1.0,
                         calc_bce=False,
                         calc_focal=False,
                         progress=False,
                         epoch=None,
                         n_batches=None):
    """Train the model for an epoch

    Parameters
    ----------
    model: hpa.model.localizers.PuzzleCAM
    dataloader: DataLoader
    criterion: callable loss function
    reg_criterion: callable loss function
    seg_criterion: torch.nn.Module
    device: str or torch.device
    reg_alpha: float, optional
    progress: bool, optional
    epoch: int, optional
    n_batches: int, optional

    Returns
    -------
    tuple[float]
        The average losses
    """
    model.eval()
    model.use_tiles = True

    metric_names = ['loss', 'full_loss', 'tile_loss', 'reg_loss', 'seg_loss']
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
        for batch_image, batch_seg, batch_label in generator:
            # copy the data onto the device
            batch_image = batch_image.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)

            # run the batch through the model
            full_class_maps, full_class_scores, tile_class_maps, tile_class_scores = model(batch_image)

            # calculate (1) full class loss, (2) tile class loss, (3) regularization term
            full_loss = criterion(full_class_scores, batch_label)
            tile_loss = criterion(tile_class_scores, batch_label)
            reg_loss = reg_criterion(full_class_maps, tile_class_maps)

            # combine the losses and backward propagate
            loss = full_loss + tile_loss + reg_alpha * reg_loss

            # calculate the segmentation loss
            seg_loss = seg_criterion(full_class_maps, batch_seg, batch_label)

            # store the losses
            metrics.insert('loss', loss.item())
            metrics.insert('full_loss', full_loss.item())
            metrics.insert('tile_loss', tile_loss.item())
            metrics.insert('reg_loss', reg_loss.item())
            metrics.insert('seg_loss', seg_loss.item())

            if calc_bce:
                bce_loss = bce_fn(full_class_scores, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(full_class_scores, batch_label)
                metrics.insert('focal_loss', focal_loss.item())

            exact_hits = calc_exact_matchs(full_class_scores, batch_label)
            f1_values = calc_f1_scores(full_class_scores, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
    return metrics.average()


def train_epoch2(model,
                 dataloader,
                 criterion,
                 optimizer,
                 device,
                 clip_grad_value=None,
                 progress=False,
                 epoch=None,
                 n_batches=None,
                 fix_seg_dim=False):
    """Train the model for an epoch

    Parameters
    ----------
    model: hpa.model.localizers.DoublePooledLocalizer
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
    tuple[float]
        The average loss
    """
    model.train()
    metrics = Metrics(['loss', 'segment_loss'])

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    segment_criterion = ClassHeatmapLoss()
    for batch_image, batch_seg, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_seg = batch_seg.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()
        class_maps, class_logits = model(batch_image)

        loss = criterion(class_logits, batch_label)
        loss.backward()
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()

        # calculate the segment loss on the side
        if fix_seg_dim:
            cmap_dim = class_maps.shape[-1]
            batch_seg = interpolate(batch_seg, size=(cmap_dim, cmap_dim), mode='bilinear', align_corners=False)
        segment_loss = segment_criterion(class_maps, batch_seg, batch_label)
        metrics.insert('loss', loss.item())
        metrics.insert('segment_loss', segment_loss.item())
    return metrics.average()


def test_epoch2(model,
                dataloader,
                criterion,
                device,
                calc_bce=False,
                calc_focal=False,
                progress=False,
                epoch=None,
                n_batches=None,
                fix_seg_dim=False):
    """Run the model for a test epoch

    Parameters
    ----------
    model: hpa.model.localizers.DoublePooledLocalizer
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

    metric_names = ['loss', 'segment_loss']
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

    segment_criterion = ClassHeatmapLoss()
    with torch.no_grad():
        for batch_image, batch_seg, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)

            class_maps, class_logits = model(batch_image)
            loss = criterion(class_logits, batch_label)

            if fix_seg_dim:
                cmap_dim = class_maps.shape[-1]
                batch_seg = interpolate(batch_seg, size=(cmap_dim, cmap_dim), mode='bilinear', align_corners=False)
            segment_loss = segment_criterion(class_maps, batch_seg, batch_label)

            metrics.insert('loss', loss.item())
            metrics.insert('segment_loss', segment_loss.item())

            if calc_bce:
                bce_loss = bce_fn(class_logits, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(class_logits, batch_label)
                metrics.insert('focal_loss', focal_loss.item())

            exact_hits = calc_exact_matchs(class_logits, batch_label)
            f1_values = calc_f1_scores(class_logits, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
    return metrics.average()


def train_epoch3(model,
                 dataloader,
                 criterion,
                 optimizer,
                 device,
                 w_classify=0.5,
                 w_background=0.5,
                 clip_grad_value=None,
                 progress=False,
                 epoch=None,
                 n_batches=None,
                 fix_seg_dim=False):
    model.train()
    metrics = Metrics(['loss', 'classify_loss', 'background_loss', 'segment_loss'])

    if progress:
        generator = tqdm(dataloader, desc=f'Epoch {epoch} (training)', total=n_batches)
    else:
        generator = dataloader

    segment_criterion = ClassHeatmapLoss()
    background_criterion = BackgroundLoss()

    for batch_image, batch_seg, batch_label in generator:
        batch_image = batch_image.to(device)
        batch_seg = batch_seg.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()
        class_maps, class_logits = model(batch_image)

        classify_loss = criterion(class_logits, batch_label)
        background_loss = background_criterion(class_maps, batch_seg, batch_label)
        loss = w_classify * classify_loss + w_background * background_loss

        loss.backward()
        if clip_grad_value is not None:
            clip_grad_norm_(model.parameters(), clip_grad_value)
        optimizer.step()

        # calculate the segment loss on the side
        if fix_seg_dim:
            cmap_dim = class_maps.shape[-1]
            batch_seg = interpolate(batch_seg, size=(cmap_dim, cmap_dim), mode='bilinear', align_corners=False)
        segment_loss = segment_criterion(class_maps, batch_seg, batch_label)
        metrics.insert('loss', loss.item())
        metrics.insert('classify_loss', classify_loss.item())
        metrics.insert('background_loss', background_loss.item())
        metrics.insert('segment_loss', segment_loss.item())
    return metrics.average()


def test_epoch3(model,
                dataloader,
                criterion,
                device,
                w_classify=0.5,
                w_background=0.5,
                calc_bce=False,
                calc_focal=False,
                progress=False,
                epoch=None,
                n_batches=None,
                fix_seg_dim=False):
    model.eval()

    metric_names = ['loss', 'classify_loss', 'background_loss', 'segment_loss']
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

    segment_criterion = ClassHeatmapLoss()
    background_criterion = BackgroundLoss()

    with torch.no_grad():
        for batch_image, batch_seg, batch_label in generator:
            batch_image = batch_image.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)

            class_maps, class_logits = model(batch_image)

            classify_loss = criterion(class_logits, batch_label)
            background_loss = background_criterion(class_maps, batch_seg, batch_label)
            loss = w_classify * classify_loss + w_background * background_loss

            if fix_seg_dim:
                cmap_dim = class_maps.shape[-1]
                batch_seg = interpolate(batch_seg, size=(cmap_dim, cmap_dim), mode='bilinear', align_corners=False)
            segment_loss = segment_criterion(class_maps, batch_seg, batch_label)

            metrics.insert('loss', loss.item())
            metrics.insert('classify_loss', classify_loss.item())
            metrics.insert('background_loss', background_loss.item())
            metrics.insert('segment_loss', segment_loss.item())

            if calc_bce:
                bce_loss = bce_fn(class_logits, batch_label)
                metrics.insert('bce_loss', bce_loss.item())
            if calc_focal:
                focal_loss = focal_fn(class_logits, batch_label)
                metrics.insert('focal_loss', focal_loss.item())

            exact_hits = calc_exact_matchs(class_logits, batch_label)
            f1_values = calc_f1_scores(class_logits, batch_label)

            metrics.bulk_insert('exact', exact_hits.tolist())
            metrics.bulk_insert('f1', f1_values.tolist())
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
