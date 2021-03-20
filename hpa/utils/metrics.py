import numpy as np
import torch

# default value to use for selecting classes from probability outputs
DEFAULT_PROB_CUTOFF = 0.4


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exact_matchs(logits, labels, prob_cutoff=DEFAULT_PROB_CUTOFF):
    # cast to numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # threshold probabilities to one-hot predictions
    probs = sigmoid(logits)
    preds = np.zeros(probs.shape)
    preds[probs > prob_cutoff] = 1.0
    return (preds == labels).all(axis=1)


def f1_scores(logits, labels, prob_cutoff=DEFAULT_PROB_CUTOFF, eps=1e-9):
    # cast to numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # threshold probabilities to one-hot predictions
    probs = sigmoid(logits)
    preds = np.zeros(probs.shape)
    preds[probs > prob_cutoff] = 1.0

    true_pos = ((preds == labels) & (preds != 0) & (labels != 0)).sum(axis=1)
    total_preds = preds.sum(axis=1)
    total_labels = labels.sum(axis=1)

    precs = true_pos / total_preds
    recall = true_pos / total_labels
    f1_score = 2 * (precs * recall) / (precs + recall + eps)
    return f1_score
