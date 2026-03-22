"""MC-Dropout uncertainty estimation for classification and regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from modelscope.utils import to_probabilities

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def _enable_dropout(model: torch.nn.Module) -> int:
    """Set all Dropout layers to training mode and return how many were found."""
    count = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            count += 1
    return count


def _check_dropout(model: torch.nn.Module) -> None:
    n = _enable_dropout(model)
    if n == 0:
        raise RuntimeError(
            "MC-Dropout requested but the model has no Dropout layers. "
            "Add dropout or remove 'mc_dropout' from uncertainty_methods."
        )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    loader: DataLoader,
    n_samples: int = 30,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Run *n_samples* stochastic forward passes with dropout enabled.

    Returns
    -------
    dict with keys:
        ``mean_probs``  : (N, C)  mean predicted probabilities.
        ``predictive_entropy`` : (N,) entropy of the mean prediction.
        ``mutual_information`` : (N,) approx epistemic uncertainty (MI).
        ``variance``    : (N, C) per-class variance across samples.
        ``labels``      : (N,)   ground-truth labels.
    """
    model.eval()
    _check_dropout(model)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    collected_labels = False

    for _t in range(n_samples):
        sample_logits = []
        sample_labels = []
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1]
            logits = model(inputs).cpu().numpy()
            sample_logits.append(logits)
            if not collected_labels:
                sample_labels.append(labels.numpy())
        all_probs.append(to_probabilities(np.concatenate(sample_logits, axis=0)))
        if not collected_labels:
            all_labels = [np.concatenate(sample_labels, axis=0)]
            collected_labels = True

    model.eval()

    stacked = np.stack(all_probs, axis=0)  # (T, N, C)
    mean_probs = stacked.mean(axis=0)  # (N, C)
    variance = stacked.var(axis=0)  # (N, C)

    eps = 1e-12
    pred_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, eps, 1.0)), axis=-1)

    per_sample_entropy = -np.sum(
        stacked * np.log(np.clip(stacked, eps, 1.0)), axis=-1
    )  # (T, N)
    mean_entropy = per_sample_entropy.mean(axis=0)  # (N,)
    mutual_info = pred_entropy - mean_entropy

    return {
        "mean_probs": mean_probs,
        "predictive_entropy": pred_entropy,
        "mutual_information": mutual_info,
        "variance": variance,
        "labels": all_labels[0],
    }


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_dropout_predict_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    n_samples: int = 30,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """MC-Dropout for regression: mean prediction + variance.

    Returns
    -------
    dict with keys:
        ``mean_preds`` : (N, ...) mean prediction across stochastic passes.
        ``variance``   : (N, ...) per-element variance.
        ``std``        : (N, ...) per-element standard deviation.
        ``targets``    : (N, ...) ground-truth targets.
    """
    model.eval()
    _check_dropout(model)

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    collected_targets = False

    for _t in range(n_samples):
        sample_preds = []
        sample_targets = []
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1]
            out = model(inputs).cpu().numpy()
            sample_preds.append(out)
            if not collected_targets:
                sample_targets.append(targets.numpy())
        all_preds.append(np.concatenate(sample_preds, axis=0))
        if not collected_targets:
            all_targets = [np.concatenate(sample_targets, axis=0)]
            collected_targets = True

    model.eval()

    stacked = np.stack(all_preds, axis=0)  # (T, N, ...)
    mean_preds = stacked.mean(axis=0)
    variance = stacked.var(axis=0)

    return {
        "mean_preds": mean_preds,
        "variance": variance,
        "std": np.sqrt(variance),
        "targets": all_targets[0],
    }
