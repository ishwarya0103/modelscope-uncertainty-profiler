"""Shared utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run *model* over *loader* and return ``(logits, labels)`` as numpy arrays.

    Works for classification (logits shape ``(N, C)``) and scalar regression
    (output shape ``(N,)`` or ``(N, 1)``).
    """
    model.eval()
    all_outputs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1]
        out = model(inputs)
        all_outputs.append(out.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_outputs).numpy(),
        torch.cat(all_labels).numpy(),
    )


@torch.no_grad()
def collect_image_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run *model* over *loader* for image-to-image tasks.

    Returns ``(predictions, targets)`` with shape ``(N, C, H, W)`` or ``(N, H, W)``.
    If *max_samples* is set, stops after collecting that many samples (to
    avoid OOM on large datasets).
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    count = 0

    for batch in loader:
        inputs, targets = batch[0].to(device), batch[1]
        preds = model(inputs)
        all_preds.append(preds.cpu())
        all_targets.append(targets)
        count += inputs.shape[0]
        if max_samples is not None and count >= max_samples:
            break

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_targets).numpy(),
    )


def to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis (numerically stable)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def ensure_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
