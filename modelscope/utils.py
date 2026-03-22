"""Shared utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run *model* over *loader* and return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1]
        logits = model(inputs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
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
