"""Uncertainty vs error visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty_scores: dict[str, np.ndarray],
    *,
    num_bins: int = 20,
    title: str = "Uncertainty vs. Error",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bin samples by uncertainty and plot mean error per bin.

    For classification *error* is 1 - accuracy within the bin.
    """
    correct = (y_true == y_pred).astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scores in uncertainty_scores.items():
        order = np.argsort(scores)
        bins = np.array_split(order, num_bins)
        mean_unc = [scores[b].mean() for b in bins if len(b) > 0]
        mean_err = [1.0 - correct[b].mean() for b in bins if len(b) > 0]
        ax.plot(mean_unc, mean_err, "o-", markersize=3, label=name)

    ax.set_xlabel("Mean uncertainty (binned)")
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
