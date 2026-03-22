"""Risk-coverage (selective prediction) curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _risk_coverage_data(
    correct: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) arrays sorted by ascending uncertainty."""
    order = np.argsort(uncertainty)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    cumulative_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cumulative_correct / np.arange(1, n + 1)
    return coverage, risk


def plot_risk_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty_scores: dict[str, np.ndarray],
    *,
    title: str = "Risk–Coverage Curve",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot risk vs coverage for each uncertainty measure.

    Parameters
    ----------
    y_true, y_pred : ground-truth and predicted labels.
    uncertainty_scores : ``{name: array}`` mapping method names to per-sample
        uncertainty scores (higher = more uncertain).
    """
    correct = (y_true == y_pred).astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scores in uncertainty_scores.items():
        cov, risk = _risk_coverage_data(correct, scores)
        ax.plot(cov, risk, label=name)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (error rate)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
