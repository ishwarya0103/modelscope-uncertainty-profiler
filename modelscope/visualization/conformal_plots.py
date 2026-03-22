"""Conformal-prediction-specific visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_coverage_vs_nominal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibration_residuals: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
    title: str = "Empirical Coverage vs Nominal Level",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Sweep over nominal coverage levels and plot empirical coverage.

    A well-calibrated conformal method should track the diagonal.
    """
    if alphas is None:
        alphas = np.linspace(0.01, 0.50, 25)

    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    residuals_sorted = np.sort(calibration_residuals)
    n_calib = len(residuals_sorted)

    nominal = 1.0 - alphas
    empirical = np.zeros_like(alphas)

    for i, alpha in enumerate(alphas):
        q_level = min(np.ceil((n_calib + 1) * (1 - alpha)) / n_calib, 1.0)
        tau = np.quantile(residuals_sorted, q_level)
        covered = ((y_t >= y_p - tau) & (y_t <= y_p + tau)).mean()
        empirical[i] = covered

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(nominal, empirical, "o-", markersize=4, color="#4C72B0", label="Empirical")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    ax.set_xlabel("Nominal coverage (1 - α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_interval_width_distribution(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    num_bins: int = 50,
    title: str = "Prediction Interval Width Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of interval widths."""
    widths = (upper - lower).ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(widths, bins=num_bins, color="#4C72B0", edgecolor="white", density=True)
    ax.axvline(widths.mean(), color="#DD8452", linestyle="-", lw=1.5, label=f"mean={widths.mean():.3f}")
    ax.axvline(np.median(widths), color="#55A868", linestyle="--", lw=1.5, label=f"median={np.median(widths):.3f}")
    ax.set_xlabel("Interval width")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_conformal_set_sizes(
    prediction_sets: list[list[int]],
    *,
    title: str = "Conformal Set Size Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of conformal classification set sizes."""
    sizes = [len(s) for s in prediction_sets]
    max_size = max(sizes)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = np.bincount(sizes, minlength=max_size + 1)
    x = np.arange(len(counts))
    ax.bar(x, counts, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Set size")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
