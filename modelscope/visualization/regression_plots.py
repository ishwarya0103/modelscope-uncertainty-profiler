"""Visualization helpers for regression tasks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Predicted vs Actual",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs actual values with a perfect-prediction diagonal."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true.ravel(), y_pred.ravel(), alpha=0.3, s=8, color="#4C72B0")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_bins: int = 50,
    title: str = "Residual Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of residuals (y_true - y_pred)."""
    residuals = (y_true - y_pred).ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=num_bins, color="#4C72B0", edgecolor="white", density=True)
    ax.axvline(0, color="k", linestyle="--", lw=1)
    ax.axvline(residuals.mean(), color="#DD8452", linestyle="-", lw=1.5, label=f"mean={residuals.mean():.3f}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_points: int = 200,
    title: str = "Conformal Prediction Intervals",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot prediction intervals for a subsample of test points.

    Points are sorted by predicted value for readability.
    """
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    lo = lower.ravel()
    hi = upper.ravel()

    if len(y_t) > max_points:
        idx = np.random.default_rng(42).choice(len(y_t), max_points, replace=False)
        y_t, y_p, lo, hi = y_t[idx], y_p[idx], lo[idx], hi[idx]

    order = np.argsort(y_p)
    y_t, y_p, lo, hi = y_t[order], y_p[order], lo[order], hi[order]
    x = np.arange(len(y_p))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, lo, hi, alpha=0.25, color="#4C72B0", label="Prediction interval")
    ax.plot(x, y_p, ".", markersize=3, color="#4C72B0", label="Predicted")
    covered = (y_t >= lo) & (y_t <= hi)
    ax.scatter(x[covered], y_t[covered], s=8, color="#55A868", label="Covered", zorder=3)
    ax.scatter(x[~covered], y_t[~covered], s=12, color="#C44E52", marker="x", label="Not covered", zorder=3)
    ax.set_xlabel("Sample index (sorted by prediction)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
