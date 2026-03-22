"""Reliability (calibration) diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modelscope.metrics.calibration import reliability_diagram_data


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
    *,
    title: str = "Reliability Diagram",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a reliability diagram with a gap bar overlay and sample-count histogram."""
    data = reliability_diagram_data(probs, labels, num_bins)
    bin_centers = (data["bin_edges"][:-1] + data["bin_edges"][1:]) / 2
    width = 1.0 / num_bins

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    nonempty = data["bin_counts"] > 0
    acc = np.where(nonempty, data["bin_accuracies"], np.nan)
    conf = np.where(nonempty, data["bin_confidences"], np.nan)

    ax_rel.bar(
        bin_centers, acc, width=width * 0.9, color="#4C72B0", edgecolor="white", label="Accuracy"
    )
    gap = np.where(nonempty, acc - conf, 0.0)
    ax_rel.bar(
        bin_centers,
        gap,
        bottom=conf,
        width=width * 0.9,
        color="#DD8452",
        edgecolor="white",
        alpha=0.5,
        label="Gap",
    )
    ax_rel.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax_rel.set_ylabel("Accuracy")
    ax_rel.set_title(title)
    ax_rel.legend(loc="upper left", fontsize=8)
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)

    ax_hist.bar(bin_centers, data["bin_counts"], width=width * 0.9, color="#4C72B0", edgecolor="white")
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
