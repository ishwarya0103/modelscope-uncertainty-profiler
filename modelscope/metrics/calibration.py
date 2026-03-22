"""Calibration metrics: ECE, MCE, Brier score, NLL, reliability diagram data."""

from __future__ import annotations

import numpy as np


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Bin predictions by confidence and compute per-bin accuracy.

    Parameters
    ----------
    probs : (N, C)  predicted probability vectors.
    labels : (N,)   integer ground-truth labels.
    num_bins : number of equal-width bins on [0, 1].

    Returns
    -------
    dict with keys ``bin_edges``, ``bin_confidences``, ``bin_accuracies``,
    ``bin_counts``.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidences = np.zeros(num_bins)
    bin_accuracies = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = correct[mask].mean()

    return {
        "bin_edges": bin_edges,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Expected Calibration Error (weighted by bin size)."""
    data = reliability_diagram_data(probs, labels, num_bins)
    n = data["bin_counts"].sum()
    if n == 0:
        return 0.0
    weights = data["bin_counts"] / n
    return float(np.sum(weights * np.abs(data["bin_accuracies"] - data["bin_confidences"])))


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Maximum Calibration Error."""
    data = reliability_diagram_data(probs, labels, num_bins)
    nonempty = data["bin_counts"] > 0
    if not nonempty.any():
        return 0.0
    return float(np.max(np.abs(data["bin_accuracies"] - data["bin_confidences"])[nonempty]))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error on one-hot probabilities)."""
    n, c = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class."""
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))
