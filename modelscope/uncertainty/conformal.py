"""Conformal prediction methods.

Split conformal for regression/image-to-image and conformal classification sets.
These provide distribution-free coverage guarantees using only a held-out
calibration set — no retraining required.
"""

from __future__ import annotations

import numpy as np

from modelscope.utils import to_probabilities


# ---------------------------------------------------------------------------
# Regression / image-to-image: residual-based split conformal
# ---------------------------------------------------------------------------

def calibrate_residual_conformal(
    y_calib: np.ndarray,
    y_pred_calib: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, float | np.ndarray]:
    """Compute the conformal threshold *tau* from calibration residuals.

    Parameters
    ----------
    y_calib : ground-truth values on the calibration set.
    y_pred_calib : model predictions on the calibration set.
    alpha : miscoverage level.  ``0.1`` → 90 % target coverage.

    Returns
    -------
    dict with ``tau`` (the threshold), ``residuals``, and ``quantile_level``.

    For scalar regression shapes are ``(N,)`` or ``(N, D)``.
    For image-to-image they can be ``(N, C, H, W)`` — residuals are computed
    element-wise and the quantile is taken over all elements.
    """
    residuals = np.abs(y_calib.ravel() - y_pred_calib.ravel()).astype(np.float64)
    n = len(residuals)
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    tau = float(np.quantile(residuals, quantile_level))

    return {
        "tau": tau,
        "residuals": residuals,
        "quantile_level": quantile_level,
        "alpha": alpha,
    }


def predict_residual_conformal(
    y_pred: np.ndarray,
    tau: float,
) -> dict[str, np.ndarray]:
    """Construct prediction intervals ``[y_pred - tau, y_pred + tau]``.

    Returns
    -------
    dict with ``lower``, ``upper``, ``interval_width`` (all same shape as
    *y_pred*).
    """
    return {
        "lower": y_pred - tau,
        "upper": y_pred + tau,
        "interval_width": np.full_like(y_pred, 2.0 * tau, dtype=np.float64),
    }


def evaluate_conformal_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Check empirical coverage and mean interval width."""
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    return {
        "empirical_coverage": float(covered.mean()),
        "mean_interval_width": float((upper - lower).mean()),
        "median_interval_width": float(np.median(upper - lower)),
    }


# ---------------------------------------------------------------------------
# Classification: conformal prediction sets (APS / threshold method)
# ---------------------------------------------------------------------------

def calibrate_conformal_classification(
    logits_calib: np.ndarray,
    labels_calib: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, float | np.ndarray]:
    """Calibrate conformal classification sets using the APS method.

    For each calibration sample, compute the cumulative softmax mass needed to
    include the true label.  The conformal threshold is the
    ``ceil((n+1)*(1-alpha))/n`` quantile of these scores.

    Parameters
    ----------
    logits_calib : ``(N, C)`` logits on the calibration set.
    labels_calib : ``(N,)`` integer ground-truth labels.
    alpha : miscoverage level.

    Returns
    -------
    dict with ``threshold``, ``scores``, ``quantile_level``.
    """
    probs = to_probabilities(logits_calib)
    n, c = probs.shape

    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        ranks[i] = int(np.where(sorted_idx[i] == labels_calib[i])[0][0])

    scores = cumsum[np.arange(n), ranks]

    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold = float(np.quantile(scores, quantile_level))

    return {
        "threshold": threshold,
        "scores": scores,
        "quantile_level": quantile_level,
        "alpha": alpha,
    }


def predict_conformal_sets(
    logits: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Return conformal prediction sets for each sample.

    For each sample, include classes in descending probability order until
    the cumulative probability exceeds *threshold*.
    """
    probs = to_probabilities(logits)
    n, c = probs.shape
    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    sets: list[list[int]] = []
    for i in range(n):
        cutoff = int(np.searchsorted(cumsum[i], threshold, side="left")) + 1
        cutoff = min(cutoff, c)
        sets.append(sorted_idx[i, :cutoff].tolist())
    return sets


def evaluate_conformal_sets(
    prediction_sets: list[list[int]],
    labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate conformal classification sets."""
    n = len(labels)
    covered = sum(1 for i in range(n) if labels[i] in prediction_sets[i])
    sizes = [len(s) for s in prediction_sets]

    return {
        "empirical_coverage": covered / n,
        "mean_set_size": float(np.mean(sizes)),
        "median_set_size": float(np.median(sizes)),
        "singleton_fraction": float(np.mean([1 if s == 1 else 0 for s in sizes])),
    }
