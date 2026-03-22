"""Standard classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
    average: str = "macro",
) -> dict[str, float]:
    """Compute standard classification metrics.

    Returns a flat dict suitable for JSON serialization.
    """
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    binary = num_classes == 2
    avg = "binary" if binary else average
    zero_div = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=zero_div)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=zero_div)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=zero_div)),
        "num_classes": num_classes,
        "num_samples": len(y_true),
    }
