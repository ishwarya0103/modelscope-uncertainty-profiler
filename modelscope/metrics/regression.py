"""Standard regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression metrics.

    *y_true* and *y_pred* can be ``(N,)`` for scalar regression or ``(N, D)``
    for multi-output.  Metrics are averaged across outputs.
    """
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    explained_var = float(explained_variance_score(y_true, y_pred))

    residuals = (y_true - y_pred).ravel()

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": explained_var,
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "num_samples": len(y_true),
    }
