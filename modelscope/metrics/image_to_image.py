"""Image-to-image quality metrics: PSNR, SSIM, per-pixel MAE/MSE."""

from __future__ import annotations

import numpy as np


def psnr(y_true: np.ndarray, y_pred: np.ndarray, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio (averaged over samples).

    *y_true*, *y_pred* have shape ``(N, ...)``.
    """
    if data_range is None:
        data_range = float(y_true.max() - y_true.min())
        if data_range == 0:
            data_range = 1.0

    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=tuple(range(1, y_true.ndim)))
    mse_per_sample = np.clip(mse_per_sample, 1e-12, None)
    psnr_per_sample = 10.0 * np.log10(data_range**2 / mse_per_sample)
    return float(psnr_per_sample.mean())


def _ssim_single(img1: np.ndarray, img2: np.ndarray, data_range: float) -> float:
    """SSIM for a single pair of images (spatial dims only, no batch)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


def ssim(y_true: np.ndarray, y_pred: np.ndarray, data_range: float | None = None) -> float:
    """Mean structural similarity index (averaged over samples).

    Uses the simplified global SSIM (no sliding window) which is fast and
    sufficient for profiling purposes.
    """
    if data_range is None:
        data_range = float(y_true.max() - y_true.min())
        if data_range == 0:
            data_range = 1.0

    scores = [_ssim_single(y_true[i], y_pred[i], data_range) for i in range(len(y_true))]
    return float(np.mean(scores))


def image_to_image_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_range: float | None = None,
) -> dict[str, float]:
    """Compute image-to-image quality metrics.

    *y_true* and *y_pred* should have shape ``(N, C, H, W)`` or ``(N, H, W)``.
    """
    flat_true = y_true.reshape(len(y_true), -1)
    flat_pred = y_pred.reshape(len(y_pred), -1)

    pixel_mae = float(np.mean(np.abs(flat_true - flat_pred)))
    pixel_mse = float(np.mean((flat_true - flat_pred) ** 2))
    pixel_rmse = float(np.sqrt(pixel_mse))

    return {
        "psnr": psnr(y_true, y_pred, data_range),
        "ssim": ssim(y_true, y_pred, data_range),
        "pixel_mae": pixel_mae,
        "pixel_mse": pixel_mse,
        "pixel_rmse": pixel_rmse,
        "num_samples": len(y_true),
    }
