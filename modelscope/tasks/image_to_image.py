"""Image-to-image task evaluator — denoising, reconstruction, depth estimation, etc."""

from __future__ import annotations

from typing import Any

import numpy as np

from modelscope.metrics.image_to_image import image_to_image_metrics
from modelscope.tasks.base import TaskEvaluator
from modelscope.uncertainty.conformal import (
    calibrate_residual_conformal,
    evaluate_conformal_coverage,
    predict_residual_conformal,
)
from modelscope.utils import collect_image_predictions


class ImageToImageEvaluator(TaskEvaluator):
    """Evaluate an image-to-image model: PSNR/SSIM, per-pixel uncertainty, conformal."""

    def evaluate(self, data_splits: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {"task": "image_to_image", "splits": {}}

        conformal_state = self._calibrate_conformal(data_splits)

        for split_name in ("val", "test"):
            loader = data_splits.get(split_name)
            if loader is None:
                continue
            results["splits"][split_name] = self._evaluate_split(
                split_name, loader, data_splits, conformal_state
            )

        return results

    def _calibrate_conformal(self, data_splits: dict[str, Any]) -> dict | None:
        if "conformal_residual" not in self.config.uncertainty_methods:
            return None

        calib_loader = data_splits.get(self.config.calibration_split)
        if calib_loader is None:
            calib_loader = data_splits.get("val")
        if calib_loader is None:
            return None

        preds, targets = collect_image_predictions(
            self.model, calib_loader, self.config.device
        )
        return calibrate_residual_conformal(targets, preds, alpha=self.config.conformal_alpha)

    def _evaluate_split(
        self, name: str, loader, data_splits, conformal_state
    ) -> dict[str, Any]:
        cfg = self.config
        preds, targets = collect_image_predictions(self.model, loader, cfg.device)

        result: dict[str, Any] = {}
        result["metrics"] = image_to_image_metrics(targets, preds)

        pixel_error = np.abs(targets - preds)
        per_sample_mae = pixel_error.reshape(len(pixel_error), -1).mean(axis=1)

        unc_scores: dict[str, np.ndarray] = {}
        unc_scores["per_sample_mae"] = per_sample_mae

        result["uncertainty"] = {}
        for uq_name, scores in unc_scores.items():
            result["uncertainty"][uq_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "median": float(np.median(scores)),
            }

        if conformal_state is not None:
            intervals = predict_residual_conformal(preds, conformal_state["tau"])
            coverage = evaluate_conformal_coverage(targets, intervals["lower"], intervals["upper"])
            result["conformal"] = {
                "tau": conformal_state["tau"],
                "alpha": conformal_state["alpha"],
                **coverage,
            }
        else:
            intervals = None

        result["_arrays"] = {
            "preds": preds,
            "targets": targets,
            "pixel_error": pixel_error,
            "uncertainty_scores": unc_scores,
        }
        if intervals is not None:
            result["_arrays"]["conformal_lower"] = intervals["lower"]
            result["_arrays"]["conformal_upper"] = intervals["upper"]

        return result
