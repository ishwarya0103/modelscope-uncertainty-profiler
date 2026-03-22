"""Regression task evaluator — scalar and multi-output regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from modelscope.metrics.regression import regression_metrics
from modelscope.tasks.base import TaskEvaluator
from modelscope.uncertainty.conformal import (
    calibrate_residual_conformal,
    evaluate_conformal_coverage,
    predict_residual_conformal,
)
from modelscope.uncertainty.deterministic import regression_variance
from modelscope.uncertainty.mc_dropout import mc_dropout_predict_regression
from modelscope.utils import collect_predictions


class RegressionEvaluator(TaskEvaluator):
    """Evaluate a regression model: metrics, residual uncertainty, conformal intervals."""

    def evaluate(self, data_splits: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {"task": "regression", "splits": {}}

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

        preds, targets = collect_predictions(self.model, calib_loader, self.config.device)
        preds = preds.squeeze()
        targets = targets.squeeze()

        return calibrate_residual_conformal(targets, preds, alpha=self.config.conformal_alpha)

    def _evaluate_split(
        self, name: str, loader, data_splits, conformal_state
    ) -> dict[str, Any]:
        cfg = self.config
        preds, targets = collect_predictions(self.model, loader, cfg.device)
        preds = preds.squeeze()
        targets = targets.squeeze()

        result: dict[str, Any] = {}
        result["metrics"] = regression_metrics(targets, preds)

        unc_scores: dict[str, np.ndarray] = {}
        residuals = np.abs(targets - preds)
        unc_scores["absolute_residual"] = residuals

        mc_result = None
        if "mc_dropout" in cfg.uncertainty_methods:
            try:
                mc_result = mc_dropout_predict_regression(
                    self.model, loader, n_samples=cfg.mc_samples, device=cfg.device
                )
                unc_scores["mc_variance"] = mc_result["variance"].squeeze()
                unc_scores["mc_std"] = mc_result["std"].squeeze()
            except RuntimeError as exc:
                result.setdefault("warnings", []).append(str(exc))

        result["uncertainty"] = {}
        for uq_name, scores in unc_scores.items():
            result["uncertainty"][uq_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "median": float(np.median(scores)),
                "correlation_with_error": float(
                    np.corrcoef(residuals.ravel(), scores.ravel())[0, 1]
                ) if scores.shape == residuals.shape else float("nan"),
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
            "uncertainty_scores": unc_scores,
        }
        if mc_result is not None:
            result["_arrays"]["mc_mean"] = mc_result["mean_preds"].squeeze()
        if intervals is not None:
            result["_arrays"]["conformal_lower"] = intervals["lower"]
            result["_arrays"]["conformal_upper"] = intervals["upper"]

        return result
