"""Classification task evaluator — orchestrates metrics + UQ for classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from modelscope.metrics.calibration import (
    brier_score,
    compute_ece,
    compute_mce,
    negative_log_likelihood,
)
from modelscope.metrics.classification import classification_metrics
from modelscope.tasks.base import TaskEvaluator
from modelscope.uncertainty.conformal import (
    calibrate_conformal_classification,
    evaluate_conformal_sets,
    predict_conformal_sets,
)
from modelscope.uncertainty.deterministic import (
    entropy,
    margin,
    max_softmax_probability,
)
from modelscope.uncertainty.mc_dropout import mc_dropout_predict
from modelscope.utils import collect_predictions, to_probabilities


class ClassificationEvaluator(TaskEvaluator):
    """Evaluate a classification model: metrics, calibration, uncertainty, conformal sets."""

    _DETERMINISTIC_UQ = {
        "entropy": entropy,
        "max_softmax": max_softmax_probability,
        "margin": margin,
    }

    def evaluate(self, data_splits: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {"task": "classification", "splits": {}}

        conformal_state = self._calibrate_conformal(data_splits)

        for split_name in ("val", "test"):
            loader = data_splits.get(split_name)
            if loader is None:
                continue
            split_result = self._evaluate_split(split_name, loader, data_splits, conformal_state)
            results["splits"][split_name] = split_result

        return results

    def _calibrate_conformal(self, data_splits: dict[str, Any]) -> dict | None:
        if "conformal_classification" not in self.config.uncertainty_methods:
            return None

        calib_loader = data_splits.get(self.config.calibration_split)
        if calib_loader is None:
            calib_loader = data_splits.get("val")
        if calib_loader is None:
            return None

        logits, labels = collect_predictions(self.model, calib_loader, self.config.device)
        return calibrate_conformal_classification(
            logits, labels, alpha=self.config.conformal_alpha
        )

    def _evaluate_split(
        self, name: str, loader, data_splits, conformal_state
    ) -> dict[str, Any]:
        cfg = self.config
        device = cfg.device

        logits, labels = collect_predictions(self.model, loader, device)
        probs = to_probabilities(logits)
        preds = probs.argmax(axis=1)

        result: dict[str, Any] = {}

        result["metrics"] = classification_metrics(labels, preds)

        result["calibration"] = {
            "ece": compute_ece(probs, labels, cfg.num_bins),
            "mce": compute_mce(probs, labels, cfg.num_bins),
            "brier_score": brier_score(probs, labels),
            "nll": negative_log_likelihood(probs, labels),
        }

        unc_scores: dict[str, np.ndarray] = {}
        for method in cfg.uncertainty_methods:
            if method in self._DETERMINISTIC_UQ:
                unc_scores[method] = self._DETERMINISTIC_UQ[method](logits)

        mc_result = None
        if "mc_dropout" in cfg.uncertainty_methods:
            try:
                mc_result = mc_dropout_predict(
                    self.model, loader, n_samples=cfg.mc_samples, device=device
                )
                unc_scores["mc_predictive_entropy"] = mc_result["predictive_entropy"]
                unc_scores["mc_mutual_information"] = mc_result["mutual_information"]
            except RuntimeError as exc:
                result.setdefault("warnings", []).append(str(exc))

        result["uncertainty"] = {}
        correct = (preds == labels).astype(float)
        for uq_name, scores in unc_scores.items():
            error = 1.0 - correct
            try:
                auroc = float(roc_auc_score(error, scores))
            except ValueError:
                auroc = float("nan")

            result["uncertainty"][uq_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "error_auroc": auroc,
            }

        # Conformal classification sets
        conformal_sets = None
        if conformal_state is not None:
            conformal_sets = predict_conformal_sets(logits, conformal_state["threshold"])
            result["conformal"] = {
                "alpha": conformal_state["alpha"],
                "threshold": conformal_state["threshold"],
                **evaluate_conformal_sets(conformal_sets, labels),
            }

        result["_arrays"] = {
            "logits": logits,
            "probs": probs,
            "labels": labels,
            "preds": preds,
            "uncertainty_scores": unc_scores,
        }
        if mc_result is not None:
            result["_arrays"]["mc_probs"] = mc_result["mean_probs"]
        if conformal_sets is not None:
            result["_arrays"]["conformal_sets"] = conformal_sets

        return result
