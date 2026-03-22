"""Main Profiler class — the public entry point."""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from modelscope.config import Config
from modelscope.report import Report
from modelscope.tasks.classification import ClassificationEvaluator
from modelscope.tasks.image_to_image import ImageToImageEvaluator
from modelscope.tasks.regression import RegressionEvaluator
from modelscope.visualization.conformal_plots import (
    plot_conformal_set_sizes,
    plot_coverage_vs_nominal,
    plot_interval_width_distribution,
)
from modelscope.visualization.regression_plots import (
    plot_predicted_vs_actual,
    plot_prediction_intervals,
    plot_residual_distribution,
)
from modelscope.visualization.reliability import plot_reliability_diagram
from modelscope.visualization.risk_coverage import plot_risk_coverage
from modelscope.visualization.uncertainty_plots import plot_uncertainty_vs_error


_EVALUATORS = {
    "classification": ClassificationEvaluator,
    "regression": RegressionEvaluator,
    "image_to_image": ImageToImageEvaluator,
}


class Profiler:
    """Orchestrates a full model profiling run.

    Usage
    -----
    >>> profiler = Profiler(model, data_splits, config)
    >>> report = profiler.run()
    >>> report.save("outputs/")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_splits: dict[str, Any],
        config: Config | None = None,
    ) -> None:
        self.model = model
        self.data_splits = data_splits
        self.config = config or Config()

    def run(self) -> Report:
        """Execute all evaluations and produce a :class:`Report`."""
        cfg = self.config

        evaluator_cls = _EVALUATORS.get(cfg.task)
        if evaluator_cls is None:
            raise ValueError(
                f"Task {cfg.task!r} is not yet supported.  "
                f"Available: {list(_EVALUATORS)}"
            )

        t0 = time.perf_counter()
        evaluator = evaluator_cls(self.model, cfg)
        results = evaluator.evaluate(self.data_splits)
        results["elapsed_seconds"] = time.perf_counter() - t0

        figures = self._generate_plots(results)

        report = Report(results, figures)
        print(report.summary())
        return report

    # ------------------------------------------------------------------
    # Plot routing
    # ------------------------------------------------------------------

    def _generate_plots(self, results: dict[str, Any]) -> dict[str, plt.Figure]:
        task = results.get("task", self.config.task)
        if task == "classification":
            return self._classification_plots(results)
        if task in ("regression", "image_to_image"):
            return self._regression_plots(results)
        return {}

    def _classification_plots(self, results: dict[str, Any]) -> dict[str, plt.Figure]:
        cfg = self.config
        figures: dict[str, plt.Figure] = {}

        for split_name, split_data in results.get("splits", {}).items():
            arrays = split_data.get("_arrays", {})
            probs = arrays.get("probs")
            labels = arrays.get("labels")
            preds = arrays.get("preds")
            unc = arrays.get("uncertainty_scores", {})

            if probs is None:
                continue

            if "reliability" in cfg.plots:
                fig = plot_reliability_diagram(
                    probs, labels, cfg.num_bins,
                    title=f"Reliability — {split_name}",
                )
                figures[f"reliability_{split_name}"] = fig

            if "risk_coverage" in cfg.plots and unc:
                fig = plot_risk_coverage(
                    labels, preds, unc,
                    title=f"Risk–Coverage — {split_name}",
                )
                figures[f"risk_coverage_{split_name}"] = fig

            if "uncertainty_vs_error" in cfg.plots and unc:
                fig = plot_uncertainty_vs_error(
                    labels, preds, unc,
                    title=f"Uncertainty vs Error — {split_name}",
                )
                figures[f"uncertainty_vs_error_{split_name}"] = fig

            conformal_sets = arrays.get("conformal_sets")
            if conformal_sets is not None:
                fig = plot_conformal_set_sizes(
                    conformal_sets,
                    title=f"Conformal Set Sizes — {split_name}",
                )
                figures[f"conformal_sets_{split_name}"] = fig

        return figures

    def _regression_plots(self, results: dict[str, Any]) -> dict[str, plt.Figure]:
        cfg = self.config
        figures: dict[str, plt.Figure] = {}

        for split_name, split_data in results.get("splits", {}).items():
            arrays = split_data.get("_arrays", {})
            preds = arrays.get("preds")
            targets = arrays.get("targets")

            if preds is None or targets is None:
                continue

            if "predicted_vs_actual" in cfg.plots:
                fig = plot_predicted_vs_actual(
                    targets, preds,
                    title=f"Predicted vs Actual — {split_name}",
                )
                figures[f"predicted_vs_actual_{split_name}"] = fig

            if "residual_distribution" in cfg.plots:
                fig = plot_residual_distribution(
                    targets, preds,
                    title=f"Residuals — {split_name}",
                )
                figures[f"residual_distribution_{split_name}"] = fig

            lower = arrays.get("conformal_lower")
            upper = arrays.get("conformal_upper")

            if lower is not None and upper is not None:
                if "prediction_intervals" in cfg.plots:
                    fig = plot_prediction_intervals(
                        targets, preds, lower, upper,
                        title=f"Conformal Intervals — {split_name}",
                    )
                    figures[f"prediction_intervals_{split_name}"] = fig

                if "interval_width" in cfg.plots:
                    fig = plot_interval_width_distribution(
                        lower, upper,
                        title=f"Interval Widths — {split_name}",
                    )
                    figures[f"interval_width_{split_name}"] = fig

                conformal_data = split_data.get("conformal", {})
                if "coverage_vs_nominal" in cfg.plots and "tau" in conformal_data:
                    calib_split = cfg.calibration_split
                    calib_arrays = results.get("splits", {}).get(calib_split, {}).get("_arrays", {})
                    calib_targets = calib_arrays.get("targets")
                    calib_preds = calib_arrays.get("preds")
                    if calib_targets is not None:
                        residuals = np.abs(calib_targets.ravel() - calib_preds.ravel())
                    else:
                        residuals = np.abs(targets.ravel() - preds.ravel())

                    fig = plot_coverage_vs_nominal(
                        targets, preds, residuals,
                        title=f"Coverage vs Nominal — {split_name}",
                    )
                    figures[f"coverage_vs_nominal_{split_name}"] = fig

        return figures
