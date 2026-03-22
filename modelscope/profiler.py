"""Main Profiler class — the public entry point."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import torch

from modelscope.config import Config
from modelscope.report import Report
from modelscope.tasks.classification import ClassificationEvaluator
from modelscope.visualization.reliability import plot_reliability_diagram
from modelscope.visualization.risk_coverage import plot_risk_coverage
from modelscope.visualization.uncertainty_plots import plot_uncertainty_vs_error


_EVALUATORS = {
    "classification": ClassificationEvaluator,
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

        evaluator = evaluator_cls(self.model, cfg)
        results = evaluator.evaluate(self.data_splits)

        figures = self._generate_plots(results)

        report = Report(results, figures)
        print(report.summary())
        return report

    def _generate_plots(self, results: dict[str, Any]) -> dict[str, plt.Figure]:
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

        return figures
