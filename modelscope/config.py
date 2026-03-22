"""Configuration dataclass for ModelScope profiling runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TaskType = Literal["classification", "regression", "segmentation", "image_to_image"]

UncertaintyMethod = Literal[
    "entropy",
    "max_softmax",
    "margin",
    "mc_dropout",
    "ensemble",
    "conformal_residual",
    "conformal_classification",
    "cqr",
]

PlotType = Literal[
    "reliability",
    "risk_coverage",
    "uncertainty_vs_error",
    "predicted_vs_actual",
    "residual_distribution",
    "prediction_intervals",
    "coverage_vs_nominal",
    "interval_width",
    "uncertainty_heatmap",
]


@dataclass
class Config:
    """All settings for a single profiling run.

    Parameters
    ----------
    task : TaskType
        The ML task family.
    uncertainty_methods : list[UncertaintyMethod]
        Which UQ methods to compute.  Defaults to deterministic scores.
    plots : list[PlotType]
        Which plots to generate.
    calibration_split : str
        Key in *data_splits* to use for calibration.  If ``"auto"``, the
        profiler will carve out 20 % of the validation set.
    mc_samples : int
        Number of stochastic forward passes for MC-Dropout.
    num_bins : int
        Number of bins for ECE / reliability diagrams.
    conformal_alpha : float
        Miscoverage level for conformal prediction.  ``0.1`` means 90 %
        target coverage.
    device : str
        Torch device string.
    output_dir : str
        Where to write the report.
    """

    task: TaskType = "classification"
    uncertainty_methods: list[UncertaintyMethod] = field(
        default_factory=lambda: ["entropy", "max_softmax", "margin"]
    )
    plots: list[PlotType] = field(
        default_factory=lambda: ["reliability", "risk_coverage", "uncertainty_vs_error"]
    )
    calibration_split: str = "val"
    mc_samples: int = 30
    num_bins: int = 15
    conformal_alpha: float = 0.1
    device: str = "cpu"
    output_dir: str = "outputs"
