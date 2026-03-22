"""Uncertainty quantification methods."""

from modelscope.uncertainty.conformal import (
    calibrate_conformal_classification,
    calibrate_residual_conformal,
    evaluate_conformal_coverage,
    evaluate_conformal_sets,
    predict_conformal_sets,
    predict_residual_conformal,
)
from modelscope.uncertainty.deterministic import (
    entropy,
    margin,
    max_softmax_probability,
    regression_variance,
)
from modelscope.uncertainty.mc_dropout import (
    mc_dropout_predict,
    mc_dropout_predict_regression,
)

__all__ = [
    "entropy",
    "margin",
    "max_softmax_probability",
    "regression_variance",
    "mc_dropout_predict",
    "mc_dropout_predict_regression",
    "calibrate_residual_conformal",
    "predict_residual_conformal",
    "evaluate_conformal_coverage",
    "calibrate_conformal_classification",
    "predict_conformal_sets",
    "evaluate_conformal_sets",
]
