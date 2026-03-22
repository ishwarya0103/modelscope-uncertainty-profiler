"""Uncertainty quantification methods."""

from modelscope.uncertainty.deterministic import (
    entropy,
    margin,
    max_softmax_probability,
)
from modelscope.uncertainty.mc_dropout import mc_dropout_predict

__all__ = [
    "entropy",
    "margin",
    "max_softmax_probability",
    "mc_dropout_predict",
]
