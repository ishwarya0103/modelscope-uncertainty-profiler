"""Deterministic (single-pass) uncertainty scores."""

from __future__ import annotations

import numpy as np

from modelscope.utils import to_probabilities


def entropy(logits: np.ndarray) -> np.ndarray:
    """Predictive entropy: -sum p_c log p_c.  Higher = more uncertain."""
    probs = to_probabilities(logits)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    return -np.sum(probs * log_probs, axis=-1)


def max_softmax_probability(logits: np.ndarray) -> np.ndarray:
    """1 - max(p_c).  Higher = more uncertain."""
    probs = to_probabilities(logits)
    return 1.0 - probs.max(axis=-1)


def margin(logits: np.ndarray) -> np.ndarray:
    """1 - (top1 - top2) probability margin.  Higher = more uncertain."""
    probs = to_probabilities(logits)
    sorted_probs = np.sort(probs, axis=-1)
    return 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
