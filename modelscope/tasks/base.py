"""Abstract base for task evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from modelscope.config import Config


class TaskEvaluator(ABC):
    """Interface every task evaluator must implement."""

    def __init__(self, model: torch.nn.Module, config: Config) -> None:
        self.model = model
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        data_splits: dict[str, Any],
    ) -> dict[str, Any]:
        """Run all metrics and UQ methods.  Return a results dict."""
        ...
