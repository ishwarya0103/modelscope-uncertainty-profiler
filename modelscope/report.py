"""Structured report: saves JSON metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class Report:
    """Holds the full profiling results and handles persistence."""

    def __init__(self, results: dict[str, Any], figures: dict[str, plt.Figure]) -> None:
        self.results = results
        self.figures = figures

    def save(self, output_dir: str | Path) -> Path:
        """Write ``metrics.json``, ``uncertainty.json``, and all plots to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        serializable = _strip_arrays(self.results)

        metrics_part: dict[str, Any] = {}
        uncertainty_part: dict[str, Any] = {}
        for split, data in serializable.get("splits", {}).items():
            metrics_part[split] = {
                "metrics": data.get("metrics"),
                "calibration": data.get("calibration"),
            }
            uncertainty_part[split] = data.get("uncertainty")

        (out / "metrics.json").write_text(json.dumps(metrics_part, indent=2, default=_json_default))
        (out / "uncertainty.json").write_text(
            json.dumps(uncertainty_part, indent=2, default=_json_default)
        )

        for fig_name, fig in self.figures.items():
            fig.savefig(out / f"{fig_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        return out

    def summary(self) -> str:
        """Return a short human-readable summary."""
        lines = [f"ModelScope Report  (task={self.results.get('task', '?')})"]
        lines.append("=" * 50)
        for split, data in self.results.get("splits", {}).items():
            m = data.get("metrics", {})
            c = data.get("calibration", {})
            lines.append(f"\n--- {split} ---")
            lines.append(f"  Accuracy : {m.get('accuracy', '?'):.4f}")
            lines.append(f"  F1       : {m.get('f1', '?'):.4f}")
            lines.append(f"  ECE      : {c.get('ece', '?'):.4f}")
            lines.append(f"  Brier    : {c.get('brier_score', '?'):.4f}")
            lines.append(f"  NLL      : {c.get('nll', '?'):.4f}")
            uq = data.get("uncertainty", {})
            if uq:
                lines.append("  Uncertainty (error AUROC):")
                for name, info in uq.items():
                    lines.append(f"    {name:30s} : {info.get('error_auroc', '?'):.4f}")
        return "\n".join(lines)


def _strip_arrays(d: Any) -> Any:
    """Recursively remove numpy array entries (keys starting with ``_``)."""
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_arrays(v) for v in d]
    return d


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
