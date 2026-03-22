"""Structured report: saves JSON metrics and plots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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

        metadata = {
            "modelscope_version": "0.2.0",
            "task": serializable.get("task"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": serializable.get("elapsed_seconds"),
        }

        metrics_part: dict[str, Any] = {}
        uncertainty_part: dict[str, Any] = {}
        for split, data in serializable.get("splits", {}).items():
            metrics_part[split] = {
                k: v for k, v in data.items()
                if k in ("metrics", "calibration", "conformal")
            }
            uncertainty_part[split] = data.get("uncertainty")

        full_report = {"metadata": metadata, "metrics": metrics_part, "uncertainty": uncertainty_part}
        (out / "metrics.json").write_text(json.dumps(metrics_part, indent=2, default=_json_default))
        (out / "uncertainty.json").write_text(
            json.dumps(uncertainty_part, indent=2, default=_json_default)
        )
        (out / "report.json").write_text(json.dumps(full_report, indent=2, default=_json_default))

        for fig_name, fig in self.figures.items():
            fig.savefig(out / f"{fig_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        return out

    def summary(self) -> str:
        """Return a short human-readable summary."""
        task = self.results.get("task", "?")
        lines = [f"ModelScope Report  (task={task})"]
        lines.append("=" * 55)

        for split, data in self.results.get("splits", {}).items():
            lines.append(f"\n--- {split} ---")
            m = data.get("metrics", {})
            c = data.get("calibration", {})
            conf = data.get("conformal", {})

            if task == "classification":
                lines.append(f"  Accuracy : {m.get('accuracy', '?'):.4f}")
                lines.append(f"  F1       : {m.get('f1', '?'):.4f}")
                if c:
                    lines.append(f"  ECE      : {c.get('ece', '?'):.4f}")
                    lines.append(f"  Brier    : {c.get('brier_score', '?'):.4f}")
                    lines.append(f"  NLL      : {c.get('nll', '?'):.4f}")
                if conf:
                    lines.append(
                        f"  Conformal: coverage={conf.get('empirical_coverage', '?'):.3f}  "
                        f"mean_set_size={conf.get('mean_set_size', '?'):.2f}"
                    )

            elif task == "regression":
                lines.append(f"  MAE      : {m.get('mae', '?'):.4f}")
                lines.append(f"  RMSE     : {m.get('rmse', '?'):.4f}")
                lines.append(f"  R²       : {m.get('r2', '?'):.4f}")
                if conf:
                    lines.append(
                        f"  Conformal: coverage={conf.get('empirical_coverage', '?'):.3f}  "
                        f"mean_width={conf.get('mean_interval_width', '?'):.4f}"
                    )

            elif task == "image_to_image":
                lines.append(f"  PSNR     : {m.get('psnr', '?'):.2f} dB")
                lines.append(f"  SSIM     : {m.get('ssim', '?'):.4f}")
                lines.append(f"  Pixel MAE: {m.get('pixel_mae', '?'):.4f}")
                if conf:
                    lines.append(
                        f"  Conformal: coverage={conf.get('empirical_coverage', '?'):.3f}  "
                        f"mean_width={conf.get('mean_interval_width', '?'):.4f}"
                    )

            uq = data.get("uncertainty", {})
            if uq:
                lines.append("  Uncertainty:")
                for name, info in uq.items():
                    parts = [f"mean={info.get('mean', '?'):.4f}"]
                    if "error_auroc" in info:
                        parts.append(f"AUROC={info['error_auroc']:.4f}")
                    if "correlation_with_error" in info:
                        parts.append(f"corr={info['correlation_with_error']:.4f}")
                    lines.append(f"    {name:30s} : {', '.join(parts)}")

        elapsed = self.results.get("elapsed_seconds")
        if elapsed is not None:
            lines.append(f"\nCompleted in {elapsed:.1f}s")

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
