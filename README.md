# ModelScope

**Scope out a model's behavior in detail** — profiling, calibration, and uncertainty quantification for PyTorch models.

Given a trained model and data loaders, ModelScope produces a complete evaluation report: standard metrics, calibration diagnostics, uncertainty scores, and publication-ready plots.

## Quick start

```bash
pip install -e ".[examples]"
python examples/cifar10_example.py
```

## Usage

```python
from modelscope import Config, Profiler

config = Config(
    task="classification",
    uncertainty_methods=["entropy", "max_softmax", "margin", "mc_dropout"],
    plots=["reliability", "risk_coverage", "uncertainty_vs_error"],
    mc_samples=20,
    device="cuda",
    output_dir="outputs/",
)

data_splits = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader,
    "calibration": calib_loader,  # optional
}

profiler = Profiler(model, data_splits, config)
report = profiler.run()
report.save(config.output_dir)
```

## What you get

```
outputs/
├── metrics.json               # accuracy, precision, recall, F1, calibration
├── uncertainty.json           # per-method summary stats & error AUROC
├── reliability_test.png       # reliability (calibration) diagram
├── risk_coverage_test.png     # selective prediction curve
└── uncertainty_vs_error_test.png
```

## Supported features (v0)

| Category | Details |
|----------|---------|
| **Metrics** | Accuracy, Precision, Recall, F1 (macro/micro/binary) |
| **Calibration** | ECE, MCE, Brier score, NLL, reliability diagrams |
| **Uncertainty** | Predictive entropy, max softmax probability, margin, MC-Dropout (predictive entropy + mutual information) |
| **Plots** | Reliability diagram, risk-coverage curve, uncertainty vs error |

## Roadmap

| Version | Scope |
|---------|-------|
| **v0** | Image classification, metrics + ECE + entropy/MC-Dropout, plots |
| **v1** | Generic PyTorch interface, regression/image-to-image, residual conformal, structured JSON report |
| **v2** | Quantile-aware conformal for segmentation/MRI, ensembles, activation/perf profiling |
| **v3** | Text/LLM tasks, RAG eval hooks, packaging + docs |
| **v4** | Advanced profiling, OOD detection, corruption benchmarks, web UI |

## License

MIT
