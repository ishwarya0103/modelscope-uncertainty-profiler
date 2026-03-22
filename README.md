# ModelScope

**Scope out a model's behavior in detail** — profiling, calibration, and uncertainty quantification for PyTorch models.

Given a trained model and data loaders, ModelScope produces a complete evaluation report: standard metrics, calibration diagnostics, uncertainty scores, conformal prediction intervals/sets, and publication-ready plots.

## Quick start

```bash
pip install -e ".[examples]"

# Classification (pretrained ResNet-18 on CIFAR-10)
python examples/cifar10_example.py

# Regression with conformal intervals (synthetic data)
python examples/regression_example.py
```

## Usage

```python
from modelscope import Config, Profiler

config = Config(
    task="regression",                          # or "classification", "image_to_image"
    uncertainty_methods=["conformal_residual", "mc_dropout"],
    plots=["predicted_vs_actual", "residual_distribution", "prediction_intervals"],
    calibration_split="calibration",
    conformal_alpha=0.1,                        # 90% target coverage
    device="mps",
    output_dir="outputs/",
)

data_splits = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader,
    "calibration": calib_loader,                # used for conformal calibration
}

profiler = Profiler(model, data_splits, config)
report = profiler.run()
report.save(config.output_dir)
```

## What you get

```
outputs/
├── report.json                    # full structured report (metadata + metrics + UQ)
├── metrics.json                   # per-split metrics and calibration
├── uncertainty.json               # per-method UQ summary stats
├── reliability_test.png           # calibration diagram (classification)
├── risk_coverage_test.png         # selective prediction curve
├── predicted_vs_actual_test.png   # scatter plot (regression)
├── residual_distribution_test.png # residual histogram
├── prediction_intervals_test.png  # conformal intervals with coverage
├── interval_width_test.png        # interval width distribution
└── conformal_sets_test.png        # set size distribution (classification)
```

## Supported features

### v0 — Classification

| Category | Details |
|----------|---------|
| **Metrics** | Accuracy, Precision, Recall, F1 (macro/micro/binary) |
| **Calibration** | ECE, MCE, Brier score, NLL, reliability diagrams |
| **Uncertainty** | Predictive entropy, max softmax, margin, MC-Dropout |
| **Plots** | Reliability diagram, risk-coverage, uncertainty vs error |

### v1 — Regression & Image-to-Image + Conformal Prediction

| Category | Details |
|----------|---------|
| **Regression metrics** | MAE, RMSE, R², explained variance |
| **Image-to-image metrics** | PSNR, SSIM, per-pixel MAE/MSE |
| **Conformal (regression)** | Split conformal with residual nonconformity scores — prediction intervals with finite-sample coverage guarantees |
| **Conformal (classification)** | Adaptive prediction sets (APS) — label sets with coverage guarantees |
| **MC-Dropout regression** | Predictive mean + variance from stochastic forward passes |
| **Plots** | Predicted vs actual, residual distribution, conformal intervals, interval width, coverage vs nominal, conformal set sizes |
| **CLI** | `python -m modelscope --config config.yaml` (model loading; data plugins coming) |

## Roadmap

| Version | Scope |
|---------|-------|
| **v0** | Image classification, metrics + ECE + entropy/MC-Dropout, plots |
| **v1** | Regression/image-to-image, residual conformal, conformal classification sets, CLI |
| **v2** | Quantile-aware conformal for segmentation/MRI, ensembles, activation/perf profiling |
| **v3** | Text/LLM tasks, RAG eval hooks, packaging + docs |
| **v4** | Advanced profiling, OOD detection, corruption benchmarks, web UI |

## License

MIT
