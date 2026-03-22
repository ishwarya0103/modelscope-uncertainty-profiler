#!/usr/bin/env python3
"""Example: profile a simple regression model with conformal prediction intervals.

Demonstrates:
  - Regression metrics (MAE, RMSE, R²)
  - Residual-based split conformal prediction
  - MC-Dropout uncertainty for regression
  - Plots: predicted-vs-actual, residual distribution, prediction intervals

Usage
-----
    pip install -e ".[examples]"
    python examples/regression_example.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from modelscope import Config, Profiler


class SimpleRegressor(nn.Module):
    """A 3-layer MLP with dropout for a 1-D regression task."""

    def __init__(self, in_features: int = 1, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_synthetic_data(n: int = 2000, noise: float = 0.3, seed: int = 42):
    """y = sin(x) + noise, x in [-3, 3]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 1)).astype(np.float32)
    y = (np.sin(x[:, 0]) + noise * rng.standard_normal(n)).astype(np.float32)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def train_model(model, loader, epochs=40, lr=1e-3, device="cpu"):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={total / len(loader.dataset):.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = make_synthetic_data(n=3000)
    train_ds, val_ds, calib_ds, test_ds = random_split(
        dataset, [1500, 500, 500, 500], generator=torch.Generator().manual_seed(0)
    )

    kw = dict(batch_size=128, num_workers=0)
    data_splits = {
        "train": DataLoader(train_ds, shuffle=True, **kw),
        "val": DataLoader(val_ds, **kw),
        "calibration": DataLoader(calib_ds, **kw),
        "test": DataLoader(test_ds, **kw),
    }

    model = SimpleRegressor().to(device)
    print("Training...")
    train_model(model, data_splits["train"], epochs=40, device=device)

    config = Config(
        task="regression",
        uncertainty_methods=["conformal_residual", "mc_dropout"],
        plots=[
            "predicted_vs_actual",
            "residual_distribution",
            "prediction_intervals",
            "interval_width",
        ],
        calibration_split="calibration",
        conformal_alpha=0.1,
        mc_samples=20,
        device=device,
        output_dir="outputs/regression",
    )

    profiler = Profiler(model, data_splits, config)
    report = profiler.run()
    out = report.save(config.output_dir)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
