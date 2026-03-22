#!/usr/bin/env python3
"""Minimal example: profile a ResNet-18 on CIFAR-10.

Usage
-----
    pip install -e ".[examples]"
    python examples/cifar10_example.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from modelscope import Config, Profiler


def get_model(num_classes: int = 10, dropout_p: float = 0.2) -> nn.Module:
    """ResNet-18 pretrained on ImageNet, adapted for CIFAR-10 with a dropout head."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def get_data(batch_size: int = 128):
    """Return train, val, test, calibration loaders for CIFAR-10."""
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_set, val_set = random_split(
        train_full, [40_000, 10_000], generator=torch.Generator().manual_seed(42)
    )
    val_set, calib_set = random_split(
        val_set, [7_000, 3_000], generator=torch.Generator().manual_seed(42)
    )

    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return {
        "train": DataLoader(train_set, shuffle=True, **kw),
        "val": DataLoader(val_set, **kw),
        "test": DataLoader(test_set, **kw),
        "calibration": DataLoader(calib_set, **kw),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = get_model().to(device)
    data_splits = get_data()

    config = Config(
        task="classification",
        uncertainty_methods=["entropy", "max_softmax", "margin", "mc_dropout"],
        plots=["reliability", "risk_coverage", "uncertainty_vs_error"],
        mc_samples=20,
        num_bins=15,
        device=device,
        output_dir="outputs/cifar10",
    )

    profiler = Profiler(model, data_splits, config)
    report = profiler.run()
    out = report.save(config.output_dir)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
