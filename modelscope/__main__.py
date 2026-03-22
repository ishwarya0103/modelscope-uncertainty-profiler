"""CLI entry point: ``python -m modelscope --config config.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from modelscope.config import Config
from modelscope.profiler import Profiler


def _load_config(path: str) -> Config:
    """Load a Config from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return Config(**raw)


def _load_model(path: str, device: str) -> torch.nn.Module:
    """Load a model from a saved checkpoint (.pt/.pth)."""
    model = torch.load(path, map_location=device, weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Expected a torch.nn.Module, got {type(model)}.  "
            "Ensure the checkpoint contains a full model (not just state_dict)."
        )
    return model.to(device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="modelscope",
        description="ModelScope — model profiling, calibration, and UQ.",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to a YAML or JSON config file.",
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to a saved model checkpoint (.pt / .pth).  "
             "If not provided, the config must specify how to load the model.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Override output directory from config.",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.output:
        cfg.output_dir = args.output

    if args.model:
        model = _load_model(args.model, cfg.device)
    else:
        print(
            "No --model provided.  In CLI mode you must supply a model checkpoint.\n"
            "For programmatic usage, use the Python API instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "Note: CLI mode currently requires you to construct DataLoaders in "
        "a script.  Use the Python API (Profiler class) for full flexibility.\n"
        "The CLI will gain data-loading plugins in a future version.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
