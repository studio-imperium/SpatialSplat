from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.spatial_loss import SpatialLossConfig, spatial_metrics


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a splat depth/alpha render against primitives.")
    parser.add_argument("--target-depth", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True)
    parser.add_argument("--predicted-depth", type=Path, required=True)
    parser.add_argument("--predicted-alpha", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coarse-size", type=int, default=64)
    parser.add_argument("--depth-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    metrics = spatial_metrics(
        target_depth=np.load(args.target_depth),
        target_mask=_load_mask(args.target_mask),
        predicted_depth=np.load(args.predicted_depth),
        predicted_alpha=np.load(args.predicted_alpha),
        config=SpatialLossConfig(
            coarse_size=args.coarse_size,
            depth_tolerance=args.depth_tolerance,
        ),
    )
    payload = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
