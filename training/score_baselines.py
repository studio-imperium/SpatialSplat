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
    parser = argparse.ArgumentParser(
        description="Score every rendered baseline splat against its primitive scene."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument("--coarse-size", type=int, default=64)
    parser.add_argument("--depth-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    config = SpatialLossConfig(
        coarse_size=args.coarse_size,
        depth_tolerance=args.depth_tolerance,
    )
    scene_dirs = sorted(path.parent for path in args.data_root.glob("*/base_depth.npy"))
    if not scene_dirs:
        raise FileNotFoundError(f"no base_depth.npy files found under {args.data_root}")

    rows: list[dict[str, float | str]] = []
    for scene_dir in scene_dirs:
        metrics = spatial_metrics(
            target_depth=np.load(scene_dir / "primitive_depth.npy"),
            target_mask=_load_mask(scene_dir / "primitive_mask.png"),
            predicted_depth=np.load(scene_dir / "base_depth.npy"),
            predicted_alpha=np.load(scene_dir / "base_alpha.npy"),
            config=config,
        )
        (scene_dir / "base_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        row = {
            "scene": scene_dir.name,
            "spatial_loss": float(metrics["spatial_loss"]),
            "spatial_score": float(metrics["spatial_score"]),
            "depth_loss": float(metrics["depth_loss"]),
            "soft_iou": float(metrics["soft_iou"]),
        }
        rows.append(row)
        print(
            f"{scene_dir.name:24s} score={row['spatial_score']:.4f} "
            f"depth={row['depth_loss']:.4f} iou={row['soft_iou']:.4f}"
        )

    loss_values = np.asarray([row["spatial_loss"] for row in rows], dtype=np.float64)
    score_values = np.asarray([row["spatial_score"] for row in rows], dtype=np.float64)
    summary = {
        "count": len(rows),
        "mean_spatial_loss": float(loss_values.mean()),
        "mean_spatial_score": float(score_values.mean()),
        "median_spatial_score": float(np.median(score_values)),
        "best_scene": rows[int(np.argmax(score_values))]["scene"],
        "worst_scene": rows[int(np.argmin(score_values))]["scene"],
        "config": {
            "coarse_size": args.coarse_size,
            "depth_tolerance": args.depth_tolerance,
        },
        "scenes": rows,
    }
    output_path = args.data_root / "baseline_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"mean score={summary['mean_spatial_score']:.4f}; "
        f"summary saved to {output_path}"
    )


if __name__ == "__main__":
    main()
