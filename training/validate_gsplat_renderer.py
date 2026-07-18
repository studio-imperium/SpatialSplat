from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from training.gaussian_depth_renderer import load_gaussian_ply
from training.gsplat_depth_renderer import (
    PLY_TO_WORLD_ROTATION,
    render_gaussian_tensors,
)
from training.render_baseline_depths import _alignment_overlay, _depth_preview
from training.scene_schema import OrthographicCamera
from training.spatial_loss import SpatialLossConfig, spatial_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare gsplat CUDA depth rendering with the saved CPU baseline."
    )
    parser.add_argument(
        "--scene-dir", type=Path, default=Path("poc_data/01_center_cube")
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    device = torch.device(args.device)
    scene_data = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    camera = OrthographicCamera(**scene_data["camera"])
    cloud = load_gaussian_ply(scene_dir / "base_splat.ply")
    target_depth = np.load(scene_dir / "primitive_depth.npy")
    target_mask = (
        np.asarray(Image.open(scene_dir / "primitive_mask.png").convert("L"))
        / 255.0
    ).astype(np.float32)

    rendered = render_gaussian_tensors(
        means=torch.from_numpy(cloud.means).to(device),
        scales=torch.from_numpy(cloud.scales).to(device),
        quaternions=torch.from_numpy(cloud.rotations).to(device),
        opacities=torch.from_numpy(cloud.opacities).to(device),
        camera=camera,
        width=camera.width,
        height=camera.height,
        rotation=PLY_TO_WORLD_ROTATION,
    )
    depth = rendered.depth.detach().cpu().numpy().astype(np.float32)
    alpha = rendered.alpha.detach().cpu().numpy().astype(np.float32)
    cpu_depth = np.load(scene_dir / "base_depth.npy")
    cpu_alpha = np.load(scene_dir / "base_alpha.npy")
    visible = (alpha > 0.05) & (cpu_alpha > 0.05)
    config = SpatialLossConfig()
    metrics = spatial_metrics(
        target_depth, target_mask, depth, alpha, config
    )
    cpu_metrics = spatial_metrics(
        target_depth, target_mask, cpu_depth, cpu_alpha, config
    )
    comparison = {
        "gsplat_metrics": metrics,
        "cpu_metrics": cpu_metrics,
        "spatial_loss_absolute_difference": abs(
            metrics["spatial_loss"] - cpu_metrics["spatial_loss"]
        ),
        "alpha_mean_absolute_error": float(np.mean(np.abs(alpha - cpu_alpha))),
        "overlap_depth_mean_absolute_error": (
            float(np.mean(np.abs(depth[visible] - cpu_depth[visible])))
            if np.any(visible)
            else None
        ),
        "overlap_pixels": int(np.count_nonzero(visible)),
    }
    np.save(scene_dir / "gsplat_baseline_depth.npy", depth)
    np.save(scene_dir / "gsplat_baseline_alpha.npy", alpha)
    _depth_preview(depth, alpha).save(scene_dir / "gsplat_baseline_depth_preview.png")
    _alignment_overlay(scene_dir, alpha).save(
        scene_dir / "gsplat_baseline_alignment_overlay.png"
    )
    (scene_dir / "gsplat_renderer_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
