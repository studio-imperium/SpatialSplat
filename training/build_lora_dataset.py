from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from training.multiview import SUPERVISION_VIEWS
from training.create_data_split import split_scene_names
from training.structure_metrics import (
    StructureGateConfig,
    structure_gate,
)


REQUIRED_FILES = (
    "generated_image.png",
    "prepared_condition.png",
    "conditioning.safetensors",
    "base_sample.safetensors",
    "target_latent.safetensors",
    "optimized_splat.ply",
    "latent_optimization_summary.json",
)


def build_item(
    scene_dir: Path,
    manifest_dir: Path,
    min_improvement: float,
    max_p95_depth_error: float,
    min_soft_iou: float,
    max_median_depth_error: float,
) -> dict:
    candidate_path = scene_dir / "candidate_selection.json"
    candidate = (
        json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate_path.is_file()
        else None
    )
    missing = [name for name in REQUIRED_FILES if not (scene_dir / name).is_file()]
    if missing:
        if candidate is not None and not candidate["selected_viable"]:
            def relative(name: str) -> str:
                return os.path.relpath(scene_dir / name, manifest_dir)

            return {
                "scene": scene_dir.name,
                "accepted": False,
                "acceptance_checks": {
                    "required_files": False,
                    "candidate_orientation": False,
                },
                "missing_files": missing,
                "image": relative("generated_image.png"),
                "prepared_condition": relative("prepared_condition.png"),
                "conditioning": relative("conditioning.safetensors"),
                "base_sample": relative("base_sample.safetensors"),
                "target_latent": relative("target_latent.safetensors"),
                "optimized_splat": relative("optimized_splat.ply"),
                "fresh_metrics": None,
            }
        raise FileNotFoundError(f"{scene_dir.name} is missing: {', '.join(missing)}")

    summary = json.loads(
        (scene_dir / "latent_optimization_summary.json").read_text(encoding="utf-8")
    )
    base = summary["fresh_anchors"]["base"]
    target = summary["fresh_anchors"]["optimized"]
    relative_improvement = float(summary["fresh_anchors"]["relative_improvement"])
    per_view = summary["fresh_anchors"].get("views", {})
    optimized_views = {
        name: values["optimized"] for name, values in per_view.items()
    }
    metric_sources = list(optimized_views.values()) or [target]
    worst_p95 = max(item["p95_normalized_depth_error"] for item in metric_sources)
    worst_median = max(
        item["median_normalized_depth_error"] for item in metric_sources
    )
    worst_iou = min(item["soft_iou"] for item in metric_sources)
    checks = {
        "required_views": set(SUPERVISION_VIEWS).issubset(per_view),
        "relative_improvement": relative_improvement >= min_improvement,
        "spatial_loss_improved": target["spatial_loss"] < base["spatial_loss"],
        "soft_iou_not_worse": target["soft_iou"] >= 0.98 * base["soft_iou"],
        "p95_depth": (
            target["object"]["median_p95_depth_error"] <= max_p95_depth_error
            if target.get("object") is not None
            else worst_p95 <= max_p95_depth_error
        ),
        "median_depth": worst_median <= max_median_depth_error,
        "soft_iou": worst_iou >= min_soft_iou,
        "candidate_orientation": candidate is None or candidate["selected_viable"],
    }
    if target.get("object") is not None:
        structure_checks = structure_gate(
            target,
            base,
            StructureGateConfig(
                max_object_p95_depth_error=max_p95_depth_error,
                min_object_soft_iou=min_soft_iou,
            ),
        )
        checks.update({f"structure_{name}": value for name, value in structure_checks.items()})
    accepted = all(checks.values())

    def relative(name: str) -> str:
        return os.path.relpath(scene_dir / name, manifest_dir)

    return {
        "scene": scene_dir.name,
        "accepted": accepted,
        "acceptance_checks": checks,
        "image": relative("generated_image.png"),
        "prepared_condition": relative("prepared_condition.png"),
        "conditioning": relative("conditioning.safetensors"),
        "base_sample": relative("base_sample.safetensors"),
        "target_latent": relative("target_latent.safetensors"),
        "optimized_splat": relative("optimized_splat.ply"),
        "fresh_metrics": {
            "base_spatial_loss": base["spatial_loss"],
            "target_spatial_loss": target["spatial_loss"],
            "base_soft_iou": base["soft_iou"],
            "target_soft_iou": target["soft_iou"],
            "worst_view_p95_depth_error": worst_p95,
            "worst_view_median_depth_error": worst_median,
            "worst_view_soft_iou": worst_iou,
            "relative_improvement": relative_improvement,
            "views": optimized_views,
            "structure": target if target.get("object") is not None else None,
        },
    }


def build_manifest(
    data_root: Path,
    output: Path,
    min_improvement: float = 0.1,
    max_p95_depth_error: float = 0.2,
    min_soft_iou: float = 0.8,
    max_median_depth_error: float = 0.15,
    split_file: Path | None = None,
    split_names: tuple[str, ...] | list[str] = (),
) -> dict:
    output = output.resolve()
    manifest_dir = output.parent
    scenes = sorted(path.parent for path in data_root.glob("*/generated_image.png"))
    if split_file is not None and split_names:
        allowed = split_scene_names(split_file, split_names)
        scenes = [scene for scene in scenes if scene.name in allowed]
    if not scenes:
        raise FileNotFoundError(f"no scenes found under {data_root}")
    items = [
        build_item(
            scene.resolve(),
            manifest_dir,
            min_improvement,
            max_p95_depth_error,
            min_soft_iou,
            max_median_depth_error,
        )
        for scene in scenes
    ]
    accepted = sum(item["accepted"] for item in items)
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_relative_improvement": min_improvement,
        "max_p95_depth_error": max_p95_depth_error,
        "min_soft_iou": min_soft_iou,
        "max_median_depth_error": max_median_depth_error,
        "required_views": list(SUPERVISION_VIEWS),
        "splits": list(split_names),
        "num_items": len(items),
        "num_accepted": accepted,
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package accepted optimized latents for Phase 2 LoRA training."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument(
        "--output", type=Path, default=Path("poc_data/lora_dataset.json")
    )
    parser.add_argument("--min-improvement", type=float, default=0.1)
    parser.add_argument("--max-p95-depth-error", type=float, default=0.2)
    parser.add_argument("--min-soft-iou", type=float, default=0.65)
    parser.add_argument("--max-median-depth-error", type=float, default=0.15)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--allow-rejected", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        args.data_root.resolve(),
        args.output,
        args.min_improvement,
        args.max_p95_depth_error,
        args.min_soft_iou,
        args.max_median_depth_error,
        args.split_file,
        args.split,
    )
    print(
        f"packaged {manifest['num_accepted']}/{manifest['num_items']} accepted targets"
    )
    if manifest["num_accepted"] != manifest["num_items"] and not args.allow_rejected:
        raise SystemExit("one or more latent targets failed the acceptance gate")


if __name__ == "__main__":
    main()
