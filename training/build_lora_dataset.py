from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


REQUIRED_FILES = (
    "generated_image.png",
    "prepared_condition.png",
    "conditioning.safetensors",
    "base_sample.safetensors",
    "target_latent.safetensors",
    "optimized_splat.ply",
    "latent_optimization_summary.json",
)


def build_item(scene_dir: Path, manifest_dir: Path, min_improvement: float) -> dict:
    missing = [name for name in REQUIRED_FILES if not (scene_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{scene_dir.name} is missing: {', '.join(missing)}")

    summary = json.loads(
        (scene_dir / "latent_optimization_summary.json").read_text(encoding="utf-8")
    )
    base = summary["fresh_anchors"]["base"]
    target = summary["fresh_anchors"]["optimized"]
    relative_improvement = float(summary["fresh_anchors"]["relative_improvement"])
    accepted = (
        relative_improvement >= min_improvement
        and target["spatial_loss"] < base["spatial_loss"]
        and target["soft_iou"] >= base["soft_iou"]
    )

    def relative(name: str) -> str:
        return os.path.relpath(scene_dir / name, manifest_dir)

    return {
        "scene": scene_dir.name,
        "accepted": accepted,
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
            "relative_improvement": relative_improvement,
        },
    }


def build_manifest(
    data_root: Path,
    output: Path,
    min_improvement: float = 0.1,
) -> dict:
    output = output.resolve()
    manifest_dir = output.parent
    scenes = sorted(path.parent for path in data_root.glob("*/generated_image.png"))
    if not scenes:
        raise FileNotFoundError(f"no scenes found under {data_root}")
    items = [build_item(scene.resolve(), manifest_dir, min_improvement) for scene in scenes]
    accepted = sum(item["accepted"] for item in items)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_relative_improvement": min_improvement,
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
    parser.add_argument("--allow-rejected", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.data_root.resolve(), args.output, args.min_improvement)
    print(
        f"packaged {manifest['num_accepted']}/{manifest['num_items']} accepted targets"
    )
    if manifest["num_accepted"] != manifest["num_items"] and not args.allow_rejected:
        raise SystemExit("one or more latent targets failed the acceptance gate")


if __name__ == "__main__":
    main()
