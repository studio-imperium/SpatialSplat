from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import safetensors.torch
import torch

from spatial_control import load_spatial_control, scene_control_tensor
from training.evaluate_lora_generation import (
    _checkpoint,
    _decode_and_score,
    _resolve,
    _save_tensors,
)
from training.lora import inject_lora, load_lora, set_lora_enabled
from training.spatial_loss import SpatialLossConfig


VARIANTS = (
    ("base", False, False),
    ("lora", True, False),
    ("control", False, True),
    ("combined", True, True),
)


def _aggregate(rows: list[dict]) -> dict:
    summary = {}
    for variant, _, _ in VARIANTS:
        metrics = [row[variant]["aggregate"] for row in rows]
        summary[variant] = {
            "mean_structure_loss": float(
                np.mean([item["structure_loss"] for item in metrics])
            ),
            "mean_p95_depth_error": float(
                np.mean([item["p95_normalized_depth_error"] for item in metrics])
            ),
            "mean_soft_iou": float(np.mean([item["soft_iou"] for item in metrics])),
        }
    base_losses = [row["base"]["aggregate"]["structure_loss"] for row in rows]
    for variant in ("lora", "control", "combined"):
        losses = [row[variant]["aggregate"]["structure_loss"] for row in rows]
        summary[variant]["win_rate_vs_base"] = float(
            np.mean([value < base for value, base in zip(losses, base_losses)])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-way decoded geometry test for Spatial Splat adapters."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("poc_data/control_generation_eval")
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--octree-seed", type=int, default=9000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for generation evaluation")
    started = time.time()
    device = torch.device(args.device)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item.get("accepted", False)]
    if not items:
        raise ValueError("manifest contains no accepted validation items")
    seeds = args.seed or [101]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from triposplat import load_decoder, load_flow_model, sample_latent

    lora_config = json.loads(args.lora_config.read_text(encoding="utf-8"))
    flow_model = load_flow_model(
        _checkpoint(args.checkpoint_root.resolve(), lora_config["base_checkpoint"]),
        device=device,
        dtype=torch.float16,
    )
    inject_lora(
        flow_model,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
        target_suffixes=lora_config["target_suffixes"],
    )
    load_lora(flow_model, args.lora.resolve())
    load_spatial_control(
        flow_model,
        args.control.resolve(),
        args.control_config.resolve(),
        device=device,
        dtype=torch.float16,
    )
    flow_model.requires_grad_(False)
    flow_model.eval()

    for item in items:
        condition = safetensors.torch.load_file(
            str(_resolve(manifest_path, item["conditioning"])), device=str(device)
        )
        scene_dir = _resolve(manifest_path, item["image"]).parent
        control = scene_control_tensor(
            scene_dir / "scene.json",
            token_count=flow_model.q_token_length,
            device=device,
        )
        for seed in seeds:
            pair_dir = output_dir / item["scene"] / f"seed_{seed:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            for variant, use_lora, use_control in VARIANTS:
                sample_path = pair_dir / f"{variant}_sample.safetensors"
                if sample_path.is_file() and not args.force:
                    continue
                print(f"[sample] {item['scene']} {seed} {variant}", flush=True)
                set_lora_enabled(flow_model, use_lora)
                generator = torch.Generator(device=device).manual_seed(seed)
                sample = sample_latent(
                    flow_model,
                    condition,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    shift=args.shift,
                    generator=generator,
                    show_progress=False,
                    control=control if use_control else None,
                    control_scale=args.control_scale,
                )
                _save_tensors(sample_path, sample)
                del sample
        del condition, control
        torch.cuda.empty_cache()

    del flow_model
    gc.collect()
    torch.cuda.empty_cache()
    decoder = load_decoder(
        _checkpoint(
            args.checkpoint_root.resolve(),
            "vae/triposplat_vae_decoder_fp16.safetensors",
        ),
        device=device,
        dtype=torch.float16,
    )
    decoder.requires_grad_(False)
    decoder.eval()
    config = SpatialLossConfig()
    rows = []
    for scene_index, item in enumerate(items):
        scene_dir = _resolve(manifest_path, item["image"]).parent
        for seed_index, seed in enumerate(seeds):
            pair_dir = output_dir / item["scene"] / f"seed_{seed:04d}"
            metrics_path = pair_dir / "metrics.json"
            if metrics_path.is_file() and not args.force:
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            row = {"scene": item["scene"], "seed": seed}
            octree_seed = args.octree_seed + scene_index * 100 + seed_index
            for variant, _, _ in VARIANTS:
                print(f"[score] {item['scene']} {seed} {variant}", flush=True)
                sample = safetensors.torch.load_file(
                    str(pair_dir / f"{variant}_sample.safetensors"),
                    device=str(device),
                )
                row[variant] = _decode_and_score(
                    decoder,
                    sample["latent"],
                    scene_dir,
                    pair_dir,
                    variant,
                    args.num_gaussians,
                    args.render_size,
                    octree_seed,
                    config,
                )
                del sample
                torch.cuda.empty_cache()
            metrics_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)

    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "aggregate": _aggregate(rows),
        "pairs": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
