from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import safetensors.torch
import torch

from training.lora import inject_lora, load_lora
from training.create_data_split import split_scene_names
from training.multiview import ensure_supervision_views
from training.render_baseline_depths import _alignment_overlay, _depth_preview
from training.spatial_loss import SpatialLossConfig
from training.structure_metrics import (
    aggregate_structure_metrics,
    structure_view_metrics,
    support_primitive_ids,
)


DEFAULT_SEEDS = (101, 202, 303)


def _checkpoint(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return str(path)


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


def _save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(
        {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in tensors.items()
        },
        str(path),
    )


def _aggregate_metrics(metrics: dict[str, dict]) -> dict[str, float]:
    values = list(metrics.values())
    if not values:
        raise ValueError("at least one view metric is required")
    spatial_loss = float(np.mean([item["spatial_loss"] for item in values]))
    return {
        "spatial_loss": spatial_loss,
        "spatial_score": float(np.exp(-spatial_loss)),
        "depth_loss": float(np.mean([item["depth_loss"] for item in values])),
        "median_normalized_depth_error": float(
            max(item["median_normalized_depth_error"] for item in values)
        ),
        "p95_normalized_depth_error": float(
            max(item["p95_normalized_depth_error"] for item in values)
        ),
        "mask_loss": float(np.mean([item["mask_loss"] for item in values])),
        "soft_iou": float(min(item["soft_iou"] for item in values)),
        "centroid_loss": float(
            np.mean([item["centroid_loss"] for item in values])
        ),
        "extent_loss": float(np.mean([item["extent_loss"] for item in values])),
    }


def summarize_pairs(pairs: list[dict]) -> dict:
    if not pairs:
        raise ValueError("at least one generation pair is required")
    base_losses = [pair["base"]["aggregate"]["spatial_loss"] for pair in pairs]
    lora_losses = [pair["lora"]["aggregate"]["spatial_loss"] for pair in pairs]
    improvements = [
        (base - lora) / max(abs(base), 1e-8)
        for base, lora in zip(base_losses, lora_losses)
    ]
    recovered = [
        pair["recovered_target_improvement"]
        for pair in pairs
        if pair["recovered_target_improvement"] is not None
    ]
    summary = {
        "num_pairs": len(pairs),
        "mean_base_spatial_loss": float(np.mean(base_losses)),
        "mean_lora_spatial_loss": float(np.mean(lora_losses)),
        "mean_relative_improvement": float(np.mean(improvements)),
        "median_relative_improvement": float(np.median(improvements)),
        "win_rate": float(np.mean([lora < base for base, lora in zip(base_losses, lora_losses)])),
        "mean_recovered_target_improvement": (
            float(np.mean(recovered)) if recovered else None
        ),
    }
    p95_pairs = [
        (
            pair["base"]["aggregate"].get("p95_normalized_depth_error"),
            pair["lora"]["aggregate"].get("p95_normalized_depth_error"),
        )
        for pair in pairs
    ]
    p95_pairs = [
        (float(base), float(lora))
        for base, lora in p95_pairs
        if base is not None and lora is not None
    ]
    if p95_pairs:
        summary.update(
            {
                "mean_base_p95_depth_error": float(
                    np.mean([base for base, _ in p95_pairs])
                ),
                "mean_lora_p95_depth_error": float(
                    np.mean([lora for _, lora in p95_pairs])
                ),
                "p95_win_rate": float(
                    np.mean([lora < base for base, lora in p95_pairs])
                ),
            }
        )
    return summary


def _write_render(
    output_dir: Path,
    prefix: str,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    boundary_path: Path,
) -> None:
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    alpha_np = alpha.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / f"{prefix}_depth.npy", depth_np)
    np.save(output_dir / f"{prefix}_alpha.npy", alpha_np)
    Image.fromarray(
        np.clip(np.round(alpha_np * 255), 0, 255).astype(np.uint8), mode="L"
    ).save(output_dir / f"{prefix}_alpha.png")
    _depth_preview(depth_np, alpha_np).save(output_dir / f"{prefix}_depth.png")
    _alignment_overlay(output_dir, alpha_np, boundary_path).save(
        output_dir / f"{prefix}_overlay.png"
    )


def _sample_outputs(
    model,
    items: list[dict],
    manifest_path: Path,
    output_dir: Path,
    seeds: list[int],
    variant: str,
    steps: int,
    guidance_scale: float,
    shift: float,
    device: torch.device,
    force: bool,
) -> None:
    from triposplat import sample_latent

    model.eval()
    for scene_index, item in enumerate(items, start=1):
        condition = safetensors.torch.load_file(
            str(_resolve(manifest_path, item["conditioning"])), device=str(device)
        )
        for seed in seeds:
            sample_path = output_dir / item["scene"] / f"seed_{seed:04d}" / f"{variant}_sample.safetensors"
            if sample_path.is_file() and not force:
                print(f"[{variant}] cached {item['scene']} seed {seed}", flush=True)
                continue
            generator = torch.Generator(device=device).manual_seed(seed)
            print(
                f"[{variant}] sampling {scene_index}/{len(items)} "
                f"{item['scene']} seed {seed}",
                flush=True,
            )
            sample = sample_latent(
                model,
                condition,
                steps=steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=generator,
                show_progress=False,
            )
            _save_tensors(sample_path, sample)
            del sample
        del condition
        torch.cuda.empty_cache()


@torch.no_grad()
def _decode_and_score(
    decoder,
    latent: torch.Tensor,
    scene_dir: Path,
    output_dir: Path,
    variant: str,
    num_gaussians: int,
    render_size: int,
    octree_seed: int,
    config: SpatialLossConfig,
) -> dict:
    from training.gsplat_depth_renderer import render_decoder_gaussian

    devices = [latent.device] if latent.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(octree_seed)
        if latent.is_cuda:
            torch.cuda.manual_seed_all(octree_seed)
        gaussian = decoder.decode(latent, num_gaussians=num_gaussians)

    gaussian.save_ply(output_dir / f"{variant}_splat.ply")
    gaussian.save_splat(output_dir / f"{variant}_splat.splat")
    view_specs = ensure_supervision_views(scene_dir)
    support_ids = support_primitive_ids(scene_dir)
    view_metrics: dict[str, dict] = {}
    for name, view in view_specs.items():
        rendered = render_decoder_gaussian(
            gaussian, view.camera, render_size, render_size
        )
        target_depth = np.load(view.depth_path)
        target_mask = np.asarray(Image.open(view.mask_path).convert("L")) / 255.0
        view_metrics[name] = structure_view_metrics(
            target_depth,
            target_mask,
            np.load(view.ids_path),
            rendered.depth.detach().cpu().numpy(),
            rendered.alpha.detach().cpu().numpy(),
            support_ids,
            spatial_config=config,
        )
        _write_render(
            output_dir,
            f"{variant}_{name}",
            rendered.depth,
            rendered.alpha,
            view.boundary_path,
        )
        del rendered
    return {"aggregate": aggregate_structure_metrics(view_metrics), "views": view_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B test normal base and LoRA generations with paired fresh noise."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("poc_data/lora_dataset_six_view.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Evaluate prepared held-out scenes instead of the training manifest.",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument(
        "--lora", type=Path, default=Path("poc_data/lora_run_six_view/flow_lora.safetensors")
    )
    parser.add_argument(
        "--lora-config",
        type=Path,
        default=Path("poc_data/lora_run_six_view/flow_lora_config.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("poc_data/fresh_generation_six_view")
    )
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--octree-seed", type=int, default=7000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for normal generation evaluation")
    started = time.time()
    device = torch.device(args.device)
    manifest_path = args.manifest.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.data_root:
        scene_dirs = sorted(
            path.parent for path in args.data_root.resolve().glob("*/generated_image.png")
        )
        if args.split_file and args.split:
            allowed = split_scene_names(args.split_file, args.split)
            scene_dirs = [scene for scene in scene_dirs if scene.name in allowed]
        items = []
        for scene_dir in scene_dirs:
            conditioning = scene_dir / "conditioning.safetensors"
            if not conditioning.is_file():
                raise FileNotFoundError(
                    f"missing held-out conditioning: {conditioning}"
                )
            items.append(
                {
                    "scene": scene_dir.name,
                    "accepted": True,
                    "image": str(scene_dir / "generated_image.png"),
                    "conditioning": str(conditioning),
                }
            )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = [item for item in manifest["items"] if item["accepted"]]
    if args.scene:
        requested = set(args.scene)
        items = [item for item in items if item["scene"] in requested]
        missing = requested - {item["scene"] for item in items}
        if missing:
            raise ValueError(
                f"requested scene(s) are not accepted: {', '.join(sorted(missing))}"
            )
    if not items:
        raise ValueError("manifest contains no selected accepted scenes")
    seeds = args.seed or list(DEFAULT_SEEDS)
    if len(set(seeds)) != len(seeds):
        raise ValueError("generation seeds must be unique")

    from triposplat import load_decoder, load_flow_model

    lora_config = json.loads(args.lora_config.read_text(encoding="utf-8"))
    flow_model = load_flow_model(
        _checkpoint(checkpoint_root, lora_config["base_checkpoint"]),
        device=device,
        dtype=torch.float16,
    )
    flow_model.requires_grad_(False)
    _sample_outputs(
        flow_model,
        items,
        manifest_path,
        output_dir,
        seeds,
        "base",
        args.steps,
        args.guidance_scale,
        args.shift,
        device,
        args.force,
    )
    inject_lora(
        flow_model,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
        target_suffixes=lora_config["target_suffixes"],
    )
    load_lora(flow_model, args.lora.resolve())
    _sample_outputs(
        flow_model,
        items,
        manifest_path,
        output_dir,
        seeds,
        "lora",
        args.steps,
        args.guidance_scale,
        args.shift,
        device,
        args.force,
    )
    del flow_model
    gc.collect()
    torch.cuda.empty_cache()

    decoder = load_decoder(
        _checkpoint(checkpoint_root, "vae/triposplat_vae_decoder_fp16.safetensors"),
        device=device,
        dtype=torch.float16,
    )
    decoder.requires_grad_(False)
    decoder.eval()
    config = SpatialLossConfig()
    pairs: list[dict] = []
    for scene_index, item in enumerate(items):
        scene_dir = _resolve(manifest_path, item["image"]).parent
        target_value = item.get("fresh_metrics", {}).get("target_spatial_loss")
        target_loss = float(target_value) if target_value is not None else None
        for seed_index, seed in enumerate(seeds):
            pair_dir = output_dir / item["scene"] / f"seed_{seed:04d}"
            metrics_path = pair_dir / "metrics.json"
            if metrics_path.is_file() and not args.force:
                pair = json.loads(metrics_path.read_text(encoding="utf-8"))
                pairs.append(pair)
                print(f"[score] cached {item['scene']} seed {seed}", flush=True)
                continue
            pair_dir.mkdir(parents=True, exist_ok=True)
            octree_seed = args.octree_seed + scene_index * 100 + seed_index
            metrics = {}
            for variant in ("base", "lora"):
                sample = safetensors.torch.load_file(
                    str(pair_dir / f"{variant}_sample.safetensors"),
                    device=str(device),
                )
                print(
                    f"[score] {item['scene']} seed {seed} {variant}", flush=True
                )
                metrics[variant] = _decode_and_score(
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
            base_loss = metrics["base"]["aggregate"]["spatial_loss"]
            lora_loss = metrics["lora"]["aggregate"]["spatial_loss"]
            available = base_loss - target_loss if target_loss is not None else None
            pair = {
                "scene": item["scene"],
                "seed": seed,
                "octree_seed": octree_seed,
                "target_spatial_loss": target_loss,
                **metrics,
                "relative_improvement": (base_loss - lora_loss)
                / max(abs(base_loss), 1e-8),
                "recovered_target_improvement": (
                    (base_loss - lora_loss) / available
                    if available is not None and available > 1e-8
                    else None
                ),
            }
            metrics_path.write_text(
                json.dumps(pair, indent=2) + "\n", encoding="utf-8"
            )
            pairs.append(pair)

    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "settings": {
            **vars(args),
            "manifest": None if args.data_root else str(manifest_path),
            "data_root": str(args.data_root.resolve()) if args.data_root else None,
            "checkpoint_root": str(checkpoint_root),
            "lora": str(args.lora.resolve()),
            "lora_config": str(args.lora_config.resolve()),
            "output_dir": str(output_dir),
            "seeds": seeds,
        },
        "aggregate": summarize_pairs(pairs),
        "pairs": pairs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
