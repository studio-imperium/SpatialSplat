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

from training.create_data_split import split_scene_names
from training.gsplat_depth_renderer import render_decoder_gaussian
from training.multiview import ensure_supervision_views
from training.spatial_loss import SpatialLossConfig
from training.structure_metrics import (
    aggregate_structure_metrics,
    structure_view_metrics,
    support_primitive_ids,
)


def _checkpoint(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return str(path)


def _save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    safetensors.torch.save_file(
        {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
        },
        str(path),
    )


def _score_gaussian(gaussian, scene_dir: Path, render_size: int) -> tuple[dict, dict]:
    view_specs = ensure_supervision_views(scene_dir)
    support_ids = support_primitive_ids(scene_dir)
    config = SpatialLossConfig()
    metrics = {}
    with torch.no_grad():
        for name, view in view_specs.items():
            rendered = render_decoder_gaussian(
                gaussian, view.camera, render_size, render_size
            )
            metrics[name] = structure_view_metrics(
                np.load(view.depth_path),
                np.asarray(Image.open(view.mask_path).convert("L")) / 255.0,
                np.load(view.ids_path),
                rendered.depth.detach().cpu().numpy(),
                rendered.alpha.detach().cpu().numpy(),
                support_ids,
                spatial_config=config,
            )
            del rendered
    return aggregate_structure_metrics(metrics), metrics


def candidate_viability(aggregate: dict) -> dict[str, bool]:
    objects = aggregate.get("object")
    if objects is None:
        return {"visible_objects": False}
    return {
        "visible_objects": True,
        "bbox": objects["min_bbox_iou"] >= 0.45,
        "centroid": objects["max_centroid_error"] <= 0.18,
        "extent": objects["max_extent_error"] <= 0.22,
        "signal": objects["min_signal_ratio"] >= 0.4,
    }


def candidate_rank(aggregate: dict, checks: dict[str, bool]) -> tuple[int, float]:
    return (-sum(checks.values()), float(aggregate["structure_loss"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate four Phase 2 candidates and keep the best structurally oriented one."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data/diverse_train"))
    parser.add_argument("--split-file", type=Path, default=Path("poc_data/diverse_train/split.json"))
    parser.add_argument("--split", action="append", default=["train", "validation"])
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument(
        "--lora",
        type=Path,
        help="Optional Phase 2 LoRA used to generate the candidate latents.",
    )
    parser.add_argument(
        "--lora-config",
        type=Path,
        help="Configuration for --lora.",
    )
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--erode-radius", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for TripoSplat candidate selection")
    seeds = args.seed or [41, 42, 43, 44]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("provide at least two unique candidate seeds")
    allowed = split_scene_names(args.split_file, args.split)
    if args.scene:
        allowed &= set(args.scene)
    scene_dirs = sorted(
        path.parent
        for path in args.data_root.glob("*/generated_image.png")
        if path.parent.name in allowed
    )
    if not scene_dirs:
        raise ValueError("no selected scenes found")

    from triposplat import TripoSplatPipeline

    checkpoint_root = args.checkpoint_root.resolve()
    pipeline = TripoSplatPipeline(
        ckpt_path=_checkpoint(checkpoint_root, "diffusion_models/triposplat_fp16.safetensors"),
        decoder_path=_checkpoint(checkpoint_root, "vae/triposplat_vae_decoder_fp16.safetensors"),
        dinov3_path=_checkpoint(checkpoint_root, "clip_vision/dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=_checkpoint(checkpoint_root, "vae/flux2-vae.safetensors"),
        rmbg_path=_checkpoint(checkpoint_root, "background_removal/birefnet.safetensors"),
        device=args.device,
    )
    if bool(args.lora) != bool(args.lora_config):
        raise ValueError("--lora and --lora-config must be provided together")
    if args.lora:
        from training.lora import inject_lora, load_lora

        lora_config = json.loads(args.lora_config.read_text(encoding="utf-8"))
        inject_lora(
            pipeline.flow_model,
            rank=lora_config["rank"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_suffixes=lora_config["target_suffixes"],
        )
        load_lora(pipeline.flow_model, args.lora)
        pipeline.flow_model.requires_grad_(False)
        pipeline.flow_model.eval()

    for scene_index, scene_dir in enumerate(scene_dirs, start=1):
        result_path = scene_dir / "candidate_selection.json"
        if result_path.is_file() and (scene_dir / "base_sample.safetensors").is_file() and not args.force:
            print(f"[{scene_index}/{len(scene_dirs)}] cached {scene_dir.name}", flush=True)
            continue
        started = time.time()
        condition_path = scene_dir / "generated_image.png"
        with Image.open(condition_path) as source:
            if source.mode != "RGBA" or source.getchannel("A").getextrema()[0] == 255:
                raise ValueError(f"{condition_path} does not have real transparency")
        prepared = pipeline.preprocess_image(condition_path, erode_radius=args.erode_radius)
        condition_generator = torch.Generator(device=args.device).manual_seed(1000)
        condition = pipeline.encode_image(prepared, generator=condition_generator)
        prepared.save(scene_dir / "prepared_condition.png")
        _save_tensors(scene_dir / "conditioning.safetensors", condition)

        rows = []
        best_rank = None
        selected_seed = None
        for seed in seeds:
            print(
                f"[{scene_index}/{len(scene_dirs)}] {scene_dir.name} candidate {seed}",
                flush=True,
            )
            generator = torch.Generator(device=args.device).manual_seed(seed)
            with torch.inference_mode():
                sample = pipeline.sample_latent(
                    condition,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    shift=args.shift,
                    generator=generator,
                    show_progress=False,
                )
                with torch.random.fork_rng(devices=[torch.device(args.device)]):
                    torch.manual_seed(7000)
                    torch.cuda.manual_seed_all(7000)
                    gaussian = pipeline.decode_latent(
                        sample["latent"], num_gaussians=args.num_gaussians
                    )
                aggregate, views = _score_gaussian(
                    gaussian, scene_dir, args.render_size
                )
            checks = candidate_viability(aggregate)
            rank = candidate_rank(aggregate, checks)
            rows.append(
                {
                    "seed": seed,
                    "viability_checks": checks,
                    "viable": all(checks.values()),
                    "aggregate": aggregate,
                    "views": views,
                }
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                selected_seed = seed
                _save_tensors(scene_dir / "base_sample.safetensors", sample)
                gaussian.save_ply(scene_dir / "base_splat.ply")
                gaussian.save_splat(scene_dir / "base_splat.splat")
            del sample, gaussian
            gc.collect()
            torch.cuda.empty_cache()

        selected = next(row for row in rows if row["seed"] == selected_seed)
        result = {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "scene": scene_dir.name,
            "selected_seed": selected_seed,
            "selected_viable": selected["viable"],
            "settings": {
                "seeds": seeds,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "shift": args.shift,
                "num_gaussians": args.num_gaussians,
                "render_size": args.render_size,
                "erode_radius": args.erode_radius,
                "lora": str(args.lora.resolve()) if args.lora else None,
            },
            "candidates": rows,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (scene_dir / "base_metrics.json").write_text(
            json.dumps(selected["aggregate"], indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"selected seed {selected_seed}; viable={selected['viable']}; "
            f"loss={selected['aggregate']['structure_loss']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
