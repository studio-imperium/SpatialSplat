from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import safetensors.torch
import torch
import torch.nn.functional as F

from training.decode_train import (
    decode_fixed_anchors,
    freeze_decoder,
    sample_fixed_anchors,
)
from training.gsplat_depth_renderer import render_decoder_gaussian
from training.render_baseline_depths import _alignment_overlay, _depth_preview
from training.scene_schema import OrthographicCamera
from training.spatial_loss import SpatialLossConfig, spatial_metrics
from training.spatial_loss_torch import spatial_loss_torch


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
        },
        str(path),
    )


def _free_cuda(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def _load_or_generate_base_sample(
    scene_dir: Path,
    checkpoint_root: Path,
    device: torch.device,
    seed: int,
    steps: int,
    guidance_scale: float,
    shift: float,
) -> dict[str, torch.Tensor]:
    sample_path = scene_dir / "base_sample.safetensors"
    if sample_path.is_file():
        print(f"Loading cached base sample: {sample_path}", flush=True)
        return safetensors.torch.load_file(str(sample_path), device=str(device))

    from triposplat import (
        encode_image,
        load_dinov3,
        load_flow_model,
        load_vae_encoder,
        sample_latent,
    )

    prepared_path = scene_dir / "prepared_condition.png"
    if not prepared_path.is_file():
        raise FileNotFoundError(
            f"missing {prepared_path}; use the exact image returned by baseline preprocessing"
        )
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Loading image encoders", flush=True)
    dinov3 = load_dinov3(
        _checkpoint(checkpoint_root, "clip_vision/dino_v3_vit_h.safetensors"),
        device=device,
        dtype=torch.bfloat16,
    )
    vae_encoder = load_vae_encoder(
        _checkpoint(checkpoint_root, "vae/flux2-vae.safetensors"),
        device=device,
        dtype=torch.bfloat16,
    )
    image = Image.open(prepared_path).convert("RGB")
    condition = encode_image(image, dinov3, vae_encoder, generator=generator)
    _save_tensors(scene_dir / "conditioning.safetensors", condition)
    del dinov3, vae_encoder, image
    _free_cuda()

    print("Loading Phase 2 flow model and sampling the base latent", flush=True)
    flow_model = load_flow_model(
        _checkpoint(
            checkpoint_root, "diffusion_models/triposplat_fp16.safetensors"
        ),
        device=device,
        dtype=torch.float16,
    )
    sample = sample_latent(
        flow_model,
        condition,
        steps=steps,
        guidance_scale=guidance_scale,
        shift=shift,
        generator=generator,
        show_progress=True,
    )
    _save_tensors(sample_path, sample)
    del flow_model, condition
    _free_cuda()
    return sample


def _resize_target(
    depth: torch.Tensor, mask: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.shape == (size, size):
        return depth, mask
    mask_4d = mask[None, None]
    coarse_mask = F.interpolate(mask_4d, size=(size, size), mode="area")[0, 0]
    weighted_depth = F.interpolate(
        (depth * mask)[None, None], size=(size, size), mode="area"
    )[0, 0]
    coarse_depth = weighted_depth / coarse_mask.clamp_min(1e-8)
    return coarse_depth, coarse_mask


def _numpy_metrics(
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    predicted_depth: torch.Tensor,
    predicted_alpha: torch.Tensor,
    config: SpatialLossConfig,
) -> dict[str, float]:
    return spatial_metrics(
        target_depth.detach().cpu().numpy(),
        target_mask.detach().cpu().numpy(),
        predicted_depth.detach().cpu().numpy(),
        predicted_alpha.detach().cpu().numpy(),
        config,
    )


def _render_metrics(
    decoder,
    latent: torch.Tensor,
    anchors: dict[str, torch.Tensor],
    camera: OrthographicCamera,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    size: int,
    config: SpatialLossConfig,
):
    with torch.no_grad():
        gaussian = decode_fixed_anchors(decoder, latent, anchors)
        rendered = render_decoder_gaussian(gaussian, camera, size, size)
        resized_depth, resized_mask = _resize_target(target_depth, target_mask, size)
        metrics = _numpy_metrics(
            resized_depth, resized_mask, rendered.depth, rendered.alpha, config
        )
    return gaussian, rendered, metrics


def _write_render_artifacts(
    scene_dir: Path,
    prefix: str,
    depth: torch.Tensor,
    alpha: torch.Tensor,
) -> None:
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    alpha_np = alpha.detach().cpu().numpy().astype(np.float32)
    np.save(scene_dir / f"{prefix}_depth.npy", depth_np)
    np.save(scene_dir / f"{prefix}_alpha.npy", alpha_np)
    Image.fromarray(
        np.clip(np.round(alpha_np * 255), 0, 255).astype(np.uint8), mode="L"
    ).save(scene_dir / f"{prefix}_alpha.png")
    _depth_preview(depth_np, alpha_np).save(
        scene_dir / f"{prefix}_depth_preview.png"
    )
    _alignment_overlay(scene_dir, alpha_np).save(
        scene_dir / f"{prefix}_alignment_overlay.png"
    )


def _improvement(before: dict[str, float], after: dict[str, float]) -> float:
    denominator = max(abs(before["spatial_loss"]), 1e-8)
    return (before["spatial_loss"] - after["spatial_loss"]) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize one TripoSplat latent against isometric primitive depth."
    )
    parser.add_argument(
        "--scene-dir", type=Path, default=Path("poc_data/01_center_cube")
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--optimization-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--anchor-seed", type=int, default=1234)
    parser.add_argument("--resample-every", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--final-render-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latent optimization")
    device = torch.device(args.device)
    scene_dir = args.scene_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    scene_data = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    camera = OrthographicCamera(**scene_data["camera"])
    config = SpatialLossConfig()

    target_depth = torch.from_numpy(
        np.load(scene_dir / "primitive_depth.npy")
    ).to(device=device, dtype=torch.float32)
    target_mask = torch.from_numpy(
        np.asarray(Image.open(scene_dir / "primitive_mask.png").convert("L"))
        / 255.0
    ).to(device=device, dtype=torch.float32)

    started = time.time()
    sample = _load_or_generate_base_sample(
        scene_dir,
        checkpoint_root,
        device,
        args.seed,
        args.sampling_steps,
        args.guidance_scale,
        args.shift,
    )
    base_latent = sample["latent"].detach().to(device=device, dtype=torch.float32)
    del sample
    _free_cuda()

    from triposplat import load_decoder

    print("Loading and freezing the Phase 1 decoder", flush=True)
    decoder = load_decoder(
        _checkpoint(
            checkpoint_root, "vae/triposplat_vae_decoder_fp16.safetensors"
        ),
        device=device,
        dtype=torch.float16,
    )
    freeze_decoder(decoder)
    latent = torch.nn.Parameter(base_latent.clone())
    optimizer = torch.optim.Adam([latent], lr=args.learning_rate)
    target_depth_train, target_mask_train = _resize_target(
        target_depth, target_mask, args.render_size
    )

    anchors = sample_fixed_anchors(
        decoder, base_latent, args.num_gaussians, args.anchor_seed
    )
    _, baseline_render, baseline_train_metrics = _render_metrics(
        decoder,
        base_latent,
        anchors,
        camera,
        target_depth,
        target_mask,
        args.render_size,
        config,
    )
    print(
        f"Training-view baseline loss: {baseline_train_metrics['spatial_loss']:.6f}",
        flush=True,
    )
    del baseline_render

    best_loss = float("inf")
    best_latent = base_latent.detach().cpu().clone()
    best_anchor_seed = args.anchor_seed
    history: list[dict[str, float | int]] = []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.optimization_steps):
        if step > 0 and args.resample_every > 0 and step % args.resample_every == 0:
            current_anchor_seed = args.anchor_seed + step
            anchors = sample_fixed_anchors(
                decoder, latent, args.num_gaussians, current_anchor_seed
            )
        else:
            current_anchor_seed = args.anchor_seed + (
                step // args.resample_every * args.resample_every
                if args.resample_every > 0
                else 0
            )

        optimizer.zero_grad(set_to_none=True)
        gaussian = decode_fixed_anchors(
            decoder, latent, anchors, activation_checkpoint=True
        )
        rendered = render_decoder_gaussian(
            gaussian, camera, args.render_size, args.render_size
        )
        losses = spatial_loss_torch(
            target_depth_train,
            target_mask_train,
            rendered.depth,
            rendered.alpha,
            config,
        )
        prior_loss = F.mse_loss(latent, base_latent)
        total_loss = losses["spatial_loss"] + args.prior_weight * prior_loss
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [latent], args.gradient_clip
        )
        optimizer.step()

        record = {
            "step": step + 1,
            "anchor_seed": current_anchor_seed,
            "total_loss": float(total_loss.detach()),
            "spatial_loss": float(losses["spatial_loss"].detach()),
            "depth_loss": float(losses["depth_loss"].detach()),
            "mask_loss": float(losses["mask_loss"].detach()),
            "soft_iou": float(losses["soft_iou"].detach()),
            "centroid_loss": float(losses["centroid_loss"].detach()),
            "extent_loss": float(losses["extent_loss"].detach()),
            "prior_loss": float(prior_loss.detach()),
            "gradient_norm": float(gradient_norm),
        }
        history.append(record)
        if record["spatial_loss"] < best_loss and np.isfinite(record["spatial_loss"]):
            best_loss = record["spatial_loss"]
            best_latent = latent.detach().cpu().clone()
            best_anchor_seed = current_anchor_seed

        if step == 0 or (step + 1) % args.log_every == 0:
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            print(
                f"step {step + 1:03d}/{args.optimization_steps}: "
                f"spatial={record['spatial_loss']:.6f} "
                f"prior={record['prior_loss']:.6f} "
                f"grad={record['gradient_norm']:.4f} "
                f"peak_vram={peak_gib:.2f} GiB",
                flush=True,
            )
        del gaussian, rendered, losses, prior_loss, total_loss

    optimized_latent = best_latent.to(device=device)
    _save_tensors(
        scene_dir / "target_latent.safetensors", {"latent": optimized_latent}
    )
    history_path = scene_dir / "latent_optimization_history.jsonl"
    history_path.write_text(
        "".join(json.dumps(record) + "\n" for record in history),
        encoding="utf-8",
    )

    fixed_anchors = sample_fixed_anchors(
        decoder, base_latent, args.num_gaussians, best_anchor_seed
    )
    fresh_seed = args.anchor_seed + 10000
    fresh_base_anchors = sample_fixed_anchors(
        decoder, base_latent, args.num_gaussians, fresh_seed
    )
    fresh_target_anchors = sample_fixed_anchors(
        decoder, optimized_latent, args.num_gaussians, fresh_seed
    )

    base_fixed_g, base_fixed_r, base_fixed_m = _render_metrics(
        decoder,
        base_latent,
        fixed_anchors,
        camera,
        target_depth,
        target_mask,
        args.final_render_size,
        config,
    )
    target_fixed_g, target_fixed_r, target_fixed_m = _render_metrics(
        decoder,
        optimized_latent,
        fixed_anchors,
        camera,
        target_depth,
        target_mask,
        args.final_render_size,
        config,
    )
    base_fresh_g, base_fresh_r, base_fresh_m = _render_metrics(
        decoder,
        base_latent,
        fresh_base_anchors,
        camera,
        target_depth,
        target_mask,
        args.final_render_size,
        config,
    )
    target_fresh_g, target_fresh_r, target_fresh_m = _render_metrics(
        decoder,
        optimized_latent,
        fresh_target_anchors,
        camera,
        target_depth,
        target_mask,
        args.final_render_size,
        config,
    )

    base_fresh_g.save_ply(scene_dir / "local_base_splat.ply")
    target_fresh_g.save_ply(scene_dir / "optimized_splat.ply")
    target_fresh_g.save_splat(scene_dir / "optimized_splat.splat")
    _write_render_artifacts(
        scene_dir, "local_base", base_fresh_r.depth, base_fresh_r.alpha
    )
    _write_render_artifacts(
        scene_dir, "optimized", target_fresh_r.depth, target_fresh_r.alpha
    )

    summary = {
        "scene": scene_dir.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "settings": {
            **vars(args),
            "scene_dir": str(scene_dir),
            "checkpoint_root": str(checkpoint_root),
            "loss_config": asdict(config),
            "best_anchor_seed": best_anchor_seed,
            "fresh_anchor_seed": fresh_seed,
        },
        "training_resolution_baseline": baseline_train_metrics,
        "fixed_anchors": {
            "base": base_fixed_m,
            "optimized": target_fixed_m,
            "relative_improvement": _improvement(base_fixed_m, target_fixed_m),
        },
        "fresh_anchors": {
            "base": base_fresh_m,
            "optimized": target_fresh_m,
            "relative_improvement": _improvement(base_fresh_m, target_fresh_m),
        },
        "best_training_spatial_loss": best_loss,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    (scene_dir / "latent_optimization_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
