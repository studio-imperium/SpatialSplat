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
from training.multiview import SUPERVISION_VIEWS, ensure_supervision_views
from training.render_baseline_depths import _alignment_overlay, _depth_preview
from training.spatial_loss import SpatialLossConfig
from training.structure_loss_torch import structure_loss_torch
from training.structure_metrics import (
    aggregate_structure_metrics,
    region_masks,
    structure_view_metrics,
    support_primitive_ids,
)
from training.visual_anchor_loss import (
    gaussian_preservation_loss,
    visual_anchor_loss,
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


def _resize_rgb(rgb: torch.Tensor, size: int) -> torch.Tensor:
    if rgb.shape[:2] == (size, size):
        return rgb
    return F.interpolate(
        rgb.permute(2, 0, 1)[None],
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)


def _numpy_metrics(
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    predicted_depth: torch.Tensor,
    predicted_alpha: torch.Tensor,
    primitive_ids: np.ndarray,
    support_ids: set[int],
    config: SpatialLossConfig,
) -> dict:
    return structure_view_metrics(
        target_depth.detach().cpu().numpy(),
        target_mask.detach().cpu().numpy(),
        primitive_ids,
        predicted_depth.detach().cpu().numpy(),
        predicted_alpha.detach().cpu().numpy(),
        support_ids,
        spatial_config=config,
    )


def _render_metrics(
    decoder,
    latent: torch.Tensor,
    anchors: dict[str, torch.Tensor],
    views: dict[str, dict],
    size: int,
    config: SpatialLossConfig,
    render_rgb: bool = False,
):
    with torch.no_grad():
        gaussian = decode_fixed_anchors(decoder, latent, anchors)
        renders = {}
        metrics = {}
        for name, view in views.items():
            rendered = render_decoder_gaussian(
                gaussian, view["camera"], size, size, render_rgb=render_rgb
            )
            resized_depth, resized_mask = _resize_target(
                view["depth"], view["mask"], size
            )
            renders[name] = rendered
            metrics[name] = _numpy_metrics(
                resized_depth,
                resized_mask,
                rendered.depth,
                rendered.alpha,
                view["primitive_ids"],
                view["support_ids"],
                config,
            )
    return gaussian, renders, aggregate_structure_metrics(metrics), metrics


def _mean_visual_loss(
    renders: dict[str, object], views: dict[str, dict], size: int
) -> float | None:
    losses = []
    for name, rendered in renders.items():
        if rendered.rgb is None:
            return None
        _, target_mask = _resize_target(
            views[name]["depth"], views[name]["mask"], size
        )
        losses.append(
            visual_anchor_loss(
                _resize_rgb(views[name]["rgb"], size),
                target_mask,
                rendered.rgb,
                rendered.alpha,
            )
        )
    return float(torch.stack(losses).mean())


def _gaussian_quality(gaussian) -> dict[str, float]:
    opacity = gaussian.get_opacity.float().reshape(-1)
    scale = gaussian.get_scaling.float()
    mass = opacity * scale.prod(dim=-1)
    return {
        "opacity_sum": float(opacity.sum()),
        "mean_opacity": float(opacity.mean()),
        "effective_density_mass": float(mass.sum()),
        "mean_scale": float(scale.mean()),
    }


def _write_render_artifacts(
    scene_dir: Path,
    prefix: str,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    boundary_path: Path,
    rgb: torch.Tensor | None = None,
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
    _alignment_overlay(scene_dir, alpha_np, boundary_path).save(
        scene_dir / f"{prefix}_alignment_overlay.png"
    )
    if rgb is not None:
        rgb_np = np.clip(
            np.round(rgb.detach().cpu().numpy() * 255), 0, 255
        ).astype(np.uint8)
        Image.fromarray(rgb_np, mode="RGB").save(scene_dir / f"{prefix}_rgb.png")


def _improvement(before: dict[str, float], after: dict[str, float]) -> float:
    denominator = max(abs(before["spatial_loss"]), 1e-8)
    return (before["spatial_loss"] - after["spatial_loss"]) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize one TripoSplat latent against multi-view primitive depth."
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
    parser.add_argument("--visual-weight", type=float, default=0.0)
    parser.add_argument("--feature-preservation-weight", type=float, default=0.0)
    parser.add_argument("--opacity-preservation-weight", type=float, default=0.0)
    parser.add_argument("--scale-preservation-weight", type=float, default=0.0)
    parser.add_argument("--density-preservation-weight", type=float, default=0.0)
    parser.add_argument("--minimum-density-ratio", type=float, default=0.95)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--anchor-seed", type=int, default=1234)
    parser.add_argument("--resample-every", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--final-render-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--print-summary-json", action="store_true")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=SUPERVISION_VIEWS,
        default=list(SUPERVISION_VIEWS),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latent optimization")
    device = torch.device(args.device)
    scene_dir = args.scene_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    config = SpatialLossConfig()
    view_specs = ensure_supervision_views(scene_dir, args.views)
    support_ids = support_primitive_ids(scene_dir)
    views = {
        name: {
            "camera": spec.camera,
            "depth": torch.from_numpy(np.load(spec.depth_path)).to(
                device=device, dtype=torch.float32
            ),
            "mask": torch.from_numpy(
                np.asarray(Image.open(spec.mask_path).convert("L")).copy() / 255.0
            ).to(device=device, dtype=torch.float32),
            "rgb": torch.from_numpy(
                np.asarray(Image.open(spec.rgb_path).convert("RGB")).copy() / 255.0
            ).to(device=device, dtype=torch.float32),
            "boundary_path": spec.boundary_path,
            "primitive_ids": np.load(spec.ids_path),
            "support_ids": support_ids,
        }
        for name, spec in view_specs.items()
    }

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
    train_targets = {}
    for name, view in views.items():
        object_mask_np, support_mask_np = region_masks(
            view["primitive_ids"], support_ids
        )
        depth, mask = _resize_target(view["depth"], view["mask"], args.render_size)
        _, object_mask = _resize_target(
            view["depth"],
            torch.from_numpy(object_mask_np).to(device=device),
            args.render_size,
        )
        _, support_mask = _resize_target(
            view["depth"],
            torch.from_numpy(support_mask_np).to(device=device),
            args.render_size,
        )
        train_targets[name] = (
            depth,
            mask,
            object_mask,
            support_mask,
            _resize_rgb(view["rgb"], args.render_size),
        )

    anchors = sample_fixed_anchors(
        decoder, base_latent, args.num_gaussians, args.anchor_seed
    )
    reference_gaussian, baseline_renders, baseline_train_metrics, baseline_train_views = _render_metrics(
        decoder,
        base_latent,
        anchors,
        views,
        args.render_size,
        config,
        render_rgb=args.visual_weight > 0,
    )
    baseline_visual_loss = _mean_visual_loss(
        baseline_renders, views, args.render_size
    )
    print(
        f"Multi-view baseline loss: {baseline_train_metrics['spatial_loss']:.6f}",
        flush=True,
    )
    del baseline_renders

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
            with torch.no_grad():
                reference_gaussian = decode_fixed_anchors(
                    decoder, base_latent, anchors
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
        view_losses = {}
        visual_losses = {}
        for name, view in views.items():
            rendered = render_decoder_gaussian(
                gaussian,
                view["camera"],
                args.render_size,
                args.render_size,
                render_rgb=args.visual_weight > 0,
            )
            (
                target_depth_train,
                target_mask_train,
                object_mask_train,
                support_mask_train,
                target_rgb_train,
            ) = train_targets[name]
            view_losses[name] = structure_loss_torch(
                target_depth_train,
                target_mask_train,
                object_mask_train,
                support_mask_train,
                rendered.depth,
                rendered.alpha,
                config,
            )
            if args.visual_weight > 0:
                if rendered.rgb is None:
                    raise RuntimeError("RGB rendering was requested but not returned")
                visual_losses[name] = visual_anchor_loss(
                    target_rgb_train,
                    target_mask_train,
                    rendered.rgb,
                    rendered.alpha,
                )
        losses = {
            key: torch.stack([item[key] for item in view_losses.values()]).mean()
            for key in next(iter(view_losses.values()))
        }
        visual_loss = (
            torch.stack(list(visual_losses.values())).mean()
            if visual_losses
            else losses["spatial_loss"] * 0
        )
        preservation = gaussian_preservation_loss(
            gaussian,
            reference_gaussian,
            minimum_density_ratio=args.minimum_density_ratio,
        )
        objective_loss = (
            losses["spatial_loss"]
            + args.visual_weight * visual_loss
            + args.feature_preservation_weight
            * preservation["feature_preservation_loss"]
            + args.opacity_preservation_weight
            * preservation["opacity_preservation_loss"]
            + args.scale_preservation_weight
            * preservation["scale_preservation_loss"]
            + args.density_preservation_weight
            * preservation["density_preservation_loss"]
        )
        prior_loss = F.mse_loss(latent, base_latent)
        total_loss = objective_loss + args.prior_weight * prior_loss
        current_total_loss = float(total_loss.detach())
        if current_total_loss < best_loss and np.isfinite(current_total_loss):
            best_loss = current_total_loss
            best_latent = latent.detach().cpu().clone()
            best_anchor_seed = current_anchor_seed
        total_loss.backward(retain_graph=True)
        nonfinite_gradient_values = 0
        stable_gradient_fallback = False
        stable_total_loss = None
        if latent.grad is not None:
            finite_gradient = torch.isfinite(latent.grad)
            nonfinite_gradient_values = int((~finite_gradient).sum().item())
            if nonfinite_gradient_values:
                stable_gradient_fallback = True
                optimizer.zero_grad(set_to_none=True)
                stable_total_loss = (
                    losses["whole_spatial_loss"]
                    + args.visual_weight * visual_loss
                    + args.feature_preservation_weight
                    * preservation["feature_preservation_loss"]
                    + args.opacity_preservation_weight
                    * preservation["opacity_preservation_loss"]
                    + args.scale_preservation_weight
                    * preservation["scale_preservation_loss"]
                    + args.density_preservation_weight
                    * preservation["density_preservation_loss"]
                    + args.prior_weight * prior_loss
                )
                stable_total_loss.backward()
                if latent.grad is not None:
                    latent.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [latent], args.gradient_clip
        )
        optimizer.step()

        record = {
            "step": step + 1,
            "anchor_seed": current_anchor_seed,
            "total_loss": float(total_loss.detach()),
            "spatial_loss": float(losses["spatial_loss"].detach()),
            "visual_loss": float(visual_loss.detach()),
            "feature_preservation_loss": float(
                preservation["feature_preservation_loss"].detach()
            ),
            "opacity_preservation_loss": float(
                preservation["opacity_preservation_loss"].detach()
            ),
            "scale_preservation_loss": float(
                preservation["scale_preservation_loss"].detach()
            ),
            "density_preservation_loss": float(
                preservation["density_preservation_loss"].detach()
            ),
            "density_ratio": float(preservation["density_ratio"].detach()),
            "depth_loss": float(losses["depth_loss"].detach()),
            "mask_loss": float(losses["mask_loss"].detach()),
            "soft_iou": float(losses["soft_iou"].detach()),
            "centroid_loss": float(losses["centroid_loss"].detach()),
            "extent_loss": float(losses["extent_loss"].detach()),
            "prior_loss": float(prior_loss.detach()),
            "gradient_norm": float(gradient_norm),
            "nonfinite_gradient_values": nonfinite_gradient_values,
            "stable_gradient_fallback": stable_gradient_fallback,
            "views": {
                name: {
                    key: float(value.detach())
                    for key, value in item.items()
                }
                for name, item in view_losses.items()
            },
        }
        history.append(record)
        if step == 0 or (step + 1) % args.log_every == 0:
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            print(
                f"step {step + 1:03d}/{args.optimization_steps}: "
                f"spatial={record['spatial_loss']:.6f} "
                f"visual={record['visual_loss']:.6f} "
                f"density={record['density_ratio']:.3f} "
                f"prior={record['prior_loss']:.6f} "
                f"grad={record['gradient_norm']:.4f} "
                f"nonfinite_grad={record['nonfinite_gradient_values']} "
                f"fallback={record['stable_gradient_fallback']} "
                f"peak_vram={peak_gib:.2f} GiB",
                flush=True,
            )
        del (
            gaussian,
            rendered,
            losses,
            view_losses,
            visual_losses,
            visual_loss,
            preservation,
            objective_loss,
            prior_loss,
            total_loss,
            stable_total_loss,
        )

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

    base_fixed_g, base_fixed_r, base_fixed_m, base_fixed_views = _render_metrics(
        decoder,
        base_latent,
        fixed_anchors,
        views,
        args.final_render_size,
        config,
        render_rgb=args.visual_weight > 0,
    )
    target_fixed_g, target_fixed_r, target_fixed_m, target_fixed_views = _render_metrics(
        decoder,
        optimized_latent,
        fixed_anchors,
        views,
        args.final_render_size,
        config,
        render_rgb=args.visual_weight > 0,
    )
    base_fresh_g, base_fresh_r, base_fresh_m, base_fresh_views = _render_metrics(
        decoder,
        base_latent,
        fresh_base_anchors,
        views,
        args.final_render_size,
        config,
        render_rgb=args.visual_weight > 0,
    )
    target_fresh_g, target_fresh_r, target_fresh_m, target_fresh_views = _render_metrics(
        decoder,
        optimized_latent,
        fresh_target_anchors,
        views,
        args.final_render_size,
        config,
        render_rgb=args.visual_weight > 0,
    )

    base_fresh_g.save_ply(scene_dir / "local_base_splat.ply")
    target_fresh_g.save_ply(scene_dir / "optimized_splat.ply")
    target_fresh_g.save_splat(scene_dir / "optimized_splat.splat")
    for name, view in views.items():
        suffix = "" if name == "isometric" else f"_{name}"
        _write_render_artifacts(
            scene_dir,
            f"local_base{suffix}",
            base_fresh_r[name].depth,
            base_fresh_r[name].alpha,
            view["boundary_path"],
            base_fresh_r[name].rgb,
        )
        _write_render_artifacts(
            scene_dir,
            f"optimized{suffix}",
            target_fresh_r[name].depth,
            target_fresh_r[name].alpha,
            view["boundary_path"],
            target_fresh_r[name].rgb,
        )

    base_fixed_visual = _mean_visual_loss(
        base_fixed_r, views, args.final_render_size
    )
    target_fixed_visual = _mean_visual_loss(
        target_fixed_r, views, args.final_render_size
    )
    base_fresh_visual = _mean_visual_loss(
        base_fresh_r, views, args.final_render_size
    )
    target_fresh_visual = _mean_visual_loss(
        target_fresh_r, views, args.final_render_size
    )
    base_quality = _gaussian_quality(base_fresh_g)
    target_quality = _gaussian_quality(target_fresh_g)

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
        "training_resolution_baseline": {
            "aggregate": baseline_train_metrics,
            "views": baseline_train_views,
            "visual_anchor_loss": baseline_visual_loss,
        },
        "visual_anchor": {
            "fixed_base_loss": base_fixed_visual,
            "fixed_optimized_loss": target_fixed_visual,
            "fresh_base_loss": base_fresh_visual,
            "fresh_optimized_loss": target_fresh_visual,
        },
        "fresh_quality": {
            "base": base_quality,
            "optimized": target_quality,
            "density_mass_ratio": target_quality["effective_density_mass"]
            / max(base_quality["effective_density_mass"], 1e-8),
            "opacity_sum_ratio": target_quality["opacity_sum"]
            / max(base_quality["opacity_sum"], 1e-8),
        },
        "fixed_anchors": {
            "base": base_fixed_m,
            "optimized": target_fixed_m,
            "relative_improvement": _improvement(base_fixed_m, target_fixed_m),
            "views": {
                name: {
                    "base": base_fixed_views[name],
                    "optimized": target_fixed_views[name],
                    "relative_improvement": _improvement(
                        base_fixed_views[name], target_fixed_views[name]
                    ),
                }
                for name in views
            },
        },
        "fresh_anchors": {
            "base": base_fresh_m,
            "optimized": target_fresh_m,
            "relative_improvement": _improvement(base_fresh_m, target_fresh_m),
            "views": {
                name: {
                    "base": base_fresh_views[name],
                    "optimized": target_fresh_views[name],
                    "relative_improvement": _improvement(
                        base_fresh_views[name], target_fresh_views[name]
                    ),
                }
                for name in views
            },
        },
        "best_training_objective": best_loss,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    (scene_dir / "latent_optimization_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if args.print_summary_json:
        print(json.dumps(summary, indent=2, default=str), flush=True)
    else:
        fresh = summary["fresh_anchors"]
        print(
            f"completed {scene_dir.name}: fresh structure "
            f"{fresh['base']['structure_loss']:.4f} -> "
            f"{fresh['optimized']['structure_loss']:.4f}; "
            f"relative improvement {fresh['relative_improvement']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
