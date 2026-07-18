from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import safetensors.torch
import torch
import torch.nn.functional as F

from training.lora import (
    DEFAULT_TARGET_SUFFIXES,
    inject_lora,
    lora_parameters,
    save_lora,
)


def _checkpoint(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return str(path)


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


def _load_example(manifest_path: Path, item: dict, device: torch.device) -> dict:
    condition = safetensors.torch.load_file(
        str(_resolve(manifest_path, item["conditioning"])), device="cpu"
    )
    target = safetensors.torch.load_file(
        str(_resolve(manifest_path, item["target_latent"])), device="cpu"
    )["latent"]
    base_sample = safetensors.torch.load_file(
        str(_resolve(manifest_path, item["base_sample"])), device="cpu"
    )
    return {
        "scene": item["scene"],
        "condition": {
            name: value.to(device=device, non_blocking=True)
            for name, value in condition.items()
        },
        "target_latent": target.to(device=device, dtype=torch.float32, non_blocking=True),
        "target_camera": base_sample["camera"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
    }


def _flow_batch(example: dict, generator: torch.Generator) -> tuple[dict, torch.Tensor, dict]:
    target_latent = example["target_latent"]
    target_camera = example["target_camera"]
    t = torch.rand(
        target_latent.shape[0], device=target_latent.device, generator=generator
    ).clamp_(0.001, 0.999)
    latent_noise = torch.randn(
        target_latent.shape,
        device=target_latent.device,
        dtype=target_latent.dtype,
        generator=generator,
    )
    camera_noise = torch.randn(
        target_camera.shape,
        device=target_camera.device,
        dtype=target_camera.dtype,
        generator=generator,
    )
    latent_t = (1 - t[:, None, None]) * target_latent + t[:, None, None] * latent_noise
    camera_t = (1 - t[:, None, None]) * target_camera + t[:, None, None] * camera_noise
    targets = {
        "latent": latent_noise - target_latent,
        "camera": camera_noise - target_camera,
    }
    return {"latent": latent_t, "camera": camera_t}, t * 1000.0, targets


@torch.no_grad()
def _probe_loss(
    model,
    examples: list[dict],
    device: torch.device,
    seed: int,
    camera_weight: float,
) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for example in examples:
        noisy, timesteps, targets = _flow_batch(example, generator)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            prediction = model(noisy, timesteps, example["condition"])
            latent_loss = F.mse_loss(
                prediction["latent"].float(), targets["latent"]
            )
            camera_loss = F.mse_loss(
                prediction["camera"].float(), targets["camera"]
            )
        losses.append(float(latent_loss + camera_weight * camera_loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit Phase 2 LoRA adapters to optimized TripoSplat latents."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("poc_data/lora_dataset.json")
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--output-dir", type=Path, default=Path("poc_data/lora_run"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--camera-weight", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Phase 2 LoRA training")
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and gradient accumulation must be positive")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["accepted"]]
    if not items:
        raise ValueError("manifest contains no accepted latent targets")
    device = torch.device(args.device)
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from triposplat import load_flow_model

    model = load_flow_model(
        _checkpoint(
            checkpoint_root, "diffusion_models/triposplat_fp16.safetensors"
        ),
        device=device,
        dtype=torch.float16,
    )
    model.requires_grad_(False)
    injection = inject_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_suffixes=DEFAULT_TARGET_SUFFIXES,
    )
    model.enable_gradient_checkpointing(not args.no_gradient_checkpointing)
    model.train()

    parameters = list(lora_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    order_rng = random.Random(args.seed)
    order = list(range(len(items)))
    order_rng.shuffle(order)
    order_index = 0
    history: list[dict] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Loading {len(items)} accepted training pairs", flush=True)
    examples = [_load_example(manifest_path, item, device) for item in items]
    model.eval()
    initial_probe_loss = _probe_loss(
        model, examples, device, args.seed + 1, args.camera_weight
    )
    model.train()
    print(f"Fixed-probe initial loss: {initial_probe_loss:.6f}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        accumulated_loss = 0.0
        accumulated_latent = 0.0
        accumulated_camera = 0.0
        scenes: list[str] = []
        for _ in range(args.gradient_accumulation):
            if order_index == len(order):
                order_rng.shuffle(order)
                order_index = 0
            example = examples[order[order_index]]
            order_index += 1
            noisy, timesteps, targets = _flow_batch(example, generator)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(noisy, timesteps, example["condition"])
                latent_loss = F.mse_loss(
                    prediction["latent"].float(), targets["latent"]
                )
                camera_loss = F.mse_loss(
                    prediction["camera"].float(), targets["camera"]
                )
                loss = latent_loss + args.camera_weight * camera_loss
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())
            accumulated_latent += float(latent_loss.detach())
            accumulated_camera += float(camera_loss.detach())
            scenes.append(example["scene"])
            del noisy, timesteps, targets, prediction, loss, scaled_loss

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        denominator = args.gradient_accumulation
        record = {
            "step": step,
            "scenes": scenes,
            "loss": accumulated_loss / denominator,
            "latent_loss": accumulated_latent / denominator,
            "camera_loss": accumulated_camera / denominator,
            "gradient_norm": float(gradient_norm),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        if step == 1 or step % args.log_every == 0:
            peak = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"step {step:04d}/{args.steps}: loss={record['loss']:.6f} "
                f"latent={record['latent_loss']:.6f} grad={record['gradient_norm']:.4f} "
                f"peak_vram={peak:.2f} GiB",
                flush=True,
            )
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_lora(model, output_dir / "checkpoints" / f"step_{step:06d}.safetensors")

    model.eval()
    final_probe_loss = _probe_loss(
        model, examples, device, args.seed + 1, args.camera_weight
    )
    save_lora(model, output_dir / "flow_lora.safetensors")
    (output_dir / "flow_lora_history.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in history), encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "base_checkpoint": "diffusion_models/triposplat_fp16.safetensors",
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_suffixes": list(DEFAULT_TARGET_SUFFIXES),
        "module_names": list(injection.module_names),
        "trainable_parameters": injection.trainable_parameters,
    }
    (output_dir / "flow_lora_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    losses = [record["loss"] for record in history]
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "num_training_items": len(items),
        "steps": args.steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_reduction": (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-8),
        "initial_fixed_probe_loss": initial_probe_loss,
        "final_fixed_probe_loss": final_probe_loss,
        "fixed_probe_loss_reduction": (
            initial_probe_loss - final_probe_loss
        ) / max(abs(initial_probe_loss), 1e-8),
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "adapter": config,
        "settings": vars(args),
    }
    summary["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["settings"].items()}
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
