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

from spatial_control import (
    CONTROL_FEATURE_NAMES,
    DEFAULT_CONTROL_LAYERS,
    attach_spatial_control,
    save_spatial_control,
    scene_control_tensor,
)
from training.train_flow_lora import _flow_batch


def _checkpoint(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return str(path)


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


def _accepted_items(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [item for item in manifest["items"] if item.get("accepted", False)]


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
    default_scene_path = str(Path(item["image"]).parent / "scene.json")
    scene_path = _resolve(
        manifest_path, item.get("scene_json", default_scene_path)
    )
    control = scene_control_tensor(scene_path, token_count=target.shape[1])
    return {
        "scene": item["scene"],
        "condition": {
            name: value.to(device=device, non_blocking=True)
            for name, value in condition.items()
        },
        "control": control.to(device=device, dtype=torch.float32, non_blocking=True),
        "target_latent": target.to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "target_camera": base_sample["camera"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
    }


@torch.no_grad()
def _probe_loss(
    model,
    examples: list[dict],
    device: torch.device,
    seed: int,
    camera_weight: float,
    *,
    control_scale: float = 1.0,
    shuffled: bool = False,
    disabled: bool = False,
    lora_enabled: bool = False,
) -> float:
    from training.lora import set_lora_enabled

    set_lora_enabled(model, lora_enabled)
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for index, example in enumerate(examples):
        noisy, timesteps, targets = _flow_batch(example, generator)
        if disabled:
            control = None
        elif shuffled and len(examples) > 1:
            control = examples[(index + 1) % len(examples)]["control"]
        else:
            control = example["control"]
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            prediction = model(
                noisy,
                timesteps,
                example["condition"],
                control=control,
                control_scale=control_scale,
            )
            latent_loss = F.mse_loss(
                prediction["latent"].float(), targets["latent"]
            )
            camera_loss = F.mse_loss(
                prediction["camera"].float(), targets["camera"]
            )
        losses.append(float(latent_loss + camera_weight * camera_loss))
    return sum(losses) / len(losses)


def _write_config(path: Path, adapter, args: argparse.Namespace) -> dict:
    config = {
        "schema_version": 1,
        "base_checkpoint": "diffusion_models/triposplat_fp16.safetensors",
        "control_type": "primitive_sdf_vecseq",
        "feature_names": list(CONTROL_FEATURE_NAMES),
        "hidden_channels": adapter.hidden_channels,
        "layer_indices": list(adapter.layer_indices),
        "trainable_parameters": sum(
            parameter.numel() for parameter in adapter.parameters()
        ),
        "sobol_seed": 123,
        "default_control_scale": args.control_scale,
        "lora_training_probability": args.lora_probability,
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a primitive-SDF control adapter for TripoSplat Phase 2."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("poc_data/diverse_train/lora_train_manifest.json"),
    )
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument(
        "--lora",
        type=Path,
        help="Optional frozen Phase 2 LoRA used for compositional training.",
    )
    parser.add_argument(
        "--lora-config",
        type=Path,
        help="Configuration for --lora.",
    )
    parser.add_argument(
        "--lora-probability",
        type=float,
        default=0.5,
        help="Fraction of training steps run with the frozen LoRA enabled.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("poc_data/spatial_control_run")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument(
        "--control-layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTROL_LAYERS),
    )
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--camera-weight", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Phase 2 control-adapter training")
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and gradient accumulation must be positive")
    if bool(args.lora) != bool(args.lora_config):
        raise ValueError("--lora and --lora-config must be provided together")
    if not 0.0 <= args.lora_probability <= 1.0:
        raise ValueError("--lora-probability must be between zero and one")
    if not args.lora and args.lora_probability != 0.0:
        args.lora_probability = 0.0

    manifest_path = args.manifest.resolve()
    items = _accepted_items(manifest_path)
    if not items:
        raise ValueError("manifest contains no accepted latent targets")
    validation_path = (
        args.validation_manifest.resolve() if args.validation_manifest else None
    )
    validation_items = _accepted_items(validation_path) if validation_path else []
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from triposplat import load_flow_model

    model = load_flow_model(
        _checkpoint(
            args.checkpoint_root.resolve(),
            "diffusion_models/triposplat_fp16.safetensors",
        ),
        device=device,
        dtype=torch.float16,
    )
    if args.lora:
        from training.lora import inject_lora, load_lora

        lora_config = json.loads(args.lora_config.read_text(encoding="utf-8"))
        inject_lora(
            model,
            rank=lora_config["rank"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_suffixes=lora_config["target_suffixes"],
        )
        load_lora(model, args.lora)
    model.requires_grad_(False)
    adapter = attach_spatial_control(
        model,
        hidden_channels=args.hidden_channels,
        layer_indices=args.control_layers,
        device=device,
        dtype=torch.float32,
    )
    model.enable_gradient_checkpointing(not args.no_gradient_checkpointing)
    model.train()

    parameters = list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    order_rng = random.Random(args.seed)
    lora_rng = random.Random(args.seed + 17)
    order = list(range(len(items)))
    order_rng.shuffle(order)
    order_index = 0
    history: list[dict] = []
    best_validation_loss = float("inf")
    best_validation_step = None
    best_adapter_state = None
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Loading {len(items)} accepted training pairs", flush=True)
    examples = [_load_example(manifest_path, item, device) for item in items]
    validation_examples = (
        [_load_example(validation_path, item, device) for item in validation_items]
        if validation_path
        else []
    )
    model.eval()
    initial_train_probe = _probe_loss(
        model, examples, device, args.seed + 1, args.camera_weight
    )
    initial_validation_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
        )
        if validation_examples
        else None
    )
    initial_validation_lora_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
            lora_enabled=True,
        )
        if validation_examples and args.lora
        else None
    )
    model.train()
    print(
        f"Initial probes: train={initial_train_probe:.6f} "
        f"validation={initial_validation_probe}",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        accumulated_loss = 0.0
        accumulated_latent = 0.0
        accumulated_camera = 0.0
        scenes: list[str] = []
        lora_modes: list[bool] = []
        for _ in range(args.gradient_accumulation):
            if order_index == len(order):
                order_rng.shuffle(order)
                order_index = 0
            example = examples[order[order_index]]
            order_index += 1
            noisy, timesteps, targets = _flow_batch(example, generator)
            use_lora = False
            if args.lora:
                from training.lora import set_lora_enabled

                use_lora = lora_rng.random() < args.lora_probability
                set_lora_enabled(model, use_lora)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(
                    noisy,
                    timesteps,
                    example["condition"],
                    control=example["control"],
                    control_scale=args.control_scale,
                )
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
            lora_modes.append(use_lora)

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        denominator = args.gradient_accumulation
        record = {
            "step": step,
            "scenes": scenes,
            "lora_enabled": lora_modes,
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
                f"latent={record['latent_loss']:.6f} "
                f"grad={record['gradient_norm']:.4f} peak_vram={peak:.2f} GiB",
                flush=True,
            )
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_spatial_control(
                model,
                output_dir / "checkpoints" / f"step_{step:06d}.safetensors",
            )
        if (
            validation_examples
            and args.probe_every > 0
            and step % args.probe_every == 0
        ):
            model.eval()
            validation_probe = _probe_loss(
                model,
                validation_examples,
                device,
                args.seed + 2,
                args.camera_weight,
                lora_enabled=False,
            )
            validation_lora_probe = (
                _probe_loss(
                    model,
                    validation_examples,
                    device,
                    args.seed + 2,
                    args.camera_weight,
                    lora_enabled=True,
                )
                if args.lora
                else None
            )
            selection_loss = (
                (validation_probe + validation_lora_probe) * 0.5
                if validation_lora_probe is not None
                else validation_probe
            )
            record["validation_probe_loss"] = validation_probe
            record["validation_lora_probe_loss"] = validation_lora_probe
            record["validation_selection_loss"] = selection_loss
            if selection_loss < best_validation_loss:
                best_validation_loss = selection_loss
                best_validation_step = step
                best_adapter_state = {
                    name: value.detach().cpu().clone()
                    for name, value in adapter.state_dict().items()
                }
            print(
                f"probe {step:04d}: validation={validation_probe:.6f} "
                f"lora={validation_lora_probe} selected={selection_loss:.6f}",
                flush=True,
            )
            model.train()

    if best_adapter_state is not None:
        adapter.load_state_dict(
            {
                name: value.to(
                    device=next(adapter.parameters()).device,
                    dtype=next(adapter.parameters()).dtype,
                )
                for name, value in best_adapter_state.items()
            },
            strict=True,
        )

    model.eval()
    final_train_probe = _probe_loss(
        model, examples, device, args.seed + 1, args.camera_weight
    )
    final_validation_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
        )
        if validation_examples
        else None
    )
    final_validation_lora_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
            lora_enabled=True,
        )
        if validation_examples and args.lora
        else None
    )
    shuffled_validation_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
            shuffled=True,
            lora_enabled=False,
        )
        if len(validation_examples) > 1
        else None
    )
    disabled_validation_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
            disabled=True,
            lora_enabled=False,
        )
        if validation_examples
        else None
    )
    shuffled_validation_lora_probe = (
        _probe_loss(
            model,
            validation_examples,
            device,
            args.seed + 2,
            args.camera_weight,
            shuffled=True,
            lora_enabled=True,
        )
        if len(validation_examples) > 1 and args.lora
        else None
    )

    save_spatial_control(model, output_dir / "spatial_control.safetensors")
    config = _write_config(output_dir / "spatial_control_config.json", adapter, args)
    (output_dir / "spatial_control_history.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in history),
        encoding="utf-8",
    )
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "num_training_items": len(items),
        "num_validation_items": len(validation_items),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_train_probe_loss": initial_train_probe,
        "final_train_probe_loss": final_train_probe,
        "initial_validation_probe_loss": initial_validation_probe,
        "final_validation_probe_loss": final_validation_probe,
        "initial_validation_lora_probe_loss": initial_validation_lora_probe,
        "final_validation_lora_probe_loss": final_validation_lora_probe,
        "shuffled_validation_probe_loss": shuffled_validation_probe,
        "shuffled_validation_lora_probe_loss": shuffled_validation_lora_probe,
        "disabled_validation_probe_loss": disabled_validation_probe,
        "best_step_loss": min(record["loss"] for record in history),
        "best_validation_step": best_validation_step,
        "best_validation_selection_loss": (
            best_validation_loss if best_validation_step is not None else None
        ),
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "adapter": config,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
