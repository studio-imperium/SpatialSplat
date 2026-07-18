from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch


def _checkpoint(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return str(path)


def _save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    safetensors.torch.save_file(
        {name: value.detach().to(device="cpu").contiguous() for name, value in tensors.items()},
        str(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cached TripoSplat baselines for every POC condition image."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    args = parser.parse_args()

    from triposplat import TripoSplatPipeline

    pipeline = TripoSplatPipeline(
        ckpt_path=_checkpoint(
            args.checkpoint_root, "diffusion_models/triposplat_fp16.safetensors"
        ),
        decoder_path=_checkpoint(
            args.checkpoint_root, "vae/triposplat_vae_decoder_fp16.safetensors"
        ),
        dinov3_path=_checkpoint(args.checkpoint_root, "clip_vision/dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=_checkpoint(args.checkpoint_root, "vae/flux2-vae.safetensors"),
        rmbg_path=_checkpoint(
            args.checkpoint_root, "background_removal/birefnet.safetensors"
        ),
        device=args.device,
    )

    scene_dirs = sorted(path.parent for path in args.data_root.glob("*/generated_image.png"))
    if not scene_dirs:
        raise FileNotFoundError(f"no generated_image.png files found under {args.data_root}")

    for scene_dir in scene_dirs:
        condition_path = scene_dir / "generated_image.png"
        prepared = pipeline.preprocess_image(condition_path)
        generator = torch.Generator(device=pipeline.flow_model.device).manual_seed(args.seed)
        condition = pipeline.encode_image(prepared, generator=generator)
        sample = pipeline.sample_latent(
            condition,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            generator=generator,
            show_progress=True,
        )
        gaussian = pipeline.decode_latent(
            sample["latent"], num_gaussians=args.num_gaussians
        )

        prepared.save(scene_dir / "prepared_condition.png")
        _save_tensors(scene_dir / "conditioning.safetensors", condition)
        _save_tensors(
            scene_dir / "base_sample.safetensors",
            {name: value for name, value in sample.items() if isinstance(value, torch.Tensor)},
        )
        gaussian.save_ply(scene_dir / "base_splat.ply")
        gaussian.save_splat(scene_dir / "base_splat.splat")
        metadata = {
            "condition": str(condition_path),
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "num_gaussians": args.num_gaussians,
        }
        (scene_dir / "base_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed {scene_dir.name}")


if __name__ == "__main__":
    main()
