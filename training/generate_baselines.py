from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import safetensors.torch
import torch

from training.create_data_split import split_scene_names


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


def has_real_alpha(path: Path) -> bool:
    with Image.open(path) as image:
        if image.mode != "RGBA":
            return False
        extrema = image.getchannel("A").getextrema()
        return extrema is not None and extrema[0] < 255


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cached TripoSplat baselines for every POC condition image."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--erode-radius", type=int, default=1)
    parser.add_argument(
        "--require-alpha",
        action="store_true",
        help="Reject RGB inputs instead of silently invoking background removal.",
    )
    args = parser.parse_args()

    if args.erode_radius < 0:
        parser.error("--erode-radius must be nonnegative")

    scene_dirs = sorted(path.parent for path in args.data_root.glob("*/generated_image.png"))
    if args.split_file and args.split:
        allowed = split_scene_names(args.split_file, args.split)
        scene_dirs = [scene for scene in scene_dirs if scene.name in allowed]
    if not scene_dirs:
        raise FileNotFoundError(f"no generated_image.png files found under {args.data_root}")
    alpha_by_scene = {
        scene_dir: has_real_alpha(scene_dir / "generated_image.png")
        for scene_dir in scene_dirs
    }
    if args.require_alpha:
        missing_alpha = [
            scene_dir / "generated_image.png"
            for scene_dir, has_alpha in alpha_by_scene.items()
            if not has_alpha
        ]
        if missing_alpha:
            raise ValueError(
                "input has no transparent pixels; refusing to run automatic "
                f"background removal: {missing_alpha[0]}"
            )

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

    for scene_dir in scene_dirs:
        if (
            args.skip_existing
            and (scene_dir / "conditioning.safetensors").is_file()
            and (scene_dir / "base_sample.safetensors").is_file()
        ):
            print(f"cached {scene_dir.name}")
            continue
        condition_path = scene_dir / "generated_image.png"
        input_has_alpha = alpha_by_scene[scene_dir]
        prepared = pipeline.preprocess_image(
            condition_path, erode_radius=args.erode_radius
        )
        condition_generator = torch.Generator(
            device=pipeline.flow_model.device
        ).manual_seed(args.condition_seed)
        condition = pipeline.encode_image(prepared, generator=condition_generator)
        generator = torch.Generator(device=pipeline.flow_model.device).manual_seed(args.seed)
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
            "condition_seed": args.condition_seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "num_gaussians": args.num_gaussians,
            "input_has_real_alpha": input_has_alpha,
            "background_removal_used": not input_has_alpha,
            "erode_radius": args.erode_radius,
        }
        (scene_dir / "base_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed {scene_dir.name}")


if __name__ == "__main__":
    main()
