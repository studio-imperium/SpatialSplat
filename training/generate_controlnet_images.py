from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from training.add_condition_alpha import border_connected_alpha


DEFAULT_SCENES = (
    "01_twin_pines",
    "07_sandstone_arch",
    "13_cliff_steps",
    "18_ice_pillars",
    "26_garden_fountain",
    "30_garden_bench",
    "31_tank_farm",
    "37_robot_rover",
    "41_wooden_chair",
    "50_asymmetric_sculpture",
)

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEPTH_CONTROLNET = "diffusers/controlnet-depth-sdxl-1.0-small"
CANNY_CONTROLNET = "diffusers/controlnet-canny-sdxl-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"


@dataclass(frozen=True)
class GenerationSettings:
    width: int
    height: int
    steps: int
    guidance_scale: float
    strength: float
    depth_scale: float
    boundary_scale: float
    seed: int


def build_prompt(spec: dict) -> str:
    floor_rule = (
        "Flat complete ground slab at the shown height."
        if spec["has_floor"]
        else "no floor or platform."
    )
    return (
        f"Isolated on pure white, no scenery. Realistic miniature: {spec['description']} "
        "Exact control geometry: isometric orientation, count, positions, scale, bounding boxes, surface heights, "
        "occlusion. "
        f"{floor_rule} "
        "Natural textures. No added, moved, rotated, or flipped forms."
    )


def prepare_controls(scene_dir: Path, size: tuple[int, int]) -> list[Image.Image]:
    depth = Image.open(scene_dir / "primitive_depth_preview.png").convert("RGB")
    boundary = Image.open(scene_dir / "primitive_boundary.png").convert("RGB")
    resampling = Image.Resampling.LANCZOS
    return [depth.resize(size, resampling), boundary.resize(size, resampling)]


def prepare_init_image(scene_dir: Path, size: tuple[int, int]) -> Image.Image:
    control = Image.open(scene_dir / "primitive_control.png").convert("RGB")
    mask = Image.open(scene_dir / "primitive_mask.png").convert("L")
    white = Image.new("RGB", control.size, "white")
    white.paste(control, mask=mask)
    return white.resize(size, Image.Resampling.LANCZOS)


def alpha_coverage(image: Image.Image) -> dict[str, float]:
    alpha = np.asarray(image.getchannel("A"))
    return {
        "foreground_fraction": float((alpha > 127).mean()),
        "transparent_fraction": float((alpha == 0).mean()),
    }


def apply_condition_alpha(image: Image.Image, scene_dir: Path) -> Image.Image:
    rgba = border_connected_alpha(image)
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    protected = Image.open(scene_dir / "primitive_mask.png").convert("L").resize(
        rgba.size, Image.Resampling.LANCZOS
    )
    protected_alpha = np.asarray(protected, dtype=np.uint8)
    merged = Image.fromarray(np.maximum(alpha, protected_alpha), mode="L")
    rgba.putalpha(merged)
    return rgba


def validate_alpha(coverage: dict[str, float]) -> None:
    if coverage["foreground_fraction"] < 0.02:
        raise ValueError("alpha extraction retained too little foreground")
    if coverage["transparent_fraction"] < 0.10:
        raise ValueError("alpha extraction did not find a usable neutral background")


def select_scene_dirs(root: Path, names: list[str] | None, all_scenes: bool) -> list[Path]:
    selected = sorted(path for path in root.iterdir() if path.is_dir())
    if not all_scenes:
        wanted = set(names or DEFAULT_SCENES)
        selected = [path for path in selected if path.name in wanted]
        missing = wanted - {path.name for path in selected}
        if missing:
            raise FileNotFoundError(f"missing scene directories: {', '.join(sorted(missing))}")
    if not selected:
        raise ValueError(f"no scenes selected under {root}")
    return selected


def write_generation_summary(root: Path, scene_dirs: list[Path]) -> list[dict]:
    metadata = []
    for scene_dir in scene_dirs:
        path = scene_dir / "generation_metadata.json"
        if path.exists():
            metadata.append(json.loads(path.read_text(encoding="utf-8")))
    (root / "controlnet_generation_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_pipeline():
    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )

    dtype = torch.float16
    controlnets = [
        ControlNetModel.from_pretrained(
            DEPTH_CONTROLNET,
            torch_dtype=dtype,
            use_safetensors=True,
        ),
        ControlNetModel.from_pretrained(
            CANNY_CONTROLNET,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ),
    ]
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnets,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    return pipeline


def generate_scene(pipeline, scene_dir: Path, settings: GenerationSettings) -> dict:
    import torch

    spec = json.loads((scene_dir / "condition_spec.json").read_text(encoding="utf-8"))
    prompt = build_prompt(spec)
    controls = prepare_controls(scene_dir, (settings.width, settings.height))
    init_image = prepare_init_image(scene_dir, (settings.width, settings.height))
    generator = torch.Generator(device="cuda").manual_seed(settings.seed)
    started = time.perf_counter()
    image = pipeline(
        prompt=prompt,
        negative_prompt=(
            "rotated, flipped, mirrored, moved, resized, missing objects, extra objects, warped floor, "
            "scenery, horizon, patterned background, outside shadow, text"
        ),
        image=init_image,
        control_image=controls,
        num_inference_steps=settings.steps,
        guidance_scale=settings.guidance_scale,
        strength=settings.strength,
        controlnet_conditioning_scale=[settings.depth_scale, settings.boundary_scale],
        generator=generator,
        width=settings.width,
        height=settings.height,
    ).images[0]
    elapsed = time.perf_counter() - started

    rgb_path = scene_dir / "generated_image_rgb.png"
    alpha_path = scene_dir / "generated_image.png"
    image.convert("RGB").save(rgb_path)
    rgba = apply_condition_alpha(image, scene_dir)
    coverage = alpha_coverage(rgba)
    validate_alpha(coverage)
    rgba.save(alpha_path)

    metadata = {
        "scene_id": spec["scene_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "sdxl_multi_controlnet",
        "models": {
            "base": BASE_MODEL,
            "depth_controlnet": DEPTH_CONTROLNET,
            "boundary_controlnet": CANNY_CONTROLNET,
            "vae": VAE_MODEL,
        },
        "settings": asdict(settings),
        "prompt": prompt,
        "elapsed_seconds": elapsed,
        "alpha": coverage,
        "alpha_method": "border_connected_neutral_plus_control_subject_mask",
        "outputs": {
            "rgb": rgb_path.name,
            "rgba": alpha_path.name,
        },
    }
    (scene_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diverse scene images with SDXL depth and boundary ControlNets."
    )
    parser.add_argument("--root", type=Path, default=Path("poc_data/diverse_train"))
    parser.add_argument("--scene", action="append", help="Scene directory name; repeat to select several.")
    parser.add_argument("--all", action="store_true", help="Generate every scene below --root.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--alpha-only",
        action="store_true",
        help="Rebuild RGBA outputs from existing generated_image_rgb.png files.",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--depth-scale", type=float, default=0.95)
    parser.add_argument("--boundary-scale", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=24000)
    args = parser.parse_args()

    scene_dirs = select_scene_dirs(args.root, args.scene, args.all)
    pending = scene_dirs if args.alpha_only else [
        path for path in scene_dirs
        if args.overwrite or not (path / "generated_image.png").exists()
    ]
    if not pending:
        print("All selected scenes already have generated_image.png; nothing to do.")
        return

    if args.alpha_only:
        for scene_dir in pending:
            rgb_path = scene_dir / "generated_image_rgb.png"
            if not rgb_path.exists():
                raise FileNotFoundError(f"missing RGB generation: {rgb_path}")
            with Image.open(rgb_path) as image:
                rgba = apply_condition_alpha(image, scene_dir)
            coverage = alpha_coverage(rgba)
            validate_alpha(coverage)
            rgba.save(scene_dir / "generated_image.png")
            metadata_path = scene_dir / "generation_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["alpha"] = coverage
                metadata["alpha_method"] = "border_connected_neutral_plus_control_subject_mask"
                metadata_path.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
            print(f"{scene_dir.name}: foreground={coverage['foreground_fraction']:.3f}")
        write_generation_summary(
            args.root, sorted(path for path in args.root.iterdir() if path.is_dir())
        )
        return

    pipeline = load_pipeline()
    summary = []
    for index, scene_dir in enumerate(pending):
        settings = GenerationSettings(
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            depth_scale=args.depth_scale,
            boundary_scale=args.boundary_scale,
            seed=args.seed + index,
        )
        print(f"[{index + 1}/{len(pending)}] {scene_dir.name}", flush=True)
        metadata = generate_scene(pipeline, scene_dir, settings)
        summary.append(metadata)
        print(
            f"  {metadata['elapsed_seconds']:.1f}s; "
            f"foreground={metadata['alpha']['foreground_fraction']:.3f}",
            flush=True,
        )

    write_generation_summary(
        args.root, sorted(path for path in args.root.iterdir() if path.is_dir())
    )


if __name__ == "__main__":
    main()
