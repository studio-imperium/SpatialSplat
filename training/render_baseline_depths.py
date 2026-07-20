from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.gaussian_depth_renderer import load_gaussian_ply, render_gaussian_depth
from training.scene_schema import OrthographicCamera


MODEL_WORLD_SCALE = 0.96
MODEL_WORLD_TRANSLATION = (0.0, -0.25, 0.0)


def _depth_preview(depth: np.ndarray, alpha: np.ndarray) -> Image.Image:
    visible = alpha > 0.05
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(visible):
        near, far = np.percentile(depth[visible], [1, 99])
        normalized = (depth - near) / max(float(far - near), 1e-8)
        preview[visible] = np.clip(
            np.round((1.0 - normalized[visible]) * 255), 0, 255
        ).astype(np.uint8)
    return Image.fromarray(preview, mode="L")


def _boundary(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    result[1:] |= mask[1:] != mask[:-1]
    result[:-1] |= mask[:-1] != mask[1:]
    result[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    result[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return result


def _alignment_overlay(
    scene_dir: Path,
    alpha: np.ndarray,
    boundary_path: Path | None = None,
) -> Image.Image:
    background = np.repeat(
        np.clip(np.round(alpha * 120), 0, 120).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    target_boundary_image = Image.open(
        boundary_path or scene_dir / "primitive_boundary.png"
    ).convert("L")
    if target_boundary_image.size != (alpha.shape[1], alpha.shape[0]):
        target_boundary_image = target_boundary_image.resize(
            (alpha.shape[1], alpha.shape[0]), Image.Resampling.NEAREST
        )
    target_boundary = np.asarray(target_boundary_image) > 127
    predicted_boundary = _boundary(alpha > 0.2)
    background[target_boundary] = (255, 70, 70)
    background[predicted_boundary] = (60, 220, 255)
    return Image.fromarray(background, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render CPU depth and alpha images from downloaded baseline PLYs."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scenes = sorted(path.parent for path in args.data_root.glob("*/base_splat.ply"))
    if args.scene:
        requested = set(args.scene)
        scenes = [scene for scene in scenes if scene.name in requested]
        missing = requested - {scene.name for scene in scenes}
        if missing:
            raise FileNotFoundError(f"missing baseline scene(s): {', '.join(sorted(missing))}")
    if not scenes:
        raise FileNotFoundError(f"no base_splat.ply files found under {args.data_root}")

    for index, scene_dir in enumerate(scenes, start=1):
        output_depth = scene_dir / "base_depth.npy"
        if output_depth.is_file() and not args.force:
            print(f"[{index}/{len(scenes)}] skipping {scene_dir.name}: render exists")
            continue
        target_depth = np.load(scene_dir / "primitive_depth.npy")
        height, width = target_depth.shape
        scene_data = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
        camera = OrthographicCamera(**scene_data["camera"])
        cloud = load_gaussian_ply(scene_dir / "base_splat.ply")
        print(f"[{index}/{len(scenes)}] rendering {scene_dir.name}")
        rendered = render_gaussian_depth(
            cloud,
            width=width,
            height=height,
            camera=camera,
            world_scale=MODEL_WORLD_SCALE,
            world_translation=MODEL_WORLD_TRANSLATION,
        )
        np.save(output_depth, rendered.depth)
        np.save(scene_dir / "base_alpha.npy", rendered.alpha)
        Image.fromarray(
            np.clip(np.round(rendered.alpha * 255), 0, 255).astype(np.uint8), mode="L"
        ).save(scene_dir / "base_alpha.png")
        _depth_preview(rendered.depth, rendered.alpha).save(
            scene_dir / "base_depth_preview.png"
        )
        _alignment_overlay(scene_dir, rendered.alpha).save(
            scene_dir / "base_alignment_overlay.png"
        )
        (scene_dir / "base_render_metadata.json").write_text(
            json.dumps(
                {
                    "renderer": "cpu_gaussian_depth",
                    "camera_type": "orthographic",
                    "camera": asdict(camera),
                    "model_world_scale": MODEL_WORLD_SCALE,
                    "model_world_translation": MODEL_WORLD_TRANSLATION,
                    "width": width,
                    "height": height,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[{index}/{len(scenes)}] completed {scene_dir.name}")


if __name__ == "__main__":
    main()
