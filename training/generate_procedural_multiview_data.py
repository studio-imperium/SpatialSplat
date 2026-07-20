from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

from PIL import Image, ImageFilter

from training.multiview import ensure_supervision_views
from training.primitive_renderer import render_scene, save_render
from training.scene_schema import OrthographicCamera, Primitive, PrimitiveScene


PALETTE = (
    (190, 62, 52),
    (48, 130, 185),
    (52, 148, 76),
    (226, 174, 54),
    (130, 82, 170),
    (185, 108, 52),
    (92, 104, 116),
    (210, 216, 222),
)
FAMILIES = (
    "cluster",
    "arch",
    "stairs",
    "table",
    "tower",
    "grove",
    "robot",
    "bridge",
    "wall",
    "ring",
)


def _camera(resolution: int) -> OrthographicCamera:
    return OrthographicCamera(
        position=(1.7, 1.45, 1.7),
        target=(0.0, -0.08, 0.0),
        up=(0.0, 1.0, 0.0),
        ortho_scale=1.45,
        width=resolution,
        height=resolution,
    )


def _color(rng: random.Random) -> tuple[int, int, int]:
    return rng.choice(PALETTE)


def _box(name, center, size, color, yaw=0.0) -> Primitive:
    return Primitive(name, "box", center, size, color, yaw)


def _sphere(name, center, diameter, color) -> Primitive:
    return Primitive(name, "sphere", center, (diameter,) * 3, color)


def _cylinder(name, center, diameter, height, color) -> Primitive:
    return Primitive(name, "cylinder", center, (diameter, height, diameter), color)


def _ground(rng: random.Random) -> Primitive:
    return _box(
        "ground", (0.0, -0.455, 0.0), (0.98, 0.07, 0.98), _color(rng)
    )


def _on_ground(height: float) -> float:
    return -0.42 + height * 0.5


def _family_primitives(
    family: str, rng: random.Random
) -> tuple[list[Primitive], bool]:
    items: list[Primitive] = []
    has_ground = family not in {"robot"}
    if family == "cluster":
        positions = ((-0.27, 0.18), (-0.08, -0.16), (0.18, 0.14), (0.30, -0.19))
        for index, (x, z) in enumerate(positions):
            size = rng.uniform(0.14, 0.28)
            if index % 2:
                height = rng.uniform(0.16, 0.38)
                items.append(
                    _box(
                        f"cluster_{index}",
                        (x, _on_ground(height), z),
                        (size, height, size * rng.uniform(0.8, 1.2)),
                        _color(rng),
                        rng.uniform(-30, 30),
                    )
                )
            else:
                items.append(
                    _sphere(
                        f"cluster_{index}", (x, _on_ground(size), z), size, _color(rng)
                    )
                )
    elif family == "arch":
        width = rng.uniform(0.42, 0.58)
        post_height = rng.uniform(0.38, 0.56)
        post_width = rng.uniform(0.12, 0.18)
        color = _color(rng)
        items.extend(
            (
                _box("left_post", (-width / 2, _on_ground(post_height), 0.0), (post_width, post_height, 0.22), color),
                _box("right_post", (width / 2, _on_ground(post_height), 0.0), (post_width, post_height, 0.22), color),
                _box("lintel", (0.0, -0.42 + post_height + 0.06, 0.0), (width + post_width, 0.12, 0.24), _color(rng)),
            )
        )
    elif family == "stairs":
        for index in range(4):
            height = 0.08 + index * rng.uniform(0.07, 0.10)
            items.append(
                _box(
                    f"step_{index}",
                    (-0.23 + index * 0.16, _on_ground(height), 0.22 - index * 0.15),
                    (0.28, height, 0.28),
                    _color(rng),
                )
            )
    elif family == "table":
        top_height = rng.uniform(0.38, 0.50)
        top_size = (rng.uniform(0.52, 0.70), 0.10, rng.uniform(0.38, 0.54))
        items.append(_box("table_top", (0.0, -0.42 + top_height, 0.0), top_size, _color(rng), rng.uniform(-15, 15)))
        for index, (x, z) in enumerate(((-0.22, -0.14), (-0.22, 0.14), (0.22, -0.14), (0.22, 0.14))):
            items.append(_box(f"leg_{index}", (x, _on_ground(top_height), z), (0.08, top_height, 0.08), _color(rng)))
        items.append(_sphere("table_object", (0.08, -0.42 + top_height + 0.13, 0.0), 0.16, _color(rng)))
    elif family == "tower":
        heights = (0.16, 0.18, 0.20)
        y = -0.42
        for index, height in enumerate(heights):
            width = 0.42 - index * 0.08
            items.append(_box(f"level_{index}", (0.0, y + height / 2, 0.0), (width, height, width), _color(rng), rng.uniform(-20, 20)))
            y += height
        items.append(_sphere("tower_cap", (0.0, y + 0.08, 0.0), 0.16, _color(rng)))
    elif family == "grove":
        for index, (x, z) in enumerate(((-0.24, 0.08), (0.22, -0.10))):
            height = rng.uniform(0.30, 0.46)
            items.append(_cylinder(f"trunk_{index}", (x, _on_ground(height), z), 0.09, height, _color(rng)))
            items.append(_sphere(f"crown_{index}", (x, -0.42 + height + 0.12, z), rng.uniform(0.24, 0.34), _color(rng)))
    elif family == "robot":
        color = _color(rng)
        items.extend(
            (
                _box("left_leg", (-0.11, -0.31, 0.0), (0.12, 0.24, 0.14), color),
                _box("right_leg", (0.11, -0.31, 0.0), (0.12, 0.24, 0.14), color),
                _box("torso", (0.0, -0.08, 0.0), (0.34, 0.30, 0.24), _color(rng)),
                _sphere("head", (0.0, 0.15, 0.0), 0.20, _color(rng)),
                _box("left_arm", (-0.24, -0.08, 0.0), (0.12, 0.28, 0.12), color, -10),
                _box("right_arm", (0.24, -0.08, 0.0), (0.12, 0.28, 0.12), color, 10),
            )
        )
    elif family == "bridge":
        items.extend(
            (
                _box("deck", (0.0, -0.14, 0.0), (0.78, 0.12, 0.30), _color(rng), rng.uniform(-12, 12)),
                _box("left_support", (-0.27, -0.30, 0.0), (0.14, 0.32, 0.26), _color(rng)),
                _box("right_support", (0.27, -0.30, 0.0), (0.14, 0.32, 0.26), _color(rng)),
                _cylinder("marker", (0.0, 0.0, 0.0), 0.11, 0.22, _color(rng)),
            )
        )
    elif family == "wall":
        for row in range(2):
            for column in range(3 - row):
                x = (column - (2 - row) / 2) * 0.23
                items.append(_box(f"block_{row}_{column}", (x, -0.31 + row * 0.22, 0.0), (0.22, 0.21, 0.20), _color(rng), rng.uniform(-6, 6)))
    elif family == "ring":
        items.append(_cylinder("center", (0.0, -0.16, 0.0), 0.18, 0.52, _color(rng)))
        for index, (x, z) in enumerate(((0.30, 0.0), (-0.30, 0.0), (0.0, 0.30), (0.0, -0.30))):
            items.append(_cylinder(f"marker_{index}", (x, -0.36, z), 0.14, 0.12, _color(rng)))
    else:
        raise ValueError(f"unknown procedural family: {family}")
    return items, has_ground


def prepare_condition(image: Image.Image, size: int = 1024) -> Image.Image:
    """Match TripoSplat's RGBA preprocessing without loading background removal."""
    image = image.convert("RGBA")
    scale = size / min(image.size)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    image.putalpha(image.getchannel("A").filter(ImageFilter.MinFilter(3)))
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("procedural render contains no foreground")
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.6
    image = image.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    prepared = Image.new("RGB", (size, size), "black")
    prepared.paste(image, mask=image.getchannel("A"))
    return prepared


def generate_dataset(
    output: Path,
    num_scenes: int = 20,
    resolution: int = 512,
    seed: int = 2026,
) -> dict:
    if num_scenes < 3:
        raise ValueError("at least three scenes are required")
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(num_scenes):
        family = FAMILIES[index % len(FAMILIES)]
        scene_seed = seed + index * 7919
        rng = random.Random(scene_seed)
        items, has_ground = _family_primitives(family, rng)
        primitives = tuple([_ground(rng), *items] if has_ground else items)
        scene_id = f"{index + 1:02d}_{family}_{index // len(FAMILIES) + 1}"
        scene = PrimitiveScene(
            scene_id,
            f"Procedural {family} composition with exact six-view supervision.",
            _camera(resolution),
            primitives,
        )
        scene_dir = output / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene.write_json(scene_dir / "scene.json")
        save_render(render_scene(scene), scene_dir)
        ensure_supervision_views(scene_dir, force=True)
        shutil.copy2(scene_dir / "primitive_control.png", scene_dir / "generated_image.png")
        prepare_condition(Image.open(scene_dir / "generated_image.png")).save(
            scene_dir / "prepared_condition.png"
        )
        metadata = {
            "scene_id": scene_id,
            "family": family,
            "seed": scene_seed,
            "has_floor": has_ground,
            "input_type": "exact_procedural_primitive_render",
            "supervision": "six_view_rgb_depth_mask_ids",
        }
        (scene_dir / "condition_spec.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        names.append(scene_id)

    shuffled = names.copy()
    random.Random(seed + 1).shuffle(shuffled)
    train_count = max(1, round(num_scenes * 0.6))
    validation_count = max(1, round(num_scenes * 0.2))
    split = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "policy": "Deterministic 60/20/20 procedural POC split.",
        "splits": {
            "train": sorted(shuffled[:train_count]),
            "validation": sorted(shuffled[train_count : train_count + validation_count]),
            "test": sorted(shuffled[train_count + validation_count :]),
        },
    }
    (output / "split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    return split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact procedural six-view Spatial Splat training data."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("poc_data/procedural_multiview")
    )
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    split = generate_dataset(args.output, args.num_scenes, args.resolution, args.seed)
    print(json.dumps({name: len(items) for name, items in split["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
