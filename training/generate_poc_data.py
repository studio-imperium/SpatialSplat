from __future__ import annotations

import argparse
from pathlib import Path

from training.primitive_renderer import render_scene, save_render
from training.scene_schema import OrthographicCamera, Primitive, PrimitiveScene


COLORS = {
    "coral": (232, 88, 78),
    "cyan": (39, 177, 196),
    "yellow": (237, 188, 58),
    "green": (78, 170, 105),
    "violet": (133, 99, 189),
    "platform": (100, 108, 118),
}


def _box(name: str, center, size, color, yaw=0.0) -> Primitive:
    return Primitive(name, "box", center, size, COLORS[color], yaw)


def _sphere(name: str, center, diameter, color) -> Primitive:
    return Primitive(name, "sphere", center, (diameter, diameter, diameter), COLORS[color])


def _cylinder(name: str, center, diameter, height, color) -> Primitive:
    return Primitive(name, "cylinder", center, (diameter, height, diameter), COLORS[color])


def build_scenes(resolution: int) -> list[PrimitiveScene]:
    camera = OrthographicCamera(
        position=(1.7, 1.45, 1.7),
        target=(0.0, -0.08, 0.0),
        up=(0.0, 1.0, 0.0),
        ortho_scale=1.45,
        width=resolution,
        height=resolution,
    )
    platform = _box("platform", (0.0, -0.46, 0.0), (0.96, 0.08, 0.96), "platform")
    definitions = [
        (
            "01_center_cube",
            "A coral cube centered on a square platform.",
            (_box("cube", (0.0, -0.22, 0.0), (0.34, 0.40, 0.34), "coral", 12),),
        ),
        (
            "02_center_sphere",
            "A cyan sphere centered on a square platform.",
            (_sphere("sphere", (0.0, -0.23, 0.0), 0.38, "cyan"),),
        ),
        (
            "03_center_cylinder",
            "A yellow upright cylinder centered on a square platform.",
            (_cylinder("cylinder", (0.0, -0.21, 0.0), 0.32, 0.42, "yellow"),),
        ),
        (
            "04_cube_sphere",
            "A green cube on the left and violet sphere on the right.",
            (
                _box("cube", (-0.23, -0.28, 0.02), (0.26, 0.28, 0.26), "green", -10),
                _sphere("sphere", (0.23, -0.29, -0.02), 0.26, "violet"),
            ),
        ),
        (
            "05_front_back",
            "A short coral sphere in front of a tall cyan block.",
            (
                _sphere("front_sphere", (-0.05, -0.31, 0.24), 0.22, "coral"),
                _box("back_tower", (0.08, -0.20, -0.18), (0.26, 0.52, 0.24), "cyan", 8),
            ),
        ),
        (
            "06_three_step",
            "Three yellow blocks increasing in height from left to right.",
            (
                _box("step_1", (-0.28, -0.34, 0.02), (0.20, 0.16, 0.22), "yellow"),
                _box("step_2", (0.0, -0.28, 0.0), (0.20, 0.28, 0.22), "yellow"),
                _box("step_3", (0.28, -0.21, -0.02), (0.20, 0.42, 0.22), "yellow"),
            ),
        ),
        (
            "07_pillars",
            "Two green cylinders with a coral block between them.",
            (
                _cylinder("left_pillar", (-0.27, -0.24, 0.0), 0.18, 0.44, "green"),
                _box("center_block", (0.0, -0.31, 0.02), (0.24, 0.22, 0.24), "coral", 15),
                _cylinder("right_pillar", (0.27, -0.24, 0.0), 0.18, 0.44, "green"),
            ),
        ),
        (
            "08_diagonal_pair",
            "A violet tower behind and left of a cyan sphere.",
            (
                _box("tower", (-0.20, -0.20, -0.20), (0.24, 0.52, 0.24), "violet", -18),
                _sphere("sphere", (0.20, -0.29, 0.20), 0.26, "cyan"),
            ),
        ),
        (
            "09_four_corners",
            "Four differently colored shapes placed near the platform corners.",
            (
                _box("northwest", (-0.24, -0.33, -0.24), (0.18, 0.18, 0.18), "coral"),
                _sphere("northeast", (0.24, -0.33, -0.24), 0.18, "cyan"),
                _cylinder("southwest", (-0.24, -0.31, 0.24), 0.17, 0.22, "yellow"),
                _box("southeast", (0.24, -0.30, 0.24), (0.18, 0.24, 0.18), "green", 20),
            ),
        ),
        (
            "10_center_cluster",
            "A compact cluster with a tall center and three lower surrounding forms.",
            (
                _cylinder("center", (0.0, -0.19, 0.0), 0.20, 0.54, "violet"),
                _box("left", (-0.25, -0.32, 0.05), (0.20, 0.20, 0.20), "coral", -12),
                _sphere("right", (0.24, -0.33, 0.02), 0.18, "cyan"),
                _box("front", (0.02, -0.34, 0.27), (0.20, 0.16, 0.18), "yellow", 18),
            ),
        ),
    ]
    return [
        PrimitiveScene(scene_id, description, camera, (platform, *primitives))
        for scene_id, description, primitives in definitions
    ]


def build_heldout_scenes(resolution: int) -> list[PrimitiveScene]:
    camera = OrthographicCamera(
        position=(1.7, 1.45, 1.7),
        target=(0.0, -0.08, 0.0),
        up=(0.0, 1.0, 0.0),
        ortho_scale=1.45,
        width=resolution,
        height=resolution,
    )
    platform = _box("platform", (0.0, -0.46, 0.0), (0.96, 0.08, 0.96), "platform")
    definitions = [
        (
            "11_archway",
            "A freestanding stone archway made from two upright supports and one top beam.",
            (
                _box("left_support", (-0.24, -0.20, 0.0), (0.16, 0.44, 0.22), "coral"),
                _box("right_support", (0.24, -0.20, 0.0), (0.16, 0.44, 0.22), "coral"),
                _box("top_beam", (0.0, 0.07, 0.0), (0.64, 0.12, 0.22), "yellow"),
            ),
        ),
        (
            "12_raised_table",
            "A raised square worktable with a thin top and four clearly separated legs.",
            (
                _box("left_back_leg", (-0.25, -0.29, -0.21), (0.10, 0.26, 0.10), "green"),
                _box("right_back_leg", (0.25, -0.29, -0.21), (0.10, 0.26, 0.10), "green"),
                _box("left_front_leg", (-0.25, -0.29, 0.21), (0.10, 0.26, 0.10), "green"),
                _box("right_front_leg", (0.25, -0.29, 0.21), (0.10, 0.26, 0.10), "green"),
                _box("table_top", (0.0, -0.11, 0.0), (0.68, 0.10, 0.58), "cyan", 5),
            ),
        ),
        (
            "13_diagonal_procession",
            "Three separate forms arranged diagonally from a short front-left sphere through a center block to a tall rear-right cylinder.",
            (
                _sphere("front_sphere", (-0.28, -0.33, 0.26), 0.18, "violet"),
                _box("center_block", (0.0, -0.28, 0.0), (0.24, 0.28, 0.24), "yellow", -12),
                _cylinder("rear_cylinder", (0.27, -0.20, -0.25), 0.20, 0.44, "coral"),
            ),
        ),
    ]
    return [
        PrimitiveScene(scene_id, description, camera, (platform, *primitives))
        for scene_id, description, primitives in definitions
    ]


def generation_prompt(scene: PrimitiveScene) -> str:
    return (
        "Transform the attached primitive proxy into a richly rendered, believable miniature scene while "
        "preserving the exact fixed isometric camera, platform footprint, object count, broad silhouette, "
        "relative positions, sizes, and occlusion order. Decorative texture and small details are encouraged. "
        "Do not move, remove, merge, or add any large object or structure. Keep the full platform visible, "
        "use a clean neutral background, and add no text or watermark. Scene description: "
        f"{scene.description}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic primitive artifacts for the POC.")
    parser.add_argument("--output", type=Path, default=Path("poc_data"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scenes = (
        build_heldout_scenes(args.resolution)
        if args.split == "heldout"
        else build_scenes(args.resolution)
    )
    for scene in scenes:
        scene_dir = args.output / scene.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene.write_json(scene_dir / "scene.json")
        (scene_dir / "generation_prompt.txt").write_text(generation_prompt(scene) + "\n", encoding="utf-8")
        save_render(render_scene(scene), scene_dir)
        print(scene_dir)


if __name__ == "__main__":
    main()
