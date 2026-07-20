from pathlib import Path
import json

import numpy as np
from PIL import Image
import pytest

from training.generate_controlnet_images import (
    DEFAULT_SCENES,
    alpha_coverage,
    apply_condition_alpha,
    build_prompt,
    prepare_controls,
    prepare_init_image,
    select_scene_dirs,
    validate_alpha,
    write_generation_summary,
)


def test_prompt_prioritizes_geometry_and_distinguishes_floor():
    grounded = build_prompt({"description": "A table on grass.", "has_floor": True})
    standalone = build_prompt({"description": "A robot.", "has_floor": False})

    assert "orientation" in grounded
    assert "bounding boxes" in grounded
    assert "surface heights" in grounded
    assert "ground slab" in grounded
    assert "no floor" in standalone


def test_prepare_controls_returns_depth_then_boundary(tmp_path: Path):
    Image.new("L", (8, 8), 100).save(tmp_path / "primitive_depth_preview.png")
    Image.new("L", (8, 8), 200).save(tmp_path / "primitive_boundary.png")

    depth, boundary = prepare_controls(tmp_path, (16, 12))

    assert depth.size == (16, 12)
    assert boundary.size == (16, 12)
    assert depth.mode == "RGB"
    assert np.asarray(depth)[0, 0, 0] == 100
    assert np.asarray(boundary)[0, 0, 0] == 200


def test_prepare_init_image_places_proxy_on_white(tmp_path: Path):
    Image.new("RGB", (4, 4), (120, 40, 20)).save(tmp_path / "primitive_control.png")
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    mask.save(tmp_path / "primitive_mask.png")

    image = prepare_init_image(tmp_path, (4, 4))

    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((1, 1)) == (120, 40, 20)


def test_alpha_validation_rejects_opaque_result():
    opaque = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
    coverage = alpha_coverage(opaque)

    with pytest.raises(ValueError, match="neutral background"):
        validate_alpha(coverage)


def test_condition_alpha_protects_neutral_subject(tmp_path: Path):
    image = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", (8, 8), 0)
    for y in range(3, 6):
        for x in range(2, 7):
            mask.putpixel((x, y), 255)
    mask.save(tmp_path / "primitive_mask.png")

    rgba = apply_condition_alpha(image, tmp_path)
    alpha = np.asarray(rgba.getchannel("A"))

    assert alpha[4, 4] == 255
    assert alpha[0, 0] == 0


def test_default_pilot_selection_is_exact(tmp_path: Path):
    for name in DEFAULT_SCENES:
        (tmp_path / name).mkdir()
    (tmp_path / "unused").mkdir()

    selected = select_scene_dirs(tmp_path, names=None, all_scenes=False)

    assert [path.name for path in selected] == sorted(DEFAULT_SCENES)


def test_missing_requested_scene_is_reported(tmp_path: Path):
    (tmp_path / "present").mkdir()

    with pytest.raises(FileNotFoundError, match="missing"):
        select_scene_dirs(tmp_path, ["present", "absent"], all_scenes=False)


def test_summary_collects_existing_scene_metadata(tmp_path: Path):
    scenes = [tmp_path / "a", tmp_path / "b"]
    for index, scene in enumerate(scenes):
        scene.mkdir()
        (scene / "generation_metadata.json").write_text(
            json.dumps({"scene_id": scene.name, "index": index}), encoding="utf-8"
        )

    summary = write_generation_summary(tmp_path, scenes)

    assert [item["scene_id"] for item in summary] == ["a", "b"]
    assert json.loads((tmp_path / "controlnet_generation_summary.json").read_text()) == summary
