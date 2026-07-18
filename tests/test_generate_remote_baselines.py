from pathlib import Path

import pytest

from training.generate_remote_baselines import _scene_directories


def _add_scene(root: Path, name: str) -> None:
    scene = root / name
    scene.mkdir()
    (scene / "generated_image.png").write_bytes(b"image")


def test_scene_directories_are_sorted(tmp_path: Path) -> None:
    _add_scene(tmp_path, "02_second")
    _add_scene(tmp_path, "01_first")

    scenes = _scene_directories(tmp_path, [])

    assert [scene.name for scene in scenes] == ["01_first", "02_second"]


def test_scene_directories_respect_requested_order(tmp_path: Path) -> None:
    _add_scene(tmp_path, "01_first")
    _add_scene(tmp_path, "02_second")

    scenes = _scene_directories(tmp_path, ["02_second", "01_first"])

    assert [scene.name for scene in scenes] == ["02_second", "01_first"]


def test_scene_directories_reject_unknown_scene(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="unknown scene"):
        _scene_directories(tmp_path, ["missing"])
