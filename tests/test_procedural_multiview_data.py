from pathlib import Path

from PIL import Image

from training.generate_procedural_multiview_data import generate_dataset
from training.multiview import SUPERVISION_VIEWS, ensure_supervision_views


def test_procedural_dataset_has_exact_six_view_visual_targets(tmp_path: Path) -> None:
    split = generate_dataset(tmp_path, num_scenes=5, resolution=48, seed=11)

    assert sum(len(items) for items in split["splits"].values()) == 5
    scene_dirs = sorted(path.parent for path in tmp_path.glob("*/scene.json"))
    assert len(scene_dirs) == 5
    for scene_dir in scene_dirs:
        assert (scene_dir / "generated_image.png").is_file()
        assert Image.open(scene_dir / "prepared_condition.png").size == (1024, 1024)
        views = ensure_supervision_views(scene_dir)
        assert tuple(views) == SUPERVISION_VIEWS
        assert all(view.rgb_path.is_file() for view in views.values())
