import numpy as np

from training.generate_poc_data import build_scenes
from training.primitive_renderer import boundary_from_ids, render_scene


def test_scene_renderer_produces_finite_foreground() -> None:
    result = render_scene(build_scenes(128)[0])

    assert result.rgb.shape == (128, 128, 3)
    assert result.depth.shape == (128, 128)
    assert result.mask.any()
    assert np.all(result.depth[result.mask] > 0)
    assert np.all(result.depth[~result.mask] == 0)
    assert len(np.unique(result.primitive_ids[result.mask])) == 2


def test_boundary_tracks_visible_primitive_edges() -> None:
    result = render_scene(build_scenes(128)[3])
    boundary = boundary_from_ids(result.primitive_ids)

    assert boundary.any()
    assert not boundary[~result.mask].any()
