import numpy as np

from training.generate_poc_data import build_scenes
from training.multiview import SUPERVISION_VIEWS, supervision_cameras
from training.primitive_renderer import camera_rays, render_scene
from training.scene_schema import PrimitiveScene


def test_supervision_cameras_face_the_same_target() -> None:
    base = build_scenes(64)[0].camera
    cameras = supervision_cameras(base)

    assert tuple(cameras) == SUPERVISION_VIEWS
    for camera in cameras.values():
        _, directions = camera_rays(camera)
        expected = np.asarray(camera.target) - np.asarray(camera.position)
        expected /= np.linalg.norm(expected)
        assert np.allclose(directions[0, 0], expected, atol=1e-6)

    assert np.allclose(
        np.asarray(cameras["left"].position) + np.asarray(cameras["right"].position),
        2 * np.asarray(base.target),
    )
    assert np.allclose(
        np.asarray(cameras["front"].position) + np.asarray(cameras["back"].position),
        2 * np.asarray(base.target),
    )


def test_all_supervision_views_render_foreground() -> None:
    scene = build_scenes(64)[4]
    for camera in supervision_cameras(scene.camera).values():
        result = render_scene(
            PrimitiveScene(
                scene.scene_id,
                scene.description,
                camera,
                scene.primitives,
            )
        )
        assert result.mask.any()
        assert np.isfinite(result.depth[result.mask]).all()
