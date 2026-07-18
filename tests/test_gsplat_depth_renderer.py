import numpy as np
import torch

from training.gaussian_depth_renderer import VIEWER_TRANSFORM
from training.gsplat_depth_renderer import (
    MODEL_TO_WORLD_ROTATION,
    orthographic_camera_tensors,
    transform_gaussians,
)
from training.scene_schema import OrthographicCamera


def test_model_transform_matches_ply_and_viewer_transforms() -> None:
    ply_transform = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    expected = VIEWER_TRANSFORM @ ply_transform
    np.testing.assert_array_equal(
        np.asarray(MODEL_TO_WORLD_ROTATION, dtype=np.float32), expected
    )


def test_gaussian_world_transform_has_finite_gradients() -> None:
    means = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    scales = torch.tensor([[0.01, 0.02, 0.03]], requires_grad=True)
    quaternions = torch.tensor(
        [[1.0, 0.1, -0.2, 0.3]], requires_grad=True
    )
    world_means, world_covariances = transform_gaussians(
        means, scales, quaternions
    )
    (world_means.square().sum() + world_covariances.square().sum()).backward()

    for value in (means, scales, quaternions):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_orthographic_camera_projects_target_to_image_center() -> None:
    camera = OrthographicCamera(
        position=(1.7, 1.45, 1.7),
        target=(0.0, -0.08, 0.0),
        up=(0.0, 1.0, 0.0),
        ortho_scale=1.45,
        width=512,
        height=512,
    )
    view, intrinsics = orthographic_camera_tensors(
        camera, 512, 512, torch.device("cpu")
    )
    target = torch.tensor([*camera.target, 1.0])
    camera_target = view[0] @ target
    projected = torch.stack(
        (
            intrinsics[0, 0, 0] * camera_target[0] + intrinsics[0, 0, 2],
            intrinsics[0, 1, 1] * camera_target[1] + intrinsics[0, 1, 2],
        )
    )

    assert camera_target[2] > 0
    torch.testing.assert_close(projected, torch.tensor([256.0, 256.0]))
