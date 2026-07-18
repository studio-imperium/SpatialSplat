from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from training.scene_schema import OrthographicCamera

if TYPE_CHECKING:
    from triposplat import Gaussian


MODEL_WORLD_SCALE = 0.96
MODEL_WORLD_TRANSLATION = (0.0, -0.25, 0.0)

# TripoSplat writes raw decoder coordinates through Gaussian._DEFAULT_TRANSFORM,
# then the hosted viewer applies VIEWER_TRANSFORM. Their product maps raw xyz to
# world yzx. Keeping this explicit lets the differentiable render match the CPU
# evaluator without exporting and reloading a PLY.
MODEL_TO_WORLD_ROTATION = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)
PLY_TO_WORLD_ROTATION = (
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class TorchGaussianDepthRender:
    depth: torch.Tensor
    alpha: torch.Tensor


def quaternion_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to differentiable 3x3 rotation matrices."""
    q = quaternions / torch.linalg.vector_norm(
        quaternions, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def transform_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quaternions: torch.Tensor,
    rotation: tuple[tuple[float, float, float], ...] = MODEL_TO_WORLD_ROTATION,
    world_scale: float = MODEL_WORLD_SCALE,
    world_translation: tuple[float, float, float] = MODEL_WORLD_TRANSLATION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-space means and covariance matrices for gsplat."""
    dtype = torch.float32
    means = means.to(dtype=dtype)
    scales = scales.to(dtype=dtype)
    quaternions = quaternions.to(dtype=dtype)
    transform = torch.tensor(rotation, device=means.device, dtype=dtype)
    translation = torch.tensor(
        world_translation, device=means.device, dtype=dtype
    )

    rotations = quaternion_matrices(quaternions)
    raw_covariances = (
        rotations
        @ torch.diag_embed(scales.square())
        @ rotations.transpose(-1, -2)
    )
    world_means = (means @ transform.T) * world_scale + translation
    world_covariances = (
        transform[None] @ raw_covariances @ transform.T[None]
    ) * world_scale**2
    return world_means.contiguous(), world_covariances.contiguous()


def orthographic_camera_tensors(
    camera: OrthographicCamera,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build OpenCV-style world-to-camera and intrinsic matrices for gsplat."""
    position = torch.tensor(camera.position, device=device, dtype=torch.float32)
    target = torch.tensor(camera.target, device=device, dtype=torch.float32)
    up = torch.tensor(camera.up, device=device, dtype=torch.float32)
    forward = target - position
    forward = forward / torch.linalg.vector_norm(forward).clamp_min(1e-8)
    right = torch.linalg.cross(forward, up)
    right = right / torch.linalg.vector_norm(right).clamp_min(1e-8)
    camera_up = torch.linalg.cross(right, forward)
    camera_up = camera_up / torch.linalg.vector_norm(camera_up).clamp_min(1e-8)

    # gsplat uses image-space +Y downward. The CPU renderer keeps +Y upward and
    # applies the sign during projection, so flip the second camera row here.
    rotation = torch.stack((right, -camera_up, forward), dim=0)
    view = torch.eye(4, device=device, dtype=torch.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -(rotation @ position)

    pixel_scale = height / camera.ortho_scale
    intrinsics = torch.tensor(
        (
            (pixel_scale, 0.0, width * 0.5),
            (0.0, pixel_scale, height * 0.5),
            (0.0, 0.0, 1.0),
        ),
        device=device,
        dtype=torch.float32,
    )
    return view.unsqueeze(0), intrinsics.unsqueeze(0)


def render_gaussian_tensors(
    means: torch.Tensor,
    scales: torch.Tensor,
    quaternions: torch.Tensor,
    opacities: torch.Tensor,
    camera: OrthographicCamera,
    width: int,
    height: int,
    rotation: tuple[tuple[float, float, float], ...] = MODEL_TO_WORLD_ROTATION,
    world_scale: float = MODEL_WORLD_SCALE,
    world_translation: tuple[float, float, float] = MODEL_WORLD_TRANSLATION,
    minimum_pixel_sigma: float = 0.35,
) -> TorchGaussianDepthRender:
    """Rasterize differentiable alpha and expected camera-space depth."""
    if not means.is_cuda:
        raise ValueError("gsplat rendering requires CUDA tensors")
    from gsplat.rendering import rasterization

    world_means, world_covariances = transform_gaussians(
        means,
        scales,
        quaternions,
        rotation=rotation,
        world_scale=world_scale,
        world_translation=world_translation,
    )
    viewmats, intrinsics = orthographic_camera_tensors(
        camera, width, height, means.device
    )
    rendered_depth, rendered_alpha, _ = rasterization(
        means=world_means,
        quats=None,
        scales=None,
        opacities=opacities.reshape(-1).to(dtype=torch.float32).clamp(0.0, 0.999),
        # gsplat 1.5 validates a color tensor even for depth-only modes.
        colors=torch.zeros(
            (world_means.shape[0], 1),
            device=world_means.device,
            dtype=torch.float32,
        ),
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=1e-4,
        eps2d=minimum_pixel_sigma**2,
        packed=True,
        render_mode="ED",
        camera_model="ortho",
        covars=world_covariances,
    )
    return TorchGaussianDepthRender(
        depth=rendered_depth[0, ..., 0],
        alpha=rendered_alpha[0, ..., 0],
    )


def render_decoder_gaussian(
    gaussian: "Gaussian",
    camera: OrthographicCamera,
    width: int,
    height: int,
) -> TorchGaussianDepthRender:
    return render_gaussian_tensors(
        means=gaussian.get_xyz,
        scales=gaussian.get_scaling,
        quaternions=gaussian._rotation + gaussian.rots_bias[None, :],
        opacities=gaussian.get_opacity,
        camera=camera,
        width=width,
        height=height,
    )
