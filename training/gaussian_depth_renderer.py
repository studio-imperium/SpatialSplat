from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

from training.scene_schema import OrthographicCamera


_EPS = 1e-8

# This is the pair of rotations used by the hosted TripoSplat viewer:
# yaw the PLY by +90 degrees around Y, then rotate its parent by 180 around X.
VIEWER_TRANSFORM = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GaussianCloud:
    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray


@dataclass(frozen=True)
class PerspectiveCamera:
    position: tuple[float, float, float] = (0.0, 0.3, 1.8)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vertical_fov_degrees: float = 45.0
    near: float = 0.01


@dataclass(frozen=True)
class GaussianDepthRender:
    depth: np.ndarray
    alpha: np.ndarray


def load_gaussian_ply(path: Path) -> GaussianCloud:
    vertices = PlyData.read(str(path))["vertex"].data

    def columns(names: tuple[str, ...]) -> np.ndarray:
        return np.column_stack([vertices[name] for name in names]).astype(np.float32)

    means = columns(("x", "y", "z"))
    scales = np.exp(columns(("scale_0", "scale_1", "scale_2")))
    rotations = columns(("rot_0", "rot_1", "rot_2", "rot_3"))
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), _EPS)
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -30.0, 30.0)))
    return GaussianCloud(means, scales, rotations, opacities)


def _quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), _EPS
    )
    w, x, y, z = q.T
    matrices = np.empty((len(q), 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), _EPS)


def _camera_rotation(camera: PerspectiveCamera) -> np.ndarray:
    position = np.asarray(camera.position, dtype=np.float32)
    target = np.asarray(camera.target, dtype=np.float32)
    up = np.asarray(camera.up, dtype=np.float32)
    forward = _normalize(target - position)
    right = _normalize(np.cross(forward, up))
    camera_up = _normalize(np.cross(right, forward))
    return np.stack([right, camera_up, forward], axis=0)


def render_gaussian_depth(
    cloud: GaussianCloud,
    width: int,
    height: int,
    camera: PerspectiveCamera | OrthographicCamera = PerspectiveCamera(),
    world_scale: float = 1.0,
    world_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma_radius: float = 3.0,
    minimum_pixel_sigma: float = 0.35,
    contribution_threshold: float = 1e-4,
) -> GaussianDepthRender:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if sigma_radius <= 0 or minimum_pixel_sigma <= 0:
        raise ValueError("sigma radii must be positive")
    if world_scale <= 0:
        raise ValueError("world scale must be positive")
    if cloud.means.size == 0:
        shape = (height, width)
        return GaussianDepthRender(
            depth=np.zeros(shape, dtype=np.float32),
            alpha=np.zeros(shape, dtype=np.float32),
        )

    translation = np.asarray(world_translation, dtype=np.float32)
    means = (cloud.means @ VIEWER_TRANSFORM.T) * world_scale + translation
    rotations = _quaternion_matrices(cloud.rotations)
    covariances = np.einsum(
        "nij,njk,nlk->nil",
        rotations,
        np.eye(3, dtype=np.float32)[None, :, :] * cloud.scales[:, None, :] ** 2,
        rotations,
    )
    covariances = np.einsum(
        "ij,njk,lk->nil", VIEWER_TRANSFORM, covariances, VIEWER_TRANSFORM
    )
    covariances *= world_scale**2

    world_to_camera = _camera_rotation(camera)
    camera_position = np.asarray(camera.position, dtype=np.float32)
    camera_means = (means - camera_position) @ world_to_camera.T
    camera_covariances = np.einsum(
        "ij,njk,lk->nil", world_to_camera, covariances, world_to_camera
    )
    z = camera_means[:, 2]
    near = camera.near if isinstance(camera, PerspectiveCamera) else 1e-4
    valid = (
        np.isfinite(camera_means).all(axis=1)
        & np.isfinite(camera_covariances).all(axis=(1, 2))
        & np.isfinite(cloud.opacities)
        & (z > near)
        & (cloud.opacities > contribution_threshold)
    )
    if not np.any(valid):
        shape = (height, width)
        return GaussianDepthRender(
            depth=np.zeros(shape, dtype=np.float32),
            alpha=np.zeros(shape, dtype=np.float32),
        )

    camera_means = camera_means[valid]
    camera_covariances = camera_covariances[valid]
    opacities = cloud.opacities[valid]
    z = camera_means[:, 2]

    jacobians = np.zeros((len(z), 2, 3), dtype=np.float32)
    if isinstance(camera, OrthographicCamera):
        pixel_scale = height / camera.ortho_scale
        projected = np.column_stack(
            [
                width * 0.5 + pixel_scale * camera_means[:, 0],
                height * 0.5 - pixel_scale * camera_means[:, 1],
            ]
        ).astype(np.float32)
        jacobians[:, 0, 0] = pixel_scale
        jacobians[:, 1, 1] = -pixel_scale
    else:
        focal = 0.5 * height / np.tan(
            np.deg2rad(camera.vertical_fov_degrees) * 0.5
        )
        projected = np.column_stack(
            [
                width * 0.5 + focal * camera_means[:, 0] / z,
                height * 0.5 - focal * camera_means[:, 1] / z,
            ]
        ).astype(np.float32)
        jacobians[:, 0, 0] = focal / z
        jacobians[:, 0, 2] = -focal * camera_means[:, 0] / (z * z)
        jacobians[:, 1, 1] = -focal / z
        jacobians[:, 1, 2] = focal * camera_means[:, 1] / (z * z)
    covariance_2d = np.einsum(
        "nij,njk,nlk->nil", jacobians, camera_covariances, jacobians
    )
    covariance_2d[:, 0, 0] += minimum_pixel_sigma**2
    covariance_2d[:, 1, 1] += minimum_pixel_sigma**2

    eigenvalues = np.linalg.eigvalsh(covariance_2d)
    radii = sigma_radius * np.sqrt(np.maximum(eigenvalues[:, 1], _EPS))
    finite = np.isfinite(projected).all(axis=1) & np.isfinite(radii) & (radii > 0)
    projected = projected[finite]
    covariance_2d = covariance_2d[finite]
    radii = radii[finite]
    opacities = opacities[finite]
    z = z[finite]

    order = np.argsort(z)
    transmittance = np.ones((height, width), dtype=np.float32)
    depth_numerator = np.zeros((height, width), dtype=np.float32)
    alpha_accumulated = np.zeros((height, width), dtype=np.float32)

    for index in order:
        center_x, center_y = projected[index]
        radius = float(radii[index])
        x0 = max(0, int(np.floor(center_x - radius)))
        x1 = min(width, int(np.ceil(center_x + radius + 1)))
        y0 = max(0, int(np.floor(center_y - radius)))
        y1 = min(height, int(np.ceil(center_y + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            continue

        covariance = covariance_2d[index]
        determinant = float(np.linalg.det(covariance))
        if determinant <= _EPS:
            continue
        inverse = np.linalg.inv(covariance)
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5 - center_x
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5 - center_y
        x_grid, y_grid = np.meshgrid(xs, ys)
        distance_squared = (
            inverse[0, 0] * x_grid * x_grid
            + 2 * inverse[0, 1] * x_grid * y_grid
            + inverse[1, 1] * y_grid * y_grid
        )
        inside = distance_squared <= sigma_radius**2
        if not np.any(inside):
            continue
        gaussian_alpha = np.minimum(
            opacities[index] * np.exp(-0.5 * distance_squared), 0.999
        ).astype(np.float32)
        gaussian_alpha[~inside] = 0

        patch_t = transmittance[y0:y1, x0:x1]
        contribution = patch_t * gaussian_alpha
        meaningful = contribution > contribution_threshold
        contribution[~meaningful] = 0
        depth_numerator[y0:y1, x0:x1] += contribution * z[index]
        alpha_accumulated[y0:y1, x0:x1] += contribution
        patch_t *= 1.0 - gaussian_alpha

    depth = np.divide(
        depth_numerator,
        alpha_accumulated,
        out=np.zeros_like(depth_numerator),
        where=alpha_accumulated > contribution_threshold,
    )
    alpha = np.clip(1.0 - transmittance, 0.0, 1.0)
    return GaussianDepthRender(depth=depth, alpha=alpha)
