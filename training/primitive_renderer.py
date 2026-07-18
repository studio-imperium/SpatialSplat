from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from training.scene_schema import OrthographicCamera, Primitive, PrimitiveScene


_EPS = 1e-8
_NEAR = 1e-4


@dataclass(frozen=True)
class RenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    primitive_ids: np.ndarray


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    valid = np.isfinite(norm) & (norm > _EPS)
    return np.divide(value, norm, out=np.zeros_like(value), where=valid)


def camera_rays(camera: OrthographicCamera) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(camera.position, dtype=np.float32)
    target = np.asarray(camera.target, dtype=np.float32)
    world_up = np.asarray(camera.up, dtype=np.float32)
    forward = _normalize(target - position)
    right = _normalize(np.cross(forward, world_up))
    camera_up = _normalize(np.cross(right, forward))

    aspect = camera.width / camera.height
    horizontal_span = camera.ortho_scale * aspect
    vertical_span = camera.ortho_scale
    xs = ((np.arange(camera.width, dtype=np.float32) + 0.5) / camera.width - 0.5) * horizontal_span
    ys = (0.5 - (np.arange(camera.height, dtype=np.float32) + 0.5) / camera.height) * vertical_span
    x_grid, y_grid = np.meshgrid(xs, ys)

    origins = (
        position[None, None, :]
        + x_grid[..., None] * right[None, None, :]
        + y_grid[..., None] * camera_up[None, None, :]
    )
    directions = np.broadcast_to(forward, origins.shape).copy()
    return origins, directions


def _world_to_local(vectors: np.ndarray, yaw_degrees: float) -> np.ndarray:
    angle = np.deg2rad(yaw_degrees)
    c, s = np.cos(angle), np.sin(angle)
    result = vectors.copy()
    result[..., 0] = c * vectors[..., 0] - s * vectors[..., 2]
    result[..., 2] = s * vectors[..., 0] + c * vectors[..., 2]
    return result


def _local_to_world(vectors: np.ndarray, yaw_degrees: float) -> np.ndarray:
    angle = np.deg2rad(yaw_degrees)
    c, s = np.cos(angle), np.sin(angle)
    result = vectors.copy()
    result[..., 0] = c * vectors[..., 0] + s * vectors[..., 2]
    result[..., 2] = -s * vectors[..., 0] + c * vectors[..., 2]
    return result


def _intersect_sphere(
    origins: np.ndarray, directions: np.ndarray, primitive: Primitive
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(primitive.center, dtype=np.float32)
    radius = min(primitive.size) * 0.5
    offset = origins - center
    b = np.sum(offset * directions, axis=-1)
    c = np.sum(offset * offset, axis=-1) - radius * radius
    discriminant = b * b - c
    valid = discriminant >= 0
    root = np.sqrt(np.maximum(discriminant, 0))
    near_t = -b - root
    far_t = -b + root
    t = np.where(near_t > _NEAR, near_t, far_t)
    valid &= t > _NEAR
    t = np.where(valid, t, np.inf).astype(np.float32)
    hit = origins + directions * t[..., None]
    normals = _normalize(hit - center)
    normals[~valid] = 0
    return t, normals.astype(np.float32)


def _intersect_box(
    origins: np.ndarray, directions: np.ndarray, primitive: Primitive
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(primitive.center, dtype=np.float32)
    half_size = np.asarray(primitive.size, dtype=np.float32) * 0.5
    local_origins = _world_to_local(origins - center, primitive.yaw_degrees)
    local_directions = _world_to_local(directions, primitive.yaw_degrees)

    safe_directions = np.where(
        np.abs(local_directions) < _EPS,
        np.where(local_directions < 0, -_EPS, _EPS),
        local_directions,
    )
    t0 = (-half_size - local_origins) / safe_directions
    t1 = (half_size - local_origins) / safe_directions
    t_near = np.max(np.minimum(t0, t1), axis=-1)
    t_far = np.min(np.maximum(t0, t1), axis=-1)
    t = np.where(t_near > _NEAR, t_near, t_far)
    valid = (t_far >= np.maximum(t_near, _NEAR)) & (t > _NEAR)
    t = np.where(valid, t, np.inf).astype(np.float32)

    hit_local = local_origins + local_directions * t[..., None]
    face_distance = np.abs(np.abs(hit_local) - half_size)
    axis = np.argmin(face_distance, axis=-1)
    normals_local = np.zeros_like(hit_local)
    for component in range(3):
        selected = valid & (axis == component)
        normals_local[..., component][selected] = np.sign(hit_local[..., component][selected])
    normals = _local_to_world(normals_local, primitive.yaw_degrees)
    normals[~valid] = 0
    return t, normals.astype(np.float32)


def _intersect_cylinder(
    origins: np.ndarray, directions: np.ndarray, primitive: Primitive
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(primitive.center, dtype=np.float32)
    local_origins = _world_to_local(origins - center, primitive.yaw_degrees)
    local_directions = _world_to_local(directions, primitive.yaw_degrees)
    radius = min(primitive.size[0], primitive.size[2]) * 0.5
    half_height = primitive.size[1] * 0.5

    a = local_directions[..., 0] ** 2 + local_directions[..., 2] ** 2
    b = 2 * (
        local_origins[..., 0] * local_directions[..., 0]
        + local_origins[..., 2] * local_directions[..., 2]
    )
    c = local_origins[..., 0] ** 2 + local_origins[..., 2] ** 2 - radius * radius
    discriminant = b * b - 4 * a * c
    sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0))
    denominator = np.where(np.abs(2 * a) < _EPS, _EPS, 2 * a)
    side_t0 = (-b - sqrt_discriminant) / denominator
    side_t1 = (-b + sqrt_discriminant) / denominator
    side_t = np.where(side_t0 > _NEAR, side_t0, side_t1)
    side_y = local_origins[..., 1] + side_t * local_directions[..., 1]
    side_valid = (
        (discriminant >= 0)
        & (a > _EPS)
        & (side_t > _NEAR)
        & (np.abs(side_y) <= half_height + 1e-5)
    )
    side_t = np.where(side_valid, side_t, np.inf)

    cap_candidates: list[np.ndarray] = []
    cap_signs: list[float] = []
    for cap_sign in (-1.0, 1.0):
        cap_y = cap_sign * half_height
        cap_t = (cap_y - local_origins[..., 1]) / np.where(
            np.abs(local_directions[..., 1]) < _EPS, _EPS, local_directions[..., 1]
        )
        cap_x = local_origins[..., 0] + cap_t * local_directions[..., 0]
        cap_z = local_origins[..., 2] + cap_t * local_directions[..., 2]
        cap_valid = (
            (cap_t > _NEAR)
            & (cap_x * cap_x + cap_z * cap_z <= radius * radius + 1e-5)
            & (np.abs(local_directions[..., 1]) > _EPS)
        )
        cap_candidates.append(np.where(cap_valid, cap_t, np.inf))
        cap_signs.append(cap_sign)

    candidates = np.stack([side_t, *cap_candidates], axis=-1)
    selected = np.argmin(candidates, axis=-1)
    t = np.min(candidates, axis=-1)
    valid = np.isfinite(t)
    hit_local = local_origins + local_directions * t[..., None]

    normals_local = np.zeros_like(hit_local)
    side_selected = valid & (selected == 0)
    normals_local[..., 0][side_selected] = hit_local[..., 0][side_selected] / radius
    normals_local[..., 2][side_selected] = hit_local[..., 2][side_selected] / radius
    for candidate_index, cap_sign in enumerate(cap_signs, start=1):
        cap_selected = valid & (selected == candidate_index)
        normals_local[..., 1][cap_selected] = cap_sign

    normals = _local_to_world(normals_local, primitive.yaw_degrees)
    normals[~valid] = 0
    return np.where(valid, t, np.inf).astype(np.float32), normals.astype(np.float32)


def _intersect(
    origins: np.ndarray, directions: np.ndarray, primitive: Primitive
) -> tuple[np.ndarray, np.ndarray]:
    if primitive.kind == "sphere":
        return _intersect_sphere(origins, directions, primitive)
    if primitive.kind == "box":
        return _intersect_box(origins, directions, primitive)
    if primitive.kind == "cylinder":
        return _intersect_cylinder(origins, directions, primitive)
    raise ValueError(f"unsupported primitive kind: {primitive.kind}")


def render_scene(scene: PrimitiveScene) -> RenderResult:
    origins, directions = camera_rays(scene.camera)
    height, width = scene.camera.height, scene.camera.width
    depth = np.full((height, width), np.inf, dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    primitive_ids = np.full((height, width), -1, dtype=np.int16)
    light_direction = _normalize(np.asarray([0.6, 1.0, 0.4], dtype=np.float32))

    for primitive_index, primitive in enumerate(scene.primitives):
        candidate_depth, normals = _intersect(origins, directions, primitive)
        closer = candidate_depth < depth
        if not np.any(closer):
            continue
        diffuse = np.clip(np.sum(normals * light_direction, axis=-1), 0, 1)
        intensity = 0.48 + 0.52 * diffuse
        color = np.asarray(primitive.color, dtype=np.float32) / 255.0
        shaded = color[None, None, :] * intensity[..., None]
        rgb[closer] = shaded[closer]
        depth[closer] = candidate_depth[closer]
        primitive_ids[closer] = primitive_index

    mask = np.isfinite(depth)
    rgb[~mask] = 0
    depth = np.where(mask, depth, 0).astype(np.float32)
    return RenderResult(
        rgb=np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8),
        depth=depth,
        mask=mask,
        primitive_ids=primitive_ids,
    )


def boundary_from_ids(primitive_ids: np.ndarray) -> np.ndarray:
    boundary = np.zeros(primitive_ids.shape, dtype=bool)
    boundary[1:, :] |= primitive_ids[1:, :] != primitive_ids[:-1, :]
    boundary[:-1, :] |= primitive_ids[:-1, :] != primitive_ids[1:, :]
    boundary[:, 1:] |= primitive_ids[:, 1:] != primitive_ids[:, :-1]
    boundary[:, :-1] |= primitive_ids[:, :-1] != primitive_ids[:, 1:]
    return boundary & (primitive_ids >= 0)


def save_render(result: RenderResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = result.mask.astype(np.uint8) * 255
    rgba = np.concatenate([result.rgb, alpha[..., None]], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(output_dir / "primitive_control.png")
    Image.fromarray(alpha, mode="L").save(output_dir / "primitive_mask.png")
    boundary = boundary_from_ids(result.primitive_ids).astype(np.uint8) * 255
    Image.fromarray(boundary, mode="L").save(output_dir / "primitive_boundary.png")

    preview = np.zeros(result.depth.shape, dtype=np.uint8)
    if np.any(result.mask):
        foreground = result.depth[result.mask]
        depth_min, depth_max = float(foreground.min()), float(foreground.max())
        normalized = (result.depth - depth_min) / max(depth_max - depth_min, _EPS)
        preview[result.mask] = np.clip(np.round((1 - normalized[result.mask]) * 255), 0, 255)
    Image.fromarray(preview, mode="L").save(output_dir / "primitive_depth_preview.png")
    np.save(output_dir / "primitive_depth.npy", result.depth)
    np.save(output_dir / "primitive_alpha.npy", result.mask.astype(np.float32))
    np.save(output_dir / "primitive_ids.npy", result.primitive_ids)
