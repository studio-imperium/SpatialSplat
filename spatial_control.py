from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F


CONTROL_FEATURE_NAMES = (
    "signed_distance",
    "surface_band_narrow",
    "surface_band_wide",
    "occupancy",
    "normal_x",
    "normal_y",
    "normal_z",
    "kind_box",
    "kind_sphere",
    "kind_cylinder",
    "is_support",
    "valid",
)
DEFAULT_CONTROL_LAYERS = (3, 7, 11, 15, 19, 23)
SUPPORT_NAMES = frozenset({"ground", "floor", "terrain"})

MODEL_WORLD_SCALE = 0.96
MODEL_WORLD_TRANSLATION = (0.0, -0.25, 0.0)
MODEL_TO_WORLD_ROTATION = (
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
)


def control_scale_for_step(
    control_scale: float,
    step_index: int,
    total_steps: int,
    control_end: float,
) -> float:
    """Apply geometry control early, then disable it for final cleanup steps."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= control_end <= 1.0:
        raise ValueError("control_end must be between 0 and 1")
    if not 0 <= step_index < total_steps:
        raise ValueError("step_index must identify a sampling step")
    active_steps = math.ceil(total_steps * control_end)
    return float(control_scale) if step_index < active_steps else 0.0


def _scene_dict(scene: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(scene, dict):
        return scene
    return json.loads(Path(scene).read_text(encoding="utf-8"))


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three numbers")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _rotation_matrix_xyz(
    rotation_degrees: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z = (math.radians(value) for value in rotation_degrees)
    a, b = math.cos(x), math.sin(x)
    c, d = math.cos(y), math.sin(y)
    e, f = math.cos(z), math.sin(z)
    return (
        (c * e, -c * f, d),
        (a * f + b * e * d, a * e - b * f * d, -b * c),
        (b * f - a * e * d, b * e + a * f * d, a * c),
    )


def _transform_vector(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _color_rgb(value: Any) -> list[int]:
    if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
        try:
            return [int(value[index : index + 2], 16) for index in (1, 3, 5)]
        except ValueError:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [max(0, min(255, round(float(component)))) for component in value]
        except (TypeError, ValueError):
            pass
    return [136, 136, 136]


def _worldsketch_ground_primitive(ground: Any) -> dict[str, Any] | None:
    if not isinstance(ground, dict):
        return None
    points: list[tuple[float, float, float]] = []
    colors: list[Any] = []
    for stroke_index, stroke in enumerate(ground.get("strokes", [])):
        if not isinstance(stroke, dict) or stroke.get("mode", "paint") != "paint":
            continue
        radius = max(0.0, float(stroke.get("radius", 0.0)))
        colors.append(stroke.get("color"))
        for point_index, point in enumerate(stroke.get("points", [])):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"ground stroke {stroke_index + 1} point {point_index + 1} "
                    "must contain two numbers"
                )
            x, z = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(z):
                raise ValueError("ground stroke points must be finite")
            points.extend(((x - radius, 0.0, z - radius), (x + radius, 0.05, z + radius)))
    if not points:
        return None
    minimum = tuple(min(point[axis] for point in points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in points) for axis in range(3))
    return {
        "name": "ground",
        "kind": "box",
        "position": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
        "rotation_degrees": [0.0, 0.0, 0.0],
        "color": colors[0] if colors else "#587553",
    }


def canonicalize_scene(
    scene: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Convert canonical or WorldSketch primitive JSON to the trained POC frame."""
    data = _scene_dict(scene)
    primitives = data.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise ValueError("scene must contain a non-empty primitives array")
    if all(
        isinstance(primitive, dict)
        and {"kind", "center", "size"}.issubset(primitive)
        for primitive in primitives
    ):
        return data

    source_primitives: list[dict[str, Any]] = []
    ground = _worldsketch_ground_primitive(data.get("ground"))
    if ground is not None:
        source_primitives.append(ground)
    for index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            raise ValueError(f"primitive {index + 1} must be an object")
        kind = str(primitive.get("type", "")).lower()
        if kind not in {"box", "sphere", "cylinder"}:
            raise ValueError(
                f"primitive {index + 1} has unsupported type {kind or '<missing>'!r}"
            )
        position = _vector3(primitive.get("position"), f"primitive {index + 1} position")
        size = _vector3(primitive.get("scale"), f"primitive {index + 1} scale")
        if any(component <= 0 for component in size):
            raise ValueError(f"primitive {index + 1} scale must be positive")
        rotation_radians = _vector3(
            primitive.get("rotation", (0.0, 0.0, 0.0)),
            f"primitive {index + 1} rotation",
        )
        source_primitives.append(
            {
                "name": primitive.get("name", f"primitive_{index + 1:03d}"),
                "kind": kind,
                "position": position,
                "size": size,
                "rotation_degrees": tuple(
                    math.degrees(value) for value in rotation_radians
                ),
                "color": primitive.get("color", "#888888"),
            }
        )

    corners: list[tuple[float, float, float]] = []
    for primitive in source_primitives:
        position = _vector3(primitive["position"], "primitive position")
        size = _vector3(primitive["size"], "primitive size")
        rotation = _vector3(primitive["rotation_degrees"], "primitive rotation")
        matrix = _rotation_matrix_xyz(rotation)
        for mask in range(8):
            local = tuple(
                size[axis] * (0.5 if mask & (1 << axis) else -0.5)
                for axis in range(3)
            )
            rotated = _transform_vector(matrix, local)
            corners.append(
                tuple(position[axis] + rotated[axis] for axis in range(3))
            )

    minimum = tuple(min(point[axis] for point in corners) for axis in range(3))
    maximum = tuple(max(point[axis] for point in corners) for axis in range(3))
    source_center = tuple(
        (minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)
    )
    right = (1 / math.sqrt(2), 0.0, -1 / math.sqrt(2))
    camera_up = (-1 / math.sqrt(6), 2 / math.sqrt(6), -1 / math.sqrt(6))
    projected_x = [
        sum((point[axis] - source_center[axis]) * right[axis] for axis in range(3))
        for point in corners
    ]
    projected_y = [
        sum(
            (point[axis] - source_center[axis]) * camera_up[axis]
            for axis in range(3)
        )
        for point in corners
    ]
    capture_span = 1.02 * max(
        max(projected_x) - min(projected_x),
        max(projected_y) - min(projected_y),
    )
    if capture_span <= 1e-8:
        raise ValueError("scene primitives have no usable spatial extent")
    canonical_span = 1.45
    scale = canonical_span / capture_span
    target = (0.0, -0.08, 0.0)

    canonical_primitives = []
    for primitive in source_primitives:
        position = _vector3(primitive["position"], "primitive position")
        size = _vector3(primitive["size"], "primitive size")
        rotation = _vector3(primitive["rotation_degrees"], "primitive rotation")
        canonical_primitives.append(
            {
                "name": primitive["name"],
                "kind": primitive["kind"],
                "center": [
                    (position[axis] - source_center[axis]) * scale + target[axis]
                    for axis in range(3)
                ],
                "size": [component * scale for component in size],
                "color": _color_rgb(primitive["color"]),
                "yaw_degrees": rotation[1],
                "rotation_degrees": list(rotation),
            }
        )
    return {
        "scene_id": data.get("scene_id", "worldsketch_upload"),
        "description": data.get("description", "Imported WorldSketch primitives."),
        "source_schema": "worldsketch",
        "camera": {
            "position": [1.7, 1.45, 1.7],
            "target": list(target),
            "up": [0.0, 1.0, 0.0],
            "ortho_scale": canonical_span,
            "width": 512,
            "height": 512,
        },
        "normalization": {
            "source_center": list(source_center),
            "uniform_scale": scale,
        },
        "primitives": canonical_primitives,
    }


def sobol_world_positions(
    token_count: int = 8192,
    seed: int = 123,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the TripoSplat VecSeq anchors in the calibrated POC world frame."""
    sobol = torch.quasirandom.SobolEngine(
        dimension=3, scramble=True, seed=seed
    ).draw(token_count)
    raw = sobol - 0.5
    rotation = torch.tensor(MODEL_TO_WORLD_ROTATION, dtype=torch.float32)
    translation = torch.tensor(MODEL_WORLD_TRANSLATION, dtype=torch.float32)
    world = (raw @ rotation.T) * MODEL_WORLD_SCALE + translation
    return world.to(device=device)


def _world_to_local(points: torch.Tensor, yaw_degrees: float) -> torch.Tensor:
    angle = math.radians(float(yaw_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    local = points.clone()
    local[..., 0] = c * points[..., 0] - s * points[..., 2]
    local[..., 2] = s * points[..., 0] + c * points[..., 2]
    return local


def _local_to_world(vectors: torch.Tensor, yaw_degrees: float) -> torch.Tensor:
    angle = math.radians(float(yaw_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    world = vectors.clone()
    world[..., 0] = c * vectors[..., 0] + s * vectors[..., 2]
    world[..., 2] = -s * vectors[..., 0] + c * vectors[..., 2]
    return world


def _torch_rotation_matrix_xyz(
    rotation_degrees: tuple[float, float, float], reference: torch.Tensor
) -> torch.Tensor:
    return torch.tensor(
        _rotation_matrix_xyz(rotation_degrees),
        device=reference.device,
        dtype=reference.dtype,
    )


def _primitive_world_to_local(
    vectors: torch.Tensor, primitive: dict[str, Any]
) -> torch.Tensor:
    rotation = _vector3(
        primitive.get(
            "rotation_degrees", (0.0, primitive.get("yaw_degrees", 0.0), 0.0)
        ),
        "primitive rotation_degrees",
    )
    if abs(rotation[0]) < 1e-8 and abs(rotation[2]) < 1e-8:
        return _world_to_local(vectors, rotation[1])
    return vectors @ _torch_rotation_matrix_xyz(rotation, vectors)


def _primitive_local_to_world(
    vectors: torch.Tensor, primitive: dict[str, Any]
) -> torch.Tensor:
    rotation = _vector3(
        primitive.get(
            "rotation_degrees", (0.0, primitive.get("yaw_degrees", 0.0), 0.0)
        ),
        "primitive rotation_degrees",
    )
    if abs(rotation[0]) < 1e-8 and abs(rotation[2]) < 1e-8:
        return _local_to_world(vectors, rotation[1])
    matrix = _torch_rotation_matrix_xyz(rotation, vectors)
    return vectors @ matrix.T


def _sphere_sdf_normal(
    local: torch.Tensor, size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = size.min() * 0.5
    length = torch.linalg.vector_norm(local, dim=-1)
    normal = local / length.unsqueeze(-1).clamp_min(1e-8)
    return length - radius, normal


def _box_sdf_normal(
    local: torch.Tensor, size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    half_size = size * 0.5
    q = local.abs() - half_size
    outside = q.clamp_min(0)
    outside_length = torch.linalg.vector_norm(outside, dim=-1)
    distance = outside_length + q.max(dim=-1).values.clamp_max(0)

    sign = torch.where(local >= 0, 1.0, -1.0)
    outside_normal = outside * sign
    outside_normal = outside_normal / torch.linalg.vector_norm(
        outside_normal, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    inside_axis = q.argmax(dim=-1)
    inside_normal = torch.zeros_like(local)
    inside_normal.scatter_(
        -1, inside_axis.unsqueeze(-1), sign.gather(-1, inside_axis.unsqueeze(-1))
    )
    normal = torch.where(
        (outside_length > 0).unsqueeze(-1), outside_normal, inside_normal
    )
    return distance, normal


def _cylinder_sdf_normal(
    local: torch.Tensor, size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = torch.minimum(size[0], size[2]) * 0.5
    half_height = size[1] * 0.5
    radial_length = torch.linalg.vector_norm(local[..., (0, 2)], dim=-1)
    radial_distance = radial_length - radius
    vertical_distance = local[..., 1].abs() - half_height
    pair = torch.stack((radial_distance, vertical_distance), dim=-1)
    distance = torch.linalg.vector_norm(pair.clamp_min(0), dim=-1)
    distance = distance + pair.max(dim=-1).values.clamp_max(0)

    radial_normal = torch.zeros_like(local)
    radial_normal[..., 0] = local[..., 0] / radial_length.clamp_min(1e-8)
    radial_normal[..., 2] = local[..., 2] / radial_length.clamp_min(1e-8)
    cap_normal = torch.zeros_like(local)
    cap_normal[..., 1] = torch.where(local[..., 1] >= 0, 1.0, -1.0)

    outside_radial = radial_distance.clamp_min(0)
    outside_vertical = vertical_distance.clamp_min(0)
    outside_normal = (
        radial_normal * outside_radial.unsqueeze(-1)
        + cap_normal * outside_vertical.unsqueeze(-1)
    )
    outside_normal = outside_normal / torch.linalg.vector_norm(
        outside_normal, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    inside_normal = torch.where(
        (radial_distance >= vertical_distance).unsqueeze(-1),
        radial_normal,
        cap_normal,
    )
    outside = (outside_radial > 0) | (outside_vertical > 0)
    normal = torch.where(outside.unsqueeze(-1), outside_normal, inside_normal)
    return distance, normal


def _primitive_sdf_normal(
    points: torch.Tensor, primitive: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    center = torch.tensor(primitive["center"], device=points.device, dtype=points.dtype)
    size = torch.tensor(primitive["size"], device=points.device, dtype=points.dtype)
    local = _primitive_world_to_local(points - center, primitive)
    kind = primitive["kind"]
    if kind == "sphere":
        distance, normal = _sphere_sdf_normal(local, size)
    elif kind == "box":
        distance, normal = _box_sdf_normal(local, size)
    elif kind == "cylinder":
        distance, normal = _cylinder_sdf_normal(local, size)
    else:
        raise ValueError(f"unsupported primitive kind: {kind}")
    return distance, _primitive_local_to_world(normal, primitive)


def scene_control_tensor(
    scene: str | Path | dict[str, Any],
    *,
    token_count: int = 8192,
    sobol_seed: int = 123,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build `[1, token_count, 12]` primitive control features."""
    data = canonicalize_scene(scene)
    primitives = data.get("primitives", [])
    if not primitives:
        raise ValueError("scene contains no primitives")

    points = sobol_world_positions(token_count, sobol_seed, device=device)
    distances = []
    normals = []
    for primitive in primitives:
        distance, normal = _primitive_sdf_normal(points, primitive)
        distances.append(distance)
        normals.append(normal)
    distance_stack = torch.stack(distances, dim=-1)
    normal_stack = torch.stack(normals, dim=1)
    nearest = distance_stack.argmin(dim=-1)
    union_distance = distance_stack.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
    nearest_normal = normal_stack[
        torch.arange(points.shape[0], device=points.device), nearest
    ]

    kind_lookup = {"box": 0, "sphere": 1, "cylinder": 2}
    kinds = torch.tensor(
        [kind_lookup[item["kind"]] for item in primitives], device=points.device
    )
    kind_features = F.one_hot(kinds[nearest], num_classes=3).to(torch.float32)
    support = torch.tensor(
        [item["name"].lower() in SUPPORT_NAMES for item in primitives],
        device=points.device,
        dtype=torch.float32,
    )[nearest]

    features = torch.cat(
        (
            (union_distance / 0.25).clamp(-1, 1).unsqueeze(-1),
            torch.tanh(union_distance / 0.02).unsqueeze(-1),
            torch.tanh(union_distance / 0.08).unsqueeze(-1),
            torch.where(union_distance <= 0, 1.0, -1.0).unsqueeze(-1),
            nearest_normal,
            kind_features,
            support.unsqueeze(-1),
            torch.ones_like(support).unsqueeze(-1),
        ),
        dim=-1,
    )
    return features.unsqueeze(0).contiguous()


class ControlInjection(nn.Module):
    def __init__(self, model_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.state = nn.Linear(model_channels, hidden_channels, bias=False)
        self.control = nn.Linear(hidden_channels, hidden_channels)
        self.time = nn.Linear(model_channels, hidden_channels, bias=False)
        self.output = nn.Linear(hidden_channels, model_channels, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        control: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.state.weight.dtype
        normalized = F.layer_norm(
            hidden_states.float(), hidden_states.shape[-1:]
        ).to(dtype)
        value = self.state(normalized)
        value = value + self.control(control.to(dtype))
        value = value + self.time(timestep.to(dtype)).unsqueeze(1)
        return self.output(F.silu(value)).to(hidden_states.dtype)


class SpatialControlAdapter(nn.Module):
    def __init__(
        self,
        *,
        model_channels: int = 1024,
        feature_dim: int = len(CONTROL_FEATURE_NAMES),
        hidden_channels: int = 64,
        layer_indices: Iterable[int] = DEFAULT_CONTROL_LAYERS,
    ) -> None:
        super().__init__()
        self.model_channels = int(model_channels)
        self.feature_dim = int(feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.injections = nn.ModuleDict(
            {
                str(index): ControlInjection(self.model_channels, self.hidden_channels)
                for index in self.layer_indices
            }
        )

    def encode(self, control: torch.Tensor) -> torch.Tensor:
        dtype = self.feature_encoder[0].weight.dtype
        return self.feature_encoder(control.to(dtype))

    def residual(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        encoded_control: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.injections[str(layer_index)](
            hidden_states, encoded_control, timestep
        )


def attach_spatial_control(
    model: nn.Module,
    *,
    hidden_channels: int = 64,
    layer_indices: Iterable[int] = DEFAULT_CONTROL_LAYERS,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> SpatialControlAdapter:
    if getattr(model, "spatial_control_adapter", None) is not None:
        raise ValueError("model already has a spatial control adapter")
    adapter = SpatialControlAdapter(
        model_channels=int(model.model_channels),
        hidden_channels=hidden_channels,
        layer_indices=layer_indices,
    )
    adapter.to(device=device or model.device, dtype=dtype)
    model.spatial_control_adapter = adapter
    return adapter


def control_parameters(model: nn.Module):
    adapter = getattr(model, "spatial_control_adapter", None)
    if not isinstance(adapter, SpatialControlAdapter):
        raise ValueError("model has no spatial control adapter")
    yield from adapter.parameters()


def save_spatial_control(model: nn.Module, path: Path) -> None:
    adapter = getattr(model, "spatial_control_adapter", None)
    if not isinstance(adapter, SpatialControlAdapter):
        raise ValueError("model has no spatial control adapter")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in adapter.state_dict().items()
    }
    safetensors.torch.save_file(state, str(path))


def load_spatial_control(
    model: nn.Module,
    weights_path: str | Path,
    config_path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected_features = tuple(config["feature_names"])
    if expected_features != CONTROL_FEATURE_NAMES:
        raise ValueError("control feature schema does not match this loader")
    adapter = attach_spatial_control(
        model,
        hidden_channels=int(config["hidden_channels"]),
        layer_indices=config["layer_indices"],
        device=device,
        dtype=dtype,
    )
    state = safetensors.torch.load_file(str(weights_path), device="cpu")
    adapter.load_state_dict(
        {
            name: value.to(
                device=next(adapter.parameters()).device,
                dtype=next(adapter.parameters()).dtype,
            )
            for name, value in state.items()
        },
        strict=True,
    )
    return config
