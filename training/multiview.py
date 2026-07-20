from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.primitive_renderer import boundary_from_ids, render_scene
from training.scene_schema import OrthographicCamera, Primitive, PrimitiveScene


SUPERVISION_VIEWS = ("isometric", "top", "left", "right", "front", "back")


@dataclass(frozen=True)
class SupervisionView:
    name: str
    camera: OrthographicCamera
    depth_path: Path
    mask_path: Path
    boundary_path: Path
    ids_path: Path


@dataclass(frozen=True)
class _ViewPaths:
    depth: Path
    mask: Path
    boundary: Path
    ids: Path


def _camera_distance(camera: OrthographicCamera) -> float:
    position = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    return float(np.linalg.vector_norm(position - target))


def supervision_cameras(
    camera: OrthographicCamera,
) -> dict[str, OrthographicCamera]:
    target = tuple(camera.target)
    distance = _camera_distance(camera)
    shared = {
        "target": target,
        "ortho_scale": camera.ortho_scale,
        "width": camera.width,
        "height": camera.height,
    }
    return {
        "isometric": camera,
        "top": OrthographicCamera(
            position=(target[0], target[1] + distance, target[2]),
            up=(0.0, 0.0, -1.0),
            **shared,
        ),
        "left": OrthographicCamera(
            position=(target[0] - distance, target[1], target[2]),
            up=(0.0, 1.0, 0.0),
            **shared,
        ),
        "right": OrthographicCamera(
            position=(target[0] + distance, target[1], target[2]),
            up=(0.0, 1.0, 0.0),
            **shared,
        ),
        "front": OrthographicCamera(
            position=(target[0], target[1], target[2] + distance),
            up=(0.0, 1.0, 0.0),
            **shared,
        ),
        "back": OrthographicCamera(
            position=(target[0], target[1], target[2] - distance),
            up=(0.0, 1.0, 0.0),
            **shared,
        ),
    }


def _paths(scene_dir: Path, name: str) -> _ViewPaths:
    prefix = "primitive" if name == "isometric" else f"primitive_{name}"
    return _ViewPaths(
        depth=scene_dir / f"{prefix}_depth.npy",
        mask=scene_dir / f"{prefix}_mask.png",
        boundary=scene_dir / f"{prefix}_boundary.png",
        ids=scene_dir / f"{prefix}_ids.npy",
    )


def _scene_from_json(scene_dir: Path) -> PrimitiveScene:
    data = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    return PrimitiveScene(
        scene_id=data["scene_id"],
        description=data["description"],
        camera=OrthographicCamera(**data["camera"]),
        primitives=tuple(Primitive(**primitive) for primitive in data["primitives"]),
    )


def _save_target(scene_dir: Path, name: str, result) -> None:
    prefix = f"primitive_{name}"
    mask = result.mask.astype(np.uint8) * 255
    boundary = boundary_from_ids(result.primitive_ids).astype(np.uint8) * 255
    np.save(scene_dir / f"{prefix}_depth.npy", result.depth)
    np.save(scene_dir / f"{prefix}_alpha.npy", result.mask.astype(np.float32))
    np.save(scene_dir / f"{prefix}_ids.npy", result.primitive_ids)
    Image.fromarray(mask, mode="L").save(scene_dir / f"{prefix}_mask.png")
    Image.fromarray(boundary, mode="L").save(
        scene_dir / f"{prefix}_boundary.png"
    )

    preview = np.zeros(result.depth.shape, dtype=np.uint8)
    if np.any(result.mask):
        foreground = result.depth[result.mask]
        near, far = float(foreground.min()), float(foreground.max())
        normalized = (result.depth - near) / max(far - near, 1e-8)
        preview[result.mask] = np.clip(
            np.round((1.0 - normalized[result.mask]) * 255), 0, 255
        ).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(
        scene_dir / f"{prefix}_depth_preview.png"
    )


def ensure_supervision_views(
    scene_dir: Path,
    names: tuple[str, ...] | list[str] = SUPERVISION_VIEWS,
    force: bool = False,
) -> dict[str, SupervisionView]:
    unknown = sorted(set(names) - set(SUPERVISION_VIEWS))
    if unknown:
        raise ValueError(f"unknown supervision view(s): {', '.join(unknown)}")

    scene = _scene_from_json(scene_dir)
    cameras = supervision_cameras(scene.camera)
    views: dict[str, SupervisionView] = {}
    for name in names:
        paths = _paths(scene_dir, name)
        if name != "isometric" and (
            force
            or not paths.depth.is_file()
            or not paths.mask.is_file()
            or not paths.boundary.is_file()
            or not paths.ids.is_file()
        ):
            render = render_scene(
                PrimitiveScene(
                    scene.scene_id,
                    scene.description,
                    cameras[name],
                    scene.primitives,
                )
            )
            _save_target(scene_dir, name, render)
        missing = [
            path
            for path in (paths.depth, paths.mask, paths.boundary, paths.ids)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{scene_dir.name}/{name} is missing: "
                + ", ".join(path.name for path in missing)
            )
        views[name] = SupervisionView(
            name=name,
            camera=cameras[name],
            depth_path=paths.depth,
            mask_path=paths.mask,
            boundary_path=paths.boundary,
            ids_path=paths.ids,
        )
    return views
