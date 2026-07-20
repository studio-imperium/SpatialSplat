from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Literal


PrimitiveKind = Literal["box", "sphere", "cylinder"]


@dataclass(frozen=True)
class Primitive:
    name: str
    kind: PrimitiveKind
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[int, int, int]
    yaw_degrees: float = 0.0
    rotation_degrees: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class OrthographicCamera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    ortho_scale: float
    width: int
    height: int


@dataclass(frozen=True)
class PrimitiveScene:
    scene_id: str
    description: str
    camera: OrthographicCamera
    primitives: tuple[Primitive, ...]

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
