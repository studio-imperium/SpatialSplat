from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from training.spatial_loss import SpatialLossConfig, spatial_metrics


@dataclass(frozen=True)
class StructureMetricConfig:
    roi_dilation_fraction: float = 0.06
    bbox_alpha_threshold: float = 0.1
    lambda_object: float = 1.0
    lambda_support: float = 0.5
    lambda_scene: float = 0.15


@dataclass(frozen=True)
class StructureGateConfig:
    max_object_p95_depth_error: float = 0.2
    max_object_worst_p95_depth_error: float = 0.6
    min_object_soft_iou: float = 0.65
    min_object_bbox_iou: float = 0.49
    max_object_centroid_error: float = 0.1
    max_object_extent_error: float = 0.12
    min_object_signal_ratio: float = 0.7
    min_signal_retention_vs_base: float = 0.9
    max_support_p95_depth_error: float = 0.2
    max_support_worst_p95_depth_error: float = 0.6
    max_support_flatness_error: float = 0.08
    min_support_coverage: float = 0.55


def support_primitive_ids(scene_dir: Path) -> set[int]:
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    support_names = {"ground", "floor", "terrain"}
    return {
        index
        for index, primitive in enumerate(scene["primitives"])
        if primitive["name"].lower() in support_names
    }


def region_masks(
    primitive_ids: np.ndarray, support_ids: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    foreground = primitive_ids >= 0
    support = np.isin(primitive_ids, list(support_ids)) if support_ids else np.zeros_like(foreground)
    objects = foreground & ~support
    return objects.astype(np.float32), support.astype(np.float32)


def resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == shape:
        return labels
    image = Image.fromarray(labels.astype(np.int32), mode="I")
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.int32)


def _dilate(mask: np.ndarray, fraction: float) -> np.ndarray:
    radius = max(1, int(round(max(mask.shape) * fraction)))
    size = radius * 2 + 1
    image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
    return (np.asarray(image.filter(ImageFilter.MaxFilter(size)), dtype=np.float32) / 255.0)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(intersection / max(area_a + area_b - intersection, 1))


def _depth_range(depth: np.ndarray, mask: np.ndarray) -> float:
    values = depth[mask > 0.5]
    if not values.size:
        return 1.0
    return max(float(np.ptp(values)), 1e-4)


def structure_view_metrics(
    target_depth: np.ndarray,
    target_mask: np.ndarray,
    primitive_ids: np.ndarray,
    predicted_depth: np.ndarray,
    predicted_alpha: np.ndarray,
    support_ids: set[int],
    spatial_config: SpatialLossConfig = SpatialLossConfig(),
    config: StructureMetricConfig = StructureMetricConfig(),
) -> dict:
    shape = predicted_depth.shape
    if predicted_alpha.shape != shape:
        raise ValueError("predicted depth and alpha must have the same shape")
    if target_depth.shape != shape:
        target_depth = np.asarray(
            Image.fromarray(target_depth.astype(np.float32), mode="F").resize(
                (shape[1], shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        target_mask = np.asarray(
            Image.fromarray((target_mask * 255).astype(np.uint8), mode="L").resize(
                (shape[1], shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    primitive_ids = resize_labels(primitive_ids, shape)
    object_mask, support_mask = region_masks(primitive_ids, support_ids)
    scene_range = _depth_range(target_depth, target_mask)
    whole = spatial_metrics(
        target_depth,
        target_mask,
        predicted_depth,
        predicted_alpha,
        spatial_config,
        depth_range_override=scene_range,
    )

    result: dict = {"whole": whole, "object": None, "support": None}
    weighted_loss = config.lambda_scene * whole["spatial_loss"]
    weight = config.lambda_scene

    if np.any(object_mask):
        roi = _dilate(object_mask, config.roi_dilation_fraction)
        object_alpha = np.clip(predicted_alpha, 0, 1) * roi * (1.0 - support_mask)
        metrics = spatial_metrics(
            target_depth,
            object_mask,
            predicted_depth,
            object_alpha,
            spatial_config,
            depth_range_override=scene_range,
        )
        target_area = float(object_mask.sum())
        metrics.update(
            {
                "bbox_iou": _bbox_iou(
                    _bbox(object_mask > 0.5),
                    _bbox(object_alpha > config.bbox_alpha_threshold),
                ),
                "signal_ratio": float(object_alpha.sum() / max(target_area, 1e-8)),
                "coverage": float(
                    np.mean(object_alpha[object_mask > 0.5] > spatial_config.alpha_threshold)
                ),
            }
        )
        result["object"] = metrics
        weighted_loss += config.lambda_object * metrics["spatial_loss"]
        weight += config.lambda_object

    if np.any(support_mask):
        support_alpha = np.clip(predicted_alpha, 0, 1) * support_mask
        metrics = spatial_metrics(
            target_depth,
            support_mask,
            predicted_depth,
            support_alpha,
            spatial_config,
            depth_range_override=scene_range,
        )
        overlap = (support_mask > 0.5) & (predicted_alpha > spatial_config.alpha_threshold)
        if np.any(overlap):
            residual = (predicted_depth - target_depth)[overlap] / scene_range
            centered = np.abs(residual - np.median(residual))
            flatness = float(np.percentile(centered, 95))
        else:
            flatness = 1.0
        metrics.update(
            {
                "flatness_error": flatness,
                "coverage": float(np.mean(predicted_alpha[support_mask > 0.5] > spatial_config.alpha_threshold)),
            }
        )
        result["support"] = metrics
        weighted_loss += config.lambda_support * metrics["spatial_loss"]
        weight += config.lambda_support

    result["structure_loss"] = float(weighted_loss / weight)
    result["structure_score"] = float(np.exp(-result["structure_loss"]))
    primary = result["object"] or result["whole"]
    result.update(
        {
            "spatial_loss": result["structure_loss"],
            "spatial_score": result["structure_score"],
            "depth_loss": primary["depth_loss"],
            "median_normalized_depth_error": primary["median_normalized_depth_error"],
            "p95_normalized_depth_error": primary["p95_normalized_depth_error"],
            "mask_loss": primary["mask_loss"],
            "soft_iou": primary["soft_iou"],
            "centroid_loss": primary["centroid_loss"],
            "extent_loss": primary["extent_loss"],
        }
    )
    result["config"] = asdict(config)
    return result


def aggregate_structure_metrics(views: dict[str, dict]) -> dict:
    if not views:
        raise ValueError("at least one structure view is required")
    objects = [item["object"] for item in views.values() if item["object"] is not None]
    supports = [item["support"] for item in views.values() if item["support"] is not None]
    planar_supports = [
        item["support"]
        for name, item in views.items()
        if name in {"isometric", "top"} and item["support"] is not None
    ]
    whole = [item["whole"] for item in views.values()]
    aggregate = {
        "structure_loss": float(np.mean([item["structure_loss"] for item in views.values()])),
        "whole_spatial_loss": float(np.mean([item["spatial_loss"] for item in whole])),
        "object": None,
        "support": None,
    }
    aggregate["structure_score"] = float(np.exp(-aggregate["structure_loss"]))
    if objects:
        aggregate["object"] = {
            "mean_spatial_loss": float(np.mean([item["spatial_loss"] for item in objects])),
            "worst_p95_depth_error": float(max(item["p95_normalized_depth_error"] for item in objects)),
            "median_p95_depth_error": float(
                np.median([item["p95_normalized_depth_error"] for item in objects])
            ),
            "min_soft_iou": float(min(item["soft_iou"] for item in objects)),
            "min_bbox_iou": float(min(item["bbox_iou"] for item in objects)),
            "max_centroid_error": float(max(item["centroid_loss"] for item in objects)),
            "max_extent_error": float(max(item["extent_loss"] for item in objects)),
            "min_signal_ratio": float(min(item["signal_ratio"] for item in objects)),
            "min_coverage": float(min(item["coverage"] for item in objects)),
        }
        aggregate.update(
            {
                "spatial_loss": aggregate["structure_loss"],
                "spatial_score": aggregate["structure_score"],
                "depth_loss": aggregate["object"]["mean_spatial_loss"],
                "p95_normalized_depth_error": aggregate["object"]["worst_p95_depth_error"],
                "median_normalized_depth_error": float(
                    max(item["median_normalized_depth_error"] for item in objects)
                ),
                "soft_iou": aggregate["object"]["min_soft_iou"],
                "mask_loss": float(np.mean([item["mask_loss"] for item in objects])),
                "centroid_loss": aggregate["object"]["max_centroid_error"],
                "extent_loss": aggregate["object"]["max_extent_error"],
            }
        )
    else:
        aggregate.update(
            {
                "spatial_loss": aggregate["structure_loss"],
                "spatial_score": aggregate["structure_score"],
                "depth_loss": float(np.mean([item["depth_loss"] for item in whole])),
                "p95_normalized_depth_error": float(
                    max(item["p95_normalized_depth_error"] for item in whole)
                ),
                "median_normalized_depth_error": float(
                    max(item["median_normalized_depth_error"] for item in whole)
                ),
                "soft_iou": float(min(item["soft_iou"] for item in whole)),
                "mask_loss": float(np.mean([item["mask_loss"] for item in whole])),
                "centroid_loss": float(max(item["centroid_loss"] for item in whole)),
                "extent_loss": float(max(item["extent_loss"] for item in whole)),
            }
        )
    if supports:
        aggregate["support"] = {
            "mean_spatial_loss": float(np.mean([item["spatial_loss"] for item in supports])),
            "worst_p95_depth_error": float(max(item["p95_normalized_depth_error"] for item in supports)),
            "median_p95_depth_error": float(
                np.median([item["p95_normalized_depth_error"] for item in supports])
            ),
            "planar_worst_p95_depth_error": float(
                max(
                    item["p95_normalized_depth_error"]
                    for item in (planar_supports or supports)
                )
            ),
            "worst_flatness_error": float(
                max(item["flatness_error"] for item in (planar_supports or supports))
            ),
            "min_coverage": float(min(item["coverage"] for item in supports)),
        }
    return aggregate


def structure_gate(
    aggregate: dict,
    base_aggregate: dict | None = None,
    config: StructureGateConfig = StructureGateConfig(),
) -> dict[str, bool]:
    objects = aggregate.get("object")
    if objects is None:
        return {"has_visible_objects": False}
    checks = {
        "has_visible_objects": True,
        "object_p95_depth": objects["median_p95_depth_error"] <= config.max_object_p95_depth_error,
        "object_worst_p95_ceiling": objects["worst_p95_depth_error"] <= config.max_object_worst_p95_depth_error,
        "object_soft_iou": objects["min_soft_iou"] >= config.min_object_soft_iou,
        "object_bbox_iou": objects["min_bbox_iou"] >= config.min_object_bbox_iou,
        "object_centroid": objects["max_centroid_error"] <= config.max_object_centroid_error,
        "object_extent": objects["max_extent_error"] <= config.max_object_extent_error,
        "object_signal": objects["min_signal_ratio"] >= config.min_object_signal_ratio,
    }
    support = aggregate.get("support")
    if support is not None:
        checks.update(
            {
                "support_p95_depth": support["planar_worst_p95_depth_error"] <= config.max_support_p95_depth_error,
                "support_worst_p95_ceiling": support["worst_p95_depth_error"] <= config.max_support_worst_p95_depth_error,
                "support_flatness": support["worst_flatness_error"] <= config.max_support_flatness_error,
                "support_coverage": support["min_coverage"] >= config.min_support_coverage,
            }
        )
    if base_aggregate and base_aggregate.get("object"):
        base_signal = base_aggregate["object"]["min_signal_ratio"]
        checks["signal_retained_vs_base"] = (
            objects["min_signal_ratio"]
            >= config.min_signal_retention_vs_base * base_signal
        )
    return checks
