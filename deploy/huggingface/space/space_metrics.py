from __future__ import annotations

import html
from typing import Any

import numpy as np

from training.gsplat_depth_renderer import render_decoder_gaussian
from training.multiview import SUPERVISION_VIEWS, supervision_cameras
from training.primitive_renderer import render_scene
from training.scene_schema import OrthographicCamera, Primitive, PrimitiveScene
from training.spatial_loss import SpatialLossConfig
from training.structure_metrics import (
    aggregate_structure_metrics,
    structure_view_metrics,
)


MODE_LABELS = {
    "base": "Base TripoSplat",
    "lora": "Spatial LoRA",
    "low_rank": "Rank-2 Spatial LoRA",
    "control": "Geometry Control",
    "combined": "LoRA + Geometry Control",
}


def _primitive_scene(scene: dict[str, Any], render_size: int) -> PrimitiveScene:
    camera_data = dict(scene["camera"])
    camera_data.update(width=render_size, height=render_size)
    primitives = []
    for item in scene["primitives"]:
        primitives.append(
            Primitive(
                name=str(item.get("name", "primitive")),
                kind=item["kind"],
                center=tuple(float(value) for value in item["center"]),
                size=tuple(float(value) for value in item["size"]),
                color=tuple(int(value) for value in item.get("color", (136, 136, 136))),
                yaw_degrees=float(item.get("yaw_degrees", 0.0)),
                rotation_degrees=(
                    tuple(float(value) for value in item["rotation_degrees"])
                    if item.get("rotation_degrees") is not None
                    else None
                ),
            )
        )
    return PrimitiveScene(
        scene_id=str(scene.get("scene_id", "uploaded_scene")),
        description=str(scene.get("description", "Uploaded primitive scene.")),
        camera=OrthographicCamera(**camera_data),
        primitives=tuple(primitives),
    )


def build_score_targets(
    scene: dict[str, Any], render_size: int = 256
) -> dict[str, Any]:
    if render_size % 64:
        raise ValueError("metric render size must be divisible by 64")
    primitive_scene = _primitive_scene(scene, render_size)
    cameras = supervision_cameras(primitive_scene.camera)
    support_ids = {
        index
        for index, primitive in enumerate(primitive_scene.primitives)
        if primitive.name.lower() in {"ground", "floor", "terrain"}
    }
    targets = {}
    for name in SUPERVISION_VIEWS:
        target_scene = PrimitiveScene(
            primitive_scene.scene_id,
            primitive_scene.description,
            cameras[name],
            primitive_scene.primitives,
        )
        rendered = render_scene(target_scene)
        targets[name] = {
            "camera": cameras[name],
            "depth": np.where(rendered.mask, rendered.depth, 0.0).astype(np.float32),
            "mask": rendered.mask.astype(np.float32),
            "primitive_ids": rendered.primitive_ids,
        }
    return {
        "render_size": render_size,
        "support_ids": support_ids,
        "views": targets,
    }


def score_gaussian(gaussian, targets: dict[str, Any]) -> dict[str, Any]:
    view_metrics = {}
    render_size = int(targets["render_size"])
    config = SpatialLossConfig()
    for name in SUPERVISION_VIEWS:
        target = targets["views"][name]
        predicted = render_decoder_gaussian(
            gaussian, target["camera"], render_size, render_size
        )
        view_metrics[name] = structure_view_metrics(
            target["depth"],
            target["mask"],
            target["primitive_ids"],
            predicted.depth.detach().cpu().numpy(),
            predicted.alpha.detach().cpu().numpy(),
            targets["support_ids"],
            spatial_config=config,
        )
        del predicted
    return {
        "aggregate": aggregate_structure_metrics(view_metrics),
        "views": view_metrics,
    }


def _metric(metrics: dict[str, Any], key: str, fallback: str = "-") -> str:
    value = metrics.get(key)
    return fallback if value is None else f"{float(value):.4f}"


def _mode_row(slug: str, result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    objects = aggregate.get("object") or {}
    support = aggregate.get("support") or {}
    cells = (
        html.escape(MODE_LABELS[slug]),
        f"{100.0 * float(aggregate['structure_score']):.2f}%",
        _metric(aggregate, "structure_loss"),
        _metric(objects, "worst_p95_depth_error"),
        _metric(aggregate, "median_normalized_depth_error"),
        _metric(objects, "min_soft_iou"),
        _metric(objects, "min_bbox_iou"),
        _metric(objects, "max_centroid_error"),
        _metric(objects, "max_extent_error"),
        _metric(support, "planar_worst_p95_depth_error"),
        _metric(support, "worst_flatness_error"),
    )
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def metrics_html(results: dict[str, dict[str, Any]]) -> str:
    ordered = [slug for slug in MODE_LABELS if slug in results]
    if not ordered:
        return "<p>No spatial metrics were produced.</p>"
    headers = (
        "Mode",
        "Spatial rating &uarr;",
        "Structure loss &darr;",
        "Worst P95 depth &darr;",
        "Median depth &darr;",
        "Min soft IoU &uarr;",
        "Min bbox IoU &uarr;",
        "Max centroid error &darr;",
        "Max extent error &darr;",
        "Floor P95 &darr;",
        "Floor flatness &darr;",
    )
    table = (
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%'>"
        "<thead><tr>"
        + "".join(f"<th style='text-align:left;padding:6px;border-bottom:1px solid #999'>{item}</th>" for item in headers)
        + "</tr></thead><tbody>"
        + "".join(_mode_row(slug, results[slug]) for slug in ordered)
        + "</tbody></table></div>"
    )

    detail_headers = "".join(
        f"<th style='text-align:left;padding:5px;border-bottom:1px solid #999'>{item}</th>"
        for item in ("View", "Mode", "Rating", "P95 depth", "Soft IoU", "Centroid", "Extent")
    )
    detail_rows = []
    for view in SUPERVISION_VIEWS:
        for slug in ordered:
            metrics = results[slug]["views"][view]
            primary = metrics.get("object") or metrics["whole"]
            values = (
                view,
                MODE_LABELS[slug],
                f"{100.0 * float(metrics['structure_score']):.2f}%",
                _metric(primary, "p95_normalized_depth_error"),
                _metric(primary, "soft_iou"),
                _metric(primary, "centroid_loss"),
                _metric(primary, "extent_loss"),
            )
            detail_rows.append(
                "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>"
            )
    details = (
        "<details style='margin-top:10px'><summary>Six-view breakdown</summary>"
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;margin-top:8px'>"
        f"<thead><tr>{detail_headers}</tr></thead><tbody>{''.join(detail_rows)}</tbody>"
        "</table></div></details>"
    )
    explanation = (
        "<p><small>Six orthographic views at 256 px. Spatial rating is exp(-structure loss). "
        "Higher is better for ratings and IoU; lower is better for errors. P95 is the worst "
        "normalized depth error among foreground pixels.</small></p>"
    )
    return table + details + explanation
