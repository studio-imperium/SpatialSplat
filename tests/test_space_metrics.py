from __future__ import annotations

import numpy as np

from space_metrics import build_score_targets, metrics_html


def _scene() -> dict:
    return {
        "scene_id": "metrics_test",
        "description": "A cube on a floor.",
        "camera": {
            "position": [1.7, 1.45, 1.7],
            "target": [0.0, -0.08, 0.0],
            "up": [0.0, 1.0, 0.0],
            "ortho_scale": 1.45,
            "width": 512,
            "height": 512,
        },
        "primitives": [
            {
                "name": "ground",
                "kind": "box",
                "center": [0.0, -0.45, 0.0],
                "size": [0.9, 0.1, 0.9],
                "color": [100, 100, 100],
                "yaw_degrees": 0.0,
            },
            {
                "name": "cube",
                "kind": "box",
                "center": [0.0, -0.2, 0.0],
                "size": [0.3, 0.4, 0.3],
                "color": [200, 0, 0],
                "yaw_degrees": 15.0,
            },
        ],
    }


def test_build_score_targets_renders_all_six_views() -> None:
    targets = build_score_targets(_scene(), render_size=64)

    assert set(targets["views"]) == {
        "isometric",
        "top",
        "left",
        "right",
        "front",
        "back",
    }
    assert targets["support_ids"] == {0}
    for target in targets["views"].values():
        assert target["depth"].shape == (64, 64)
        assert np.isfinite(target["depth"]).all()
        assert target["mask"].any()


def _mode_result(offset: float) -> dict:
    primary = {
        "p95_normalized_depth_error": 0.2 + offset,
        "soft_iou": 0.8 - offset,
        "centroid_loss": 0.03 + offset,
        "extent_loss": 0.04 + offset,
    }
    view = {
        "structure_score": 0.9 - offset,
        "object": primary,
        "whole": primary,
    }
    return {
        "aggregate": {
            "structure_score": 0.9 - offset,
            "structure_loss": 0.1 + offset,
            "median_normalized_depth_error": 0.05 + offset,
            "object": {
                "worst_p95_depth_error": 0.2 + offset,
                "min_soft_iou": 0.8 - offset,
                "min_bbox_iou": 0.75 - offset,
                "max_centroid_error": 0.03 + offset,
                "max_extent_error": 0.04 + offset,
            },
            "support": {
                "planar_worst_p95_depth_error": 0.1 + offset,
                "worst_flatness_error": 0.02 + offset,
            },
        },
        "views": {
            name: view
            for name in ("isometric", "top", "left", "right", "front", "back")
        },
    }


def test_metrics_html_contains_modes_and_directional_breakdown() -> None:
    output = metrics_html(
        {
            "base": _mode_result(0.0),
            "low_rank": _mode_result(-0.01),
            "control": _mode_result(-0.02),
        }
    )

    assert "Base TripoSplat" in output
    assert "Rank-2 Spatial LoRA" in output
    assert "Geometry Control" in output
    assert "Spatial rating" in output
    assert "Worst P95 depth" in output
    assert "Six-view breakdown" in output
    assert "isometric" in output
