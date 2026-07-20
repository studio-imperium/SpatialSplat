from __future__ import annotations

import numpy as np

from training.structure_metrics import (
    StructureGateConfig,
    aggregate_structure_metrics,
    structure_gate,
    structure_view_metrics,
)


def _scene(size: int = 128):
    depth = np.zeros((size, size), dtype=np.float32)
    ids = np.full((size, size), -1, dtype=np.int32)
    ids[72:112, 8:120] = 0
    depth[72:112, 8:120] = 2.0
    ids[28:72, 18:42] = 1
    depth[28:72, 18:42] = 1.0
    ids[48:72, 76:108] = 2
    depth[48:72, 76:108] = 1.35
    mask = (ids >= 0).astype(np.float32)
    return depth, mask, ids


def _score(depth, mask, ids, predicted_depth, predicted_alpha):
    return structure_view_metrics(
        depth, mask, ids, predicted_depth, predicted_alpha, {0}
    )


def test_exact_structure_passes_gate():
    depth, mask, ids = _scene()
    view = _score(depth, mask, ids, depth, mask)
    aggregate = aggregate_structure_metrics({"isometric": view})
    assert all(structure_gate(aggregate).values())
    assert aggregate["object"]["min_bbox_iou"] == 1.0
    assert aggregate["support"]["worst_flatness_error"] == 0.0


def test_large_floor_cannot_hide_flipped_objects():
    depth, mask, ids = _scene()
    object_mask = ids > 0
    predicted_alpha = mask.copy()
    predicted_alpha[object_mask] = 0
    flipped_objects = object_mask[:, ::-1]
    predicted_alpha[flipped_objects] = 1
    predicted_depth = depth.copy()
    predicted_depth[flipped_objects] = np.flip(depth * object_mask, axis=1)[flipped_objects]
    view = _score(depth, mask, ids, predicted_depth, predicted_alpha)
    aggregate = aggregate_structure_metrics({"isometric": view})
    checks = structure_gate(aggregate)
    assert not checks["object_bbox_iou"] or not checks["object_soft_iou"]
    assert view["whole"]["soft_iou"] > view["object"]["soft_iou"]


def test_shift_and_signal_loss_are_rejected():
    depth, mask, ids = _scene()
    shifted_alpha = np.roll(mask, 22, axis=1)
    shifted_depth = np.roll(depth, 22, axis=1)
    aggregate = aggregate_structure_metrics(
        {"isometric": _score(depth, mask, ids, shifted_depth, shifted_alpha)}
    )
    assert not all(structure_gate(aggregate).values())

    weak = _score(depth, mask, ids, depth, mask * 0.35)
    weak_aggregate = aggregate_structure_metrics({"isometric": weak})
    assert not structure_gate(weak_aggregate)["object_signal"]


def test_warped_floor_is_rejected_even_with_correct_objects():
    depth, mask, ids = _scene()
    predicted_depth = depth.copy()
    support = ids == 0
    x = np.linspace(-0.25, 0.25, depth.shape[1], dtype=np.float32)
    predicted_depth[support] += np.broadcast_to(x, depth.shape)[support]
    aggregate = aggregate_structure_metrics(
        {"isometric": _score(depth, mask, ids, predicted_depth, mask)}
    )
    checks = structure_gate(
        aggregate, config=StructureGateConfig(max_support_flatness_error=0.05)
    )
    assert not checks["support_flatness"]
