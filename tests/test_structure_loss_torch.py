from __future__ import annotations

import torch

from training.structure_loss_torch import structure_loss_torch


def test_structure_loss_is_differentiable_and_object_weighted():
    size = 64
    target_depth = torch.zeros(size, size)
    support_mask = torch.zeros(size, size)
    object_mask = torch.zeros(size, size)
    support_mask[40:60, 4:60] = 1
    object_mask[18:40, 10:24] = 1
    target_mask = torch.maximum(support_mask, object_mask)
    target_depth[support_mask.bool()] = 2.0
    target_depth[object_mask.bool()] = 1.0
    predicted_depth = target_depth.clone().requires_grad_(True)
    predicted_alpha = (target_mask * 0.8).clone().requires_grad_(True)

    losses = structure_loss_torch(
        target_depth,
        target_mask,
        object_mask,
        support_mask,
        predicted_depth,
        predicted_alpha,
    )
    losses["spatial_loss"].backward()
    assert torch.isfinite(losses["spatial_loss"])
    assert predicted_alpha.grad is not None
    assert torch.isfinite(losses["stable_spatial_loss"])


def test_tail_loss_penalizes_sparse_large_depth_errors():
    size = 64
    target_depth = torch.ones(size, size)
    object_mask = torch.ones(size, size)
    support_mask = torch.zeros(size, size)
    predicted_depth = target_depth.clone()
    predicted_depth[:8, :8] += 0.5
    predicted_depth.requires_grad_(True)
    predicted_alpha = torch.ones(size, size, requires_grad=True)

    losses = structure_loss_torch(
        target_depth,
        object_mask,
        object_mask,
        support_mask,
        predicted_depth,
        predicted_alpha,
    )

    assert losses["object_tail_depth_loss"] > losses["depth_loss"]
