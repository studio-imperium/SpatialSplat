from __future__ import annotations

from typing import TYPE_CHECKING

from training.spatial_loss import SpatialLossConfig
from training.spatial_loss_torch import spatial_loss_torch

if TYPE_CHECKING:
    import torch


def _depth_range(depth: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    import torch

    values = torch.where(mask > 0.5, depth, torch.nan)
    minimum = torch.nan_to_num(values, nan=torch.inf).amin()
    maximum = torch.nan_to_num(values, nan=-torch.inf).amax()
    return (maximum - minimum).clamp_min(1e-4)


def _dilate(mask: "torch.Tensor", fraction: float = 0.06) -> "torch.Tensor":
    import torch.nn.functional as F

    radius = max(1, int(round(max(mask.shape[-2:]) * fraction)))
    return F.max_pool2d(
        mask[None, None], kernel_size=radius * 2 + 1, stride=1, padding=radius
    )[0, 0]


def _coarse_depth(
    depth: "torch.Tensor", weights: "torch.Tensor", size: int = 64
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch.nn.functional as F

    coarse_weights = F.adaptive_avg_pool2d(weights[None, None], (size, size))[0, 0]
    weighted_depth = F.adaptive_avg_pool2d(
        (depth * weights)[None, None], (size, size)
    )[0, 0]
    return weighted_depth / coarse_weights.clamp_min(1e-8), coarse_weights


def _tail_depth_losses(
    target_depth: "torch.Tensor",
    target_mask: "torch.Tensor",
    predicted_depth: "torch.Tensor",
    predicted_alpha: "torch.Tensor",
    depth_range: "torch.Tensor",
    alpha_threshold: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    target_coarse, target_weights = _coarse_depth(target_depth, target_mask)
    predicted_coarse, predicted_weights = _coarse_depth(
        predicted_depth, predicted_alpha
    )
    valid = (target_weights > alpha_threshold) & (
        predicted_weights > alpha_threshold
    )
    if not bool(torch.any(valid)):
        one = predicted_depth.sum() * 0 + 1
        return one, one
    residual = (predicted_coarse - target_coarse)[valid] / depth_range
    absolute = residual.abs()
    count = max(1, int(absolute.numel() * 0.1))
    tail = torch.topk(absolute, count, sorted=False).values.mean()
    centered = (residual - residual.median()).abs()
    flatness_tail = torch.topk(centered, count, sorted=False).values.mean()
    return tail, flatness_tail


def structure_loss_torch(
    target_depth: "torch.Tensor",
    target_mask: "torch.Tensor",
    object_mask: "torch.Tensor",
    support_mask: "torch.Tensor",
    predicted_depth: "torch.Tensor",
    predicted_alpha: "torch.Tensor",
    config: SpatialLossConfig = SpatialLossConfig(),
    lambda_object: float = 1.0,
    lambda_support: float = 0.5,
    lambda_scene: float = 0.15,
) -> dict[str, "torch.Tensor"]:
    import torch

    scene_range = _depth_range(target_depth, target_mask)
    whole = spatial_loss_torch(
        target_depth,
        target_mask,
        predicted_depth,
        predicted_alpha,
        config,
        depth_range_override=scene_range,
    )
    weighted = lambda_scene * whole["spatial_loss"]
    stable_weighted = weighted
    total_weight = lambda_scene

    if bool(torch.any(object_mask > 0.01)):
        object_loss = spatial_loss_torch(
            target_depth,
            object_mask,
            predicted_depth,
            predicted_alpha * _dilate(object_mask) * (1.0 - support_mask),
            config,
            depth_range_override=scene_range,
        )
        object_tail, _ = _tail_depth_losses(
            target_depth,
            object_mask,
            predicted_depth,
            predicted_alpha * _dilate(object_mask) * (1.0 - support_mask),
            scene_range,
            config.alpha_threshold,
        )
        weighted = weighted + lambda_object * (
            object_loss["spatial_loss"] + 0.75 * object_tail
        )
        stable_weighted = stable_weighted + lambda_object * object_loss["spatial_loss"]
        total_weight += lambda_object
    else:
        object_loss = whole
        object_tail = whole["spatial_loss"] * 0

    if bool(torch.any(support_mask > 0.01)):
        support_loss = spatial_loss_torch(
            target_depth,
            support_mask,
            predicted_depth,
            predicted_alpha * support_mask,
            config,
            depth_range_override=scene_range,
        )
        support_tail, support_flatness = _tail_depth_losses(
            target_depth,
            support_mask,
            predicted_depth,
            predicted_alpha * support_mask,
            scene_range,
            config.alpha_threshold,
        )
        weighted = weighted + lambda_support * (
            support_loss["spatial_loss"]
            + support_tail
            + 0.5 * support_flatness
        )
        stable_weighted = stable_weighted + lambda_support * support_loss["spatial_loss"]
        total_weight += lambda_support
    else:
        support_loss = None
        support_tail = whole["spatial_loss"] * 0
        support_flatness = whole["spatial_loss"] * 0

    zero = whole["spatial_loss"] * 0
    return {
        "spatial_loss": weighted / total_weight,
        "stable_spatial_loss": stable_weighted / total_weight,
        "depth_loss": object_loss["depth_loss"],
        "mask_loss": object_loss["mask_loss"],
        "soft_iou": object_loss["soft_iou"],
        "centroid_loss": object_loss["centroid_loss"],
        "extent_loss": object_loss["extent_loss"],
        "object_spatial_loss": object_loss["spatial_loss"],
        "object_tail_depth_loss": object_tail,
        "support_spatial_loss": support_loss["spatial_loss"] if support_loss else zero,
        "support_tail_depth_loss": support_tail,
        "support_flatness_loss": support_flatness,
        "whole_spatial_loss": whole["spatial_loss"],
    }
