from __future__ import annotations

from typing import TYPE_CHECKING

from training.spatial_loss import SpatialLossConfig

if TYPE_CHECKING:
    import torch


def _as_batch(values: "torch.Tensor") -> "torch.Tensor":
    if values.ndim == 2:
        return values.unsqueeze(0)
    if values.ndim == 3:
        return values
    raise ValueError("depth, mask, and alpha tensors must have shape HxW or BxHxW")


def _area_reduce(values: "torch.Tensor", size: int) -> "torch.Tensor":
    import torch.nn.functional as F

    return F.adaptive_avg_pool2d(values.unsqueeze(1), (size, size)).squeeze(1)


def _weighted_depth_reduce(
    depth: "torch.Tensor", weights: "torch.Tensor", size: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    coarse_weights = _area_reduce(weights, size)
    weighted_depth = _area_reduce(depth * weights, size)
    coarse_depth = weighted_depth / coarse_weights.clamp_min(1e-8)
    return coarse_depth, coarse_weights


def _moments(weights: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    batch, height, width = weights.shape
    xs = (torch.arange(width, device=weights.device, dtype=weights.dtype) + 0.5) / width
    ys = (torch.arange(height, device=weights.device, dtype=weights.dtype) + 0.5) / height
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    total = weights.sum(dim=(1, 2)).clamp_min(1e-8)
    centroid_x = (weights * x_grid).sum(dim=(1, 2)) / total
    centroid_y = (weights * y_grid).sum(dim=(1, 2)) / total
    centroid = torch.stack([centroid_x, centroid_y], dim=-1)
    variance_x = (weights * (x_grid - centroid_x[:, None, None]).square()).sum(dim=(1, 2)) / total
    variance_y = (weights * (y_grid - centroid_y[:, None, None]).square()).sum(dim=(1, 2)) / total
    extent = 2.0 * torch.sqrt(torch.stack([variance_x, variance_y], dim=-1).clamp_min(0))
    return centroid, extent


def spatial_loss_torch(
    target_depth: "torch.Tensor",
    target_mask: "torch.Tensor",
    predicted_depth: "torch.Tensor",
    predicted_alpha: "torch.Tensor",
    config: SpatialLossConfig = SpatialLossConfig(),
) -> dict[str, "torch.Tensor"]:
    """Differentiable counterpart to `spatial_metrics` for Phase 3 training."""
    import torch

    tensors = [_as_batch(value).float() for value in (
        target_depth,
        target_mask,
        predicted_depth,
        predicted_alpha,
    )]
    target_depth, target_mask, predicted_depth, predicted_alpha = tensors
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("all depth, mask, and alpha tensors must have the same shape")

    target_mask = target_mask.clamp(0, 1)
    predicted_alpha = predicted_alpha.clamp(0, 1)
    target_coarse_depth, target_coarse_mask = _weighted_depth_reduce(
        target_depth, target_mask, config.coarse_size
    )
    predicted_coarse_depth, predicted_coarse_alpha = _weighted_depth_reduce(
        predicted_depth, predicted_alpha, config.coarse_size
    )

    foreground_depth = torch.where(target_mask > 0.5, target_depth, torch.nan)
    depth_min = torch.nan_to_num(foreground_depth, nan=torch.inf).amin(dim=(1, 2))
    depth_max = torch.nan_to_num(foreground_depth, nan=-torch.inf).amax(dim=(1, 2))
    depth_range = (depth_max - depth_min).clamp_min(1e-4)
    normalized_error = (
        (predicted_coarse_depth - target_coarse_depth).abs()
        / depth_range[:, None, None]
    )
    tolerant_error = torch.relu(normalized_error - config.depth_tolerance)
    overlap_weight = target_coarse_mask * predicted_coarse_alpha
    depth_loss_per_item = (
        (tolerant_error * overlap_weight).sum(dim=(1, 2))
        / overlap_weight.sum(dim=(1, 2)).clamp_min(1e-8)
    )

    intersection = torch.minimum(target_coarse_mask, predicted_coarse_alpha).sum(dim=(1, 2))
    union = torch.maximum(target_coarse_mask, predicted_coarse_alpha).sum(dim=(1, 2)).clamp_min(1e-8)
    soft_iou = intersection / union
    mask_loss_per_item = 1.0 - soft_iou

    target_centroid, target_extent = _moments(target_coarse_mask)
    predicted_centroid, predicted_extent = _moments(predicted_coarse_alpha)
    centroid_loss_per_item = torch.linalg.vector_norm(
        target_centroid - predicted_centroid, dim=-1
    )
    extent_loss_per_item = (target_extent - predicted_extent).abs().mean(dim=-1)

    total_per_item = (
        config.lambda_depth * depth_loss_per_item
        + config.lambda_mask * mask_loss_per_item
        + config.lambda_centroid * centroid_loss_per_item
        + config.lambda_extent * extent_loss_per_item
    )
    return {
        "spatial_loss": total_per_item.mean(),
        "depth_loss": depth_loss_per_item.mean(),
        "mask_loss": mask_loss_per_item.mean(),
        "soft_iou": soft_iou.mean(),
        "centroid_loss": centroid_loss_per_item.mean(),
        "extent_loss": extent_loss_per_item.mean(),
    }
