from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class SpatialLossConfig:
    coarse_size: int = 64
    depth_tolerance: float = 0.02
    alpha_threshold: float = 0.05
    lambda_depth: float = 1.0
    lambda_mask: float = 0.6
    lambda_centroid: float = 0.25
    lambda_extent: float = 0.25


def _area_reduce(values: np.ndarray, size: int) -> np.ndarray:
    height, width = values.shape
    if height % size or width % size:
        raise ValueError(f"input shape {values.shape} must be divisible by coarse size {size}")
    block_h, block_w = height // size, width // size
    return values.reshape(size, block_h, size, block_w).mean(axis=(1, 3))


def _weighted_depth_reduce(depth: np.ndarray, weights: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    coarse_weights = _area_reduce(weights, size)
    weighted_depth = _area_reduce(depth * weights, size)
    coarse_depth = weighted_depth / np.maximum(coarse_weights, _EPS)
    return coarse_depth, coarse_weights


def _moments(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    if total <= _EPS:
        return np.asarray([0.5, 0.5]), np.asarray([0.0, 0.0])
    height, width = weights.shape
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    x_grid, y_grid = np.meshgrid(xs, ys)
    centroid = np.asarray(
        [(weights * x_grid).sum() / total, (weights * y_grid).sum() / total], dtype=np.float64
    )
    variance = np.asarray(
        [
            (weights * (x_grid - centroid[0]) ** 2).sum() / total,
            (weights * (y_grid - centroid[1]) ** 2).sum() / total,
        ],
        dtype=np.float64,
    )
    extent = 2.0 * np.sqrt(np.maximum(variance, 0))
    return centroid, extent


def spatial_metrics(
    target_depth: np.ndarray,
    target_mask: np.ndarray,
    predicted_depth: np.ndarray,
    predicted_alpha: np.ndarray,
    config: SpatialLossConfig = SpatialLossConfig(),
) -> dict[str, float | dict[str, float | int]]:
    arrays = [target_depth, target_mask, predicted_depth, predicted_alpha]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("all depth, mask, and alpha inputs must be 2D")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all depth, mask, and alpha inputs must have the same shape")

    target_depth = target_depth.astype(np.float64, copy=False)
    target_weights = np.clip(target_mask.astype(np.float64, copy=False), 0, 1)
    predicted_depth = predicted_depth.astype(np.float64, copy=False)
    predicted_weights = np.clip(predicted_alpha.astype(np.float64, copy=False), 0, 1)

    target_coarse_depth, target_coarse_mask = _weighted_depth_reduce(
        target_depth, target_weights, config.coarse_size
    )
    predicted_coarse_depth, predicted_coarse_alpha = _weighted_depth_reduce(
        predicted_depth, predicted_weights, config.coarse_size
    )

    target_foreground = target_depth[target_weights > 0.5]
    if target_foreground.size == 0:
        raise ValueError("target mask contains no foreground")
    depth_range = max(float(np.ptp(target_foreground)), 1e-4)
    overlap = (
        (target_coarse_mask > config.alpha_threshold)
        & (predicted_coarse_alpha > config.alpha_threshold)
    )
    if np.any(overlap):
        normalized_depth_error = (
            np.abs(predicted_coarse_depth - target_coarse_depth) / depth_range
        )
        depth_loss = float(
            np.maximum(normalized_depth_error[overlap] - config.depth_tolerance, 0).mean()
        )
        median_depth_error = float(np.median(normalized_depth_error[overlap]))
        p95_depth_error = float(np.percentile(normalized_depth_error[overlap], 95))
    else:
        depth_loss = 1.0
        median_depth_error = 1.0
        p95_depth_error = 1.0

    intersection = float(np.minimum(target_coarse_mask, predicted_coarse_alpha).sum())
    union = float(np.maximum(target_coarse_mask, predicted_coarse_alpha).sum())
    soft_iou = intersection / max(union, _EPS)
    mask_loss = 1.0 - soft_iou

    target_centroid, target_extent = _moments(target_coarse_mask)
    predicted_centroid, predicted_extent = _moments(predicted_coarse_alpha)
    centroid_loss = float(np.linalg.norm(target_centroid - predicted_centroid))
    extent_loss = float(np.abs(target_extent - predicted_extent).mean())

    total = (
        config.lambda_depth * depth_loss
        + config.lambda_mask * mask_loss
        + config.lambda_centroid * centroid_loss
        + config.lambda_extent * extent_loss
    )
    return {
        "spatial_loss": float(total),
        "spatial_score": float(np.exp(-total)),
        "depth_loss": depth_loss,
        "median_normalized_depth_error": median_depth_error,
        "p95_normalized_depth_error": p95_depth_error,
        "mask_loss": float(mask_loss),
        "soft_iou": float(soft_iou),
        "centroid_loss": centroid_loss,
        "extent_loss": extent_loss,
        "config": asdict(config),
    }
