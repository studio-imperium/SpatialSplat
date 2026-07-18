import numpy as np

from training.spatial_loss import SpatialLossConfig, spatial_metrics


def _fixture(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    mask = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size * 0.28) ** 2
    depth = np.zeros((size, size), dtype=np.float32)
    depth[mask] = 2.0 + 0.2 * xx[mask] / size
    return depth, mask.astype(np.float32)


def test_exact_match_has_near_zero_loss() -> None:
    depth, mask = _fixture()
    metrics = spatial_metrics(depth, mask, depth, mask, SpatialLossConfig(coarse_size=32))

    assert metrics["spatial_loss"] < 1e-6
    assert metrics["soft_iou"] > 0.999999


def test_tolerance_ignores_small_depth_detail() -> None:
    depth, mask = _fixture()
    small_offset = depth.copy()
    small_offset[mask > 0] += 0.001
    large_offset = depth.copy()
    large_offset[mask > 0] += 0.08
    config = SpatialLossConfig(coarse_size=32, depth_tolerance=0.02)

    small = spatial_metrics(depth, mask, small_offset, mask, config)
    large = spatial_metrics(depth, mask, large_offset, mask, config)

    assert small["depth_loss"] == 0
    assert large["depth_loss"] > small["depth_loss"]


def test_missing_foreground_is_penalized() -> None:
    depth, mask = _fixture()
    missing_depth = np.zeros_like(depth)
    missing_alpha = np.zeros_like(mask)
    metrics = spatial_metrics(
        depth,
        mask,
        missing_depth,
        missing_alpha,
        SpatialLossConfig(coarse_size=32),
    )

    assert metrics["spatial_loss"] > 1.0
    assert metrics["soft_iou"] == 0
