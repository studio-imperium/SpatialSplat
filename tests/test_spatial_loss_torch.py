import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.spatial_loss import SpatialLossConfig, spatial_metrics
from training.spatial_loss_torch import spatial_loss_torch


def _fixture(size: int = 128):
    yy, xx = np.mgrid[:size, :size]
    mask = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size * 0.28) ** 2).astype(
        np.float32
    )
    depth = np.zeros((size, size), dtype=np.float32)
    depth[mask > 0] = 2.0 + 0.2 * xx[mask > 0] / size
    return depth, mask


def test_torch_loss_matches_numpy_for_exact_input() -> None:
    depth, mask = _fixture()
    config = SpatialLossConfig(coarse_size=32)
    numpy_result = spatial_metrics(depth, mask, depth, mask, config)
    torch_result = spatial_loss_torch(
        torch.from_numpy(depth),
        torch.from_numpy(mask),
        torch.from_numpy(depth),
        torch.from_numpy(mask),
        config,
    )

    assert torch_result["spatial_loss"].item() == pytest.approx(numpy_result["spatial_loss"], abs=1e-6)
    assert torch_result["soft_iou"].item() == pytest.approx(1.0, abs=1e-6)


def test_torch_loss_backpropagates_to_depth_and_alpha() -> None:
    depth, mask = _fixture()
    predicted_depth = torch.from_numpy(depth + mask * 0.08).requires_grad_(True)
    predicted_alpha = torch.from_numpy(np.clip(mask * 0.8, 0, 1)).requires_grad_(True)
    result = spatial_loss_torch(
        torch.from_numpy(depth),
        torch.from_numpy(mask),
        predicted_depth,
        predicted_alpha,
        SpatialLossConfig(coarse_size=32),
    )
    result["spatial_loss"].backward()

    assert predicted_depth.grad is not None
    assert predicted_alpha.grad is not None
    assert torch.isfinite(predicted_depth.grad).all()
    assert torch.isfinite(predicted_alpha.grad).all()
    assert predicted_depth.grad.abs().sum() > 0
    assert predicted_alpha.grad.abs().sum() > 0
