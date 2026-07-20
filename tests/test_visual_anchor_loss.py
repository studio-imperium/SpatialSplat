from types import SimpleNamespace

import torch

from training.visual_anchor_loss import (
    gaussian_preservation_loss,
    visual_anchor_loss,
)


def test_visual_anchor_loss_is_zero_for_matching_premultiplied_render() -> None:
    rgb = torch.rand(8, 8, 3)
    mask = (torch.rand(8, 8) > 0.25).float()

    loss = visual_anchor_loss(rgb, mask, rgb, mask)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_visual_anchor_loss_penalizes_color_and_alpha_changes() -> None:
    target = torch.ones(8, 8, 3)
    mask = torch.ones(8, 8)

    color_loss = visual_anchor_loss(target, mask, target * 0.5, mask)
    alpha_loss = visual_anchor_loss(target, mask, target, mask * 0.5)

    assert color_loss > 0
    assert alpha_loss > 0


def _gaussian(opacity: float, scale: float, feature: float):
    return SimpleNamespace(
        get_opacity=torch.full((4, 1), opacity),
        get_scaling=torch.full((4, 3), scale),
        _features_dc=torch.full((4, 1, 3), feature),
    )


def test_preservation_loss_detects_density_collapse() -> None:
    reference = _gaussian(0.8, 0.2, 0.4)
    current = _gaussian(0.2, 0.1, 0.1)

    losses = gaussian_preservation_loss(current, reference)

    assert losses["feature_preservation_loss"] > 0
    assert losses["opacity_preservation_loss"] > 0
    assert losses["scale_preservation_loss"] > 0
    assert losses["density_preservation_loss"] > 0
    assert losses["density_ratio"] < 0.1
