from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from triposplat import Gaussian


def visual_anchor_loss(
    target_rgb: "torch.Tensor",
    target_mask: "torch.Tensor",
    predicted_rgb: "torch.Tensor",
    predicted_alpha: "torch.Tensor",
) -> "torch.Tensor":
    """Compare premultiplied renders so color and silhouette stay coupled."""
    import torch.nn.functional as F

    target = target_rgb * target_mask[..., None]
    predicted = predicted_rgb * predicted_alpha[..., None]
    full = F.smooth_l1_loss(predicted, target, beta=0.05)
    foreground = target_mask > 0.05
    if bool(foreground.any()):
        full = full + F.smooth_l1_loss(
            predicted[foreground], target[foreground], beta=0.05
        )
    return full * 0.5


def gaussian_preservation_loss(
    current: "Gaussian",
    reference: "Gaussian",
    minimum_density_ratio: float = 0.95,
) -> dict[str, "torch.Tensor"]:
    """Penalize appearance and effective-density collapse at fixed anchors."""
    import torch
    import torch.nn.functional as F

    feature = F.smooth_l1_loss(
        current._features_dc.float(), reference._features_dc.float(), beta=0.05
    )
    opacity = F.smooth_l1_loss(
        current.get_opacity.float(), reference.get_opacity.float(), beta=0.02
    )
    scale = F.smooth_l1_loss(
        current.get_scaling.float().clamp_min(1e-6).log(),
        reference.get_scaling.float().clamp_min(1e-6).log(),
        beta=0.05,
    )
    current_mass = (
        current.get_opacity.float().reshape(-1)
        * current.get_scaling.float().prod(dim=-1)
    ).sum()
    reference_mass = (
        reference.get_opacity.float().reshape(-1)
        * reference.get_scaling.float().prod(dim=-1)
    ).sum().clamp_min(1e-8)
    density_ratio = current_mass / reference_mass
    density = torch.relu(
        torch.as_tensor(
            minimum_density_ratio,
            device=density_ratio.device,
            dtype=density_ratio.dtype,
        )
        - density_ratio
    ).square()
    return {
        "feature_preservation_loss": feature,
        "opacity_preservation_loss": opacity,
        "scale_preservation_loss": scale,
        "density_preservation_loss": density,
        "density_ratio": density_ratio,
    }
