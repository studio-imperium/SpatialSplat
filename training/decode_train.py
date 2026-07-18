from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from model import OctreeGaussianDecoder
    from triposplat import Gaussian


def freeze_decoder(decoder: "OctreeGaussianDecoder") -> None:
    decoder.eval()
    decoder.requires_grad_(False)


@torch.no_grad()
def sample_fixed_anchors(
    decoder: "OctreeGaussianDecoder",
    latent: torch.Tensor,
    num_gaussians: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    from model import OctreeProbabilityFixedlenDecoder

    num_tokens = max(1, num_gaussians // decoder.gaussians_per_point)
    devices = [latent.device] if latent.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if latent.is_cuda:
            torch.cuda.manual_seed_all(seed)
        sampled = OctreeProbabilityFixedlenDecoder.sample(
            decoder.octree,
            latent.detach(),
            num_points=num_tokens,
            level=decoder._MAX_VOXEL_LEVEL,
            temperature=1.0,
            algo="systematic",
        )
    return {name: value.detach() for name, value in sampled.items()}


def decode_fixed_anchors(
    decoder: "OctreeGaussianDecoder",
    latent: torch.Tensor,
    anchors: dict[str, torch.Tensor],
    activation_checkpoint: bool = False,
) -> "Gaussian":
    from triposplat import _build_gaussians

    if activation_checkpoint and torch.is_grad_enabled() and latent.requires_grad:
        from torch.utils.checkpoint import checkpoint

        features = checkpoint(
            lambda cond: decoder.gs(x=anchors, cond=cond)["features"],
            latent,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        prediction = {"features": features}
    else:
        prediction = decoder.gs(x=anchors, cond=latent)
    return _build_gaussians(decoder.gs, anchors, prediction)[0]
