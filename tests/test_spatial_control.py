from __future__ import annotations

import torch
import sys
import types

from spatial_control import (
    CONTROL_FEATURE_NAMES,
    attach_spatial_control,
    canonicalize_scene,
    scene_control_tensor,
    sobol_world_positions,
)


def _scene() -> dict:
    return {
        "scene_id": "control_test",
        "description": "A box and sphere on a floor.",
        "primitives": [
            {
                "name": "ground",
                "kind": "box",
                "center": [0.0, -0.45, 0.0],
                "size": [0.9, 0.1, 0.9],
                "color": [100, 100, 100],
                "yaw_degrees": 0.0,
            },
            {
                "name": "object",
                "kind": "sphere",
                "center": [0.0, -0.2, 0.0],
                "size": [0.3, 0.3, 0.3],
                "color": [200, 0, 0],
                "yaw_degrees": 0.0,
            },
        ],
    }


def _tiny_flow_model():
    try:
        import torchvision  # noqa: F401
    except ImportError:
        torchvision = types.ModuleType("torchvision")
        ops = types.ModuleType("torchvision.ops")

        def unavailable_deform_conv2d(*args, **kwargs):
            raise RuntimeError("deform_conv2d is unavailable in this unit test")

        ops.deform_conv2d = unavailable_deform_conv2d
        torchvision.ops = ops
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.ops"] = ops
    from model import LatentSeqMMFlowModel

    return LatentSeqMMFlowModel(
        q_token_length=16,
        in_channels=4,
        model_channels=32,
        cond_channels=8,
        out_channels=4,
        num_blocks=2,
        num_refiner_blocks=2,
        num_heads=4,
        num_head_channels=8,
        cam_channels=5,
        cond2_channels=None,
        mlp_ratio=2,
        share_mod=True,
        qk_rms_norm=False,
        use_shift_table=False,
    )


def test_scene_control_tensor_matches_vecseq_shape_and_schema() -> None:
    control = scene_control_tensor(_scene(), token_count=128)

    assert control.shape == (1, 128, len(CONTROL_FEATURE_NAMES))
    assert torch.isfinite(control).all()
    assert torch.equal(control[..., -1], torch.ones_like(control[..., -1]))
    assert set(torch.unique(control[..., 10]).tolist()).issubset({0.0, 1.0})


def test_worldsketch_scene_is_normalized_to_control_schema() -> None:
    source = {
        "version": 4,
        "ground": {
            "size": 144,
            "complete": True,
            "strokes": [
                {
                    "mode": "paint",
                    "color": "#587553",
                    "radius": 0.5,
                    "points": [[-3.0, -2.0], [4.0, 3.0]],
                }
            ],
        },
        "primitives": [
            {
                "type": "box",
                "position": [0.0, 2.0, 0.0],
                "rotation": [0.2, 0.5, -0.1],
                "scale": [2.0, 4.0, 1.0],
                "color": "#884422",
            },
            {
                "type": "box",
                "position": [3.0, 1.0, -1.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.0, 2.0, 1.0],
                "color": "#448822",
            },
        ],
    }

    normalized = canonicalize_scene(source)

    assert normalized["source_schema"] == "worldsketch"
    assert normalized["camera"]["target"] == [0.0, -0.08, 0.0]
    assert normalized["normalization"]["uniform_scale"] > 0
    assert normalized["primitives"][0]["name"] == "ground"
    assert normalized["primitives"][1]["kind"] == "box"
    assert len(normalized["primitives"][1]["rotation_degrees"]) == 3
    assert max(normalized["primitives"][1]["size"]) < 1.45
    control = scene_control_tensor(source, token_count=128)
    assert control.shape == (1, 128, len(CONTROL_FEATURE_NAMES))
    assert torch.isfinite(control).all()


def test_canonical_scene_is_not_renormalized() -> None:
    scene = _scene()
    assert canonicalize_scene(scene) is scene


def test_worldsketch_schema_error_names_the_bad_field() -> None:
    source = {"primitives": [{"type": "box", "scale": [1, 1, 1]}]}

    try:
        canonicalize_scene(source)
    except ValueError as error:
        assert "primitive 1 position" in str(error)
    else:
        raise AssertionError("expected an invalid WorldSketch primitive to fail")


def test_sobol_world_positions_cover_calibrated_model_bounds() -> None:
    positions = sobol_world_positions(4096)

    assert positions[:, 0].min() >= -0.49
    assert positions[:, 0].max() <= 0.49
    assert positions[:, 1].min() >= -0.74
    assert positions[:, 1].max() <= 0.24
    assert positions[:, 2].min() >= -0.49
    assert positions[:, 2].max() <= 0.49


def test_sobol_world_positions_match_viewer_axis_transform() -> None:
    raw = torch.quasirandom.SobolEngine(
        dimension=3, scramble=True, seed=123
    ).draw(32) - 0.5
    positions = sobol_world_positions(32)

    expected = torch.stack(
        (raw[:, 2] * 0.96, -raw[:, 1] * 0.96 - 0.25, raw[:, 0] * 0.96),
        dim=-1,
    )
    assert torch.allclose(positions, expected)


def test_zero_initialized_adapter_preserves_base_flow_output() -> None:
    torch.manual_seed(7)
    model = _tiny_flow_model().eval()
    inputs = {
        "latent": torch.randn(1, 16, 4),
        "camera": torch.randn(1, 1, 5),
    }
    condition = {"feature1": torch.randn(1, 9, 8)}
    timestep = torch.tensor([500.0])
    base = model(inputs, timestep, condition)

    attach_spatial_control(
        model, hidden_channels=8, layer_indices=(0, 1), device="cpu"
    )
    control = scene_control_tensor(_scene(), token_count=16)
    controlled = model(inputs, timestep, condition, control=control)

    assert torch.equal(base["latent"], controlled["latent"])
    assert torch.equal(base["camera"], controlled["camera"])


def test_control_requires_an_attached_adapter() -> None:
    model = _tiny_flow_model().eval()
    inputs = {
        "latent": torch.randn(1, 16, 4),
        "camera": torch.randn(1, 1, 5),
    }
    condition = {"feature1": torch.randn(1, 9, 8)}
    control = scene_control_tensor(_scene(), token_count=16)

    try:
        model(inputs, torch.tensor([500.0]), condition, control=control)
    except ValueError as error:
        assert "without a spatial control adapter" in str(error)
    else:
        raise AssertionError("expected control without an adapter to fail")


def test_frozen_backbone_passes_gradients_to_control_output() -> None:
    torch.manual_seed(11)
    model = _tiny_flow_model()
    model.requires_grad_(False)
    adapter = attach_spatial_control(
        model, hidden_channels=8, layer_indices=(0, 1), device="cpu"
    )
    inputs = {
        "latent": torch.randn(1, 16, 4),
        "camera": torch.randn(1, 1, 5),
    }
    condition = {"feature1": torch.randn(1, 9, 8)}
    control = scene_control_tensor(_scene(), token_count=16)

    output = model(inputs, torch.tensor([500.0]), condition, control=control)
    output["latent"].square().mean().backward()

    assert any(
        injection.output.weight.grad is not None
        and torch.count_nonzero(injection.output.weight.grad) > 0
        for injection in adapter.injections.values()
    )
    adapter_parameter_ids = {id(parameter) for parameter in adapter.parameters()}
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in adapter_parameter_ids
    )
