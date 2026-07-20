import torch
from torch import nn

from training.lora import (
    LoRALinear,
    compress_lora_state,
    inject_lora,
    lora_parameters,
    lora_state_dict,
    set_lora_enabled,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(4, 12)
        self.attn.out = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


def test_lora_linear_preserves_base_output_at_initialization() -> None:
    base = nn.Linear(4, 3)
    inputs = torch.randn(2, 4)
    expected = base(inputs)

    wrapped = LoRALinear(base, rank=2, alpha=2)

    torch.testing.assert_close(wrapped(inputs), expected)


def test_inject_lora_replaces_expected_layers_only() -> None:
    model = _Block()
    model.requires_grad_(False)

    injection = inject_lora(model, rank=2, alpha=2)

    assert len(injection.module_names) == 4
    assert isinstance(model.attn.qkv, LoRALinear)
    assert isinstance(model.mlp.mlp[0], LoRALinear)
    parameters = list(lora_parameters(model))
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert set(lora_state_dict(model)) == {
        f"{name}.{parameter}"
        for name in injection.module_names
        for parameter in ("lora_a", "lora_b")
    }


def test_set_lora_enabled_toggles_adapter_residual() -> None:
    base = nn.Linear(4, 3)
    wrapped = LoRALinear(base, rank=2, alpha=2)
    wrapped.lora_b.data.fill_(0.25)
    model = nn.Sequential(wrapped)
    inputs = torch.randn(2, 4)

    enabled = model(inputs)
    set_lora_enabled(model, False)
    disabled = model(inputs)
    torch.testing.assert_close(disabled, base(inputs))
    assert not torch.equal(enabled, disabled)

    set_lora_enabled(model, True)
    torch.testing.assert_close(model(inputs), enabled)


def test_compress_lora_state_returns_best_lower_rank_update() -> None:
    torch.manual_seed(7)
    lora_a = torch.randn(4, 6)
    lora_b = torch.randn(5, 4)
    state = {
        "block.lora_a": lora_a,
        "block.lora_b": lora_b,
    }

    compressed, report = compress_lora_state(state, target_rank=2)
    actual = compressed["block.lora_b"] @ compressed["block.lora_a"]
    expected_u, expected_s, expected_vh = torch.linalg.svd(lora_b @ lora_a)
    expected = (
        expected_u[:, :2]
        @ torch.diag(expected_s[:2])
        @ expected_vh[:2]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert compressed["block.lora_a"].shape == (2, 6)
    assert compressed["block.lora_b"].shape == (5, 2)
    assert report["source_rank"] == 4
    assert report["target_rank"] == 2
    assert 0.0 < report["retained_update_energy"] <= 1.0
