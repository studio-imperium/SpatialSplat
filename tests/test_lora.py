import torch
from torch import nn

from training.lora import LoRALinear, inject_lora, lora_parameters, lora_state_dict


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
