from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_TARGET_SUFFIXES = (
    "attn.qkv",
    "attn.out",
    "mlp.mlp.0",
    "mlp.mlp.2",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_input = self.dropout(inputs).to(self.lora_a.dtype)
        adapter = F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
        return base_output + adapter.to(base_output.dtype) * self.scaling


@dataclass(frozen=True)
class LoRAInjection:
    module_names: tuple[str, ...]
    trainable_parameters: int


def _matches(name: str, suffixes: Iterable[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, nn.Sequential):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_suffixes: Iterable[str] = DEFAULT_TARGET_SUFFIXES,
) -> LoRAInjection:
    suffixes = tuple(target_suffixes)
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, suffixes)
    ]
    if not candidates:
        raise ValueError(f"no linear modules matched suffixes: {suffixes}")

    for name, module in candidates:
        _replace_module(model, name, LoRALinear(module, rank, alpha, dropout))

    trainable = sum(parameter.numel() for parameter in lora_parameters(model))
    return LoRAInjection(tuple(name for name, _ in candidates), trainable)


def lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous()
            state[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous()
    return state


def save_lora(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(lora_state_dict(model), str(path))


def load_lora(model: nn.Module, path: Path) -> None:
    state = safetensors.torch.load_file(str(path), device="cpu")
    modules = dict(model.named_modules())
    for key, value in state.items():
        module_name, _, parameter_name = key.rpartition(".")
        module = modules.get(module_name)
        if not isinstance(module, LoRALinear):
            raise KeyError(f"LoRA module not found: {module_name}")
        parameter = getattr(module, parameter_name)
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.scaling = module.alpha / module.rank if enabled else 0.0


def compress_lora_state(
    state: dict[str, torch.Tensor], target_rank: int
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    """Return the best per-layer rank-k approximation of a LoRA state."""
    if target_rank <= 0:
        raise ValueError("target_rank must be positive")

    module_names = sorted(
        key[: -len(".lora_a")]
        for key in state
        if key.endswith(".lora_a")
    )
    if not module_names:
        raise ValueError("state does not contain LoRA weights")

    compressed: dict[str, torch.Tensor] = {}
    retained_energy = 0.0
    total_energy = 0.0
    source_rank = None
    for module_name in module_names:
        key_a = f"{module_name}.lora_a"
        key_b = f"{module_name}.lora_b"
        if key_b not in state:
            raise KeyError(f"missing paired LoRA weight: {key_b}")
        lora_a = state[key_a].float()
        lora_b = state[key_b].float()
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(f"LoRA weights must be matrices: {module_name}")
        if lora_a.shape[0] != lora_b.shape[1]:
            raise ValueError(f"LoRA rank mismatch: {module_name}")
        if source_rank is None:
            source_rank = int(lora_a.shape[0])
        elif source_rank != int(lora_a.shape[0]):
            raise ValueError("all LoRA layers must use the same source rank")
        if target_rank > source_rank:
            raise ValueError(
                f"target_rank {target_rank} exceeds source rank {source_rank}"
            )

        # B @ A has rank at most the source LoRA rank. Reduce its SVD to a
        # tiny source_rank x source_rank matrix rather than materializing the
        # full dense layer update.
        q_b, r_b = torch.linalg.qr(lora_b, mode="reduced")
        q_a, r_a = torch.linalg.qr(lora_a.T, mode="reduced")
        u_small, singular_values, vh_small = torch.linalg.svd(
            r_b @ r_a.T, full_matrices=False
        )
        singular_values = singular_values[:target_rank]
        roots = singular_values.sqrt()
        new_b = (q_b @ u_small[:, :target_rank]) * roots.unsqueeze(0)
        new_a = roots.unsqueeze(1) * (vh_small[:target_rank] @ q_a.T)

        compressed[key_a] = new_a.to(state[key_a].dtype).contiguous()
        compressed[key_b] = new_b.to(state[key_b].dtype).contiguous()
        all_energy = torch.linalg.svdvals(r_b @ r_a.T).square()
        total_energy += float(all_energy.sum())
        retained_energy += float(all_energy[:target_rank].sum())

    assert source_rank is not None
    return compressed, {
        "source_rank": source_rank,
        "target_rank": target_rank,
        "retained_update_energy": retained_energy / max(total_energy, 1e-12),
        "source_parameters": sum(tensor.numel() for tensor in state.values()),
        "target_parameters": sum(tensor.numel() for tensor in compressed.values()),
    }
