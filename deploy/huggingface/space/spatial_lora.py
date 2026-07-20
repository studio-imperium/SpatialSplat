from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_input = self.dropout(inputs).to(self.lora_a.dtype)
        adapter = F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
        return base_output + adapter.to(base_output.dtype) * self.scaling


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
    rank: int,
    alpha: float,
    dropout: float,
    target_suffixes: Iterable[str],
) -> tuple[str, ...]:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, target_suffixes)
    ]
    if not candidates:
        raise ValueError("no model layers matched the Spatial Splat LoRA config")
    for name, module in candidates:
        _replace_module(model, name, LoRALinear(module, rank, alpha, dropout))
    return tuple(name for name, _ in candidates)


def load_spatial_lora(
    model: nn.Module,
    weights_path: str | Path,
    config_path: str | Path,
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    injected = inject_lora(
        model,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config["dropout"]),
        target_suffixes=config["target_suffixes"],
    )
    expected = tuple(config["module_names"])
    if injected != expected:
        raise ValueError("base TripoSplat layers do not match this adapter")

    load_spatial_lora_weights(model, weights_path, config_path)
    return config


def load_spatial_lora_weights(
    model: nn.Module,
    weights_path: str | Path,
    config_path: str | Path,
) -> dict:
    """Hot-swap a LoRA whose rank is no larger than the injected adapter."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected = tuple(config["module_names"])
    modules = dict(model.named_modules())
    state = safetensors.torch.load_file(str(weights_path), device="cpu")
    for module_name in expected:
        module = modules.get(module_name)
        if not isinstance(module, LoRALinear):
            raise KeyError(f"LoRA module not found: {module_name}")
        source_a = state[f"{module_name}.lora_a"]
        source_b = state[f"{module_name}.lora_b"]
        source_rank = int(source_a.shape[0])
        if source_b.shape[1] != source_rank:
            raise ValueError(f"LoRA rank mismatch: {module_name}")
        if source_rank > module.rank:
            raise ValueError(
                f"source rank {source_rank} exceeds injected rank {module.rank}"
            )
        if source_a.shape[1] != module.lora_a.shape[1]:
            raise ValueError(f"LoRA input width mismatch: {module_name}")
        if source_b.shape[0] != module.lora_b.shape[0]:
            raise ValueError(f"LoRA output width mismatch: {module_name}")

        source_scaling = float(config["alpha"]) / int(config["rank"])
        scaling_ratio = source_scaling / (module.alpha / module.rank)
        module.lora_a.data.zero_()
        module.lora_b.data.zero_()
        module.lora_a.data[:source_rank].copy_(
            source_a.to(device=module.lora_a.device, dtype=module.lora_a.dtype)
        )
        module.lora_b.data[:, :source_rank].copy_(
            source_b.to(device=module.lora_b.device, dtype=module.lora_b.dtype)
            * scaling_ratio
        )
    return config


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.scaling = module.alpha / module.rank if enabled else 0.0
