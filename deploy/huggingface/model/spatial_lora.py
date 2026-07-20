from __future__ import annotations

# The deployment-ready loader is shared verbatim with the Space package.
# Copy this file beside the adapter config and import load_spatial_lora.
from pathlib import Path
import json
import math
from typing import Iterable

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_input = self.dropout(inputs).to(self.lora_a.dtype)
        adapter = F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
        return base_output + adapter.to(base_output.dtype) * self.scaling


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, nn.Sequential):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def load_spatial_lora(model: nn.Module, weights_path: str | Path, config_path: str | Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    suffixes: Iterable[str] = config["target_suffixes"]
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in suffixes)
    ]
    for name, module in candidates:
        _replace_module(
            model,
            name,
            LoRALinear(module, int(config["rank"]), float(config["alpha"]), float(config["dropout"])),
        )
    if tuple(name for name, _ in candidates) != tuple(config["module_names"]):
        raise ValueError("base TripoSplat layers do not match this adapter")
    state = safetensors.torch.load_file(str(weights_path), device="cpu")
    modules = dict(model.named_modules())
    for key, value in state.items():
        module_name, _, parameter_name = key.rpartition(".")
        module = modules[module_name]
        parameter = getattr(module, parameter_name)
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return config
