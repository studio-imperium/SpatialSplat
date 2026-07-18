from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.lora import inject_lora, load_lora
from training.train_flow_lora import _checkpoint, _load_example, _probe_loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a LoRA checkpoint on the trainer's fixed flow-matching probe."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["accepted"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))

    from triposplat import load_flow_model

    model = load_flow_model(
        _checkpoint(
            args.checkpoint_root.resolve(),
            config["base_checkpoint"],
        ),
        device=device,
        dtype=torch.float16,
    )
    model.requires_grad_(False)
    inject_lora(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config["dropout"],
        target_suffixes=config["target_suffixes"],
    )
    load_lora(model, args.lora)
    model.eval()
    examples = [_load_example(manifest_path, item, device) for item in items]
    loss = _probe_loss(model, examples, device, args.seed, args.camera_weight)
    print(json.dumps({"num_items": len(items), "fixed_probe_loss": loss}, indent=2))


if __name__ == "__main__":
    main()
