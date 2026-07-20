from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import safetensors.torch

from training.lora import compress_lora_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress a trained Spatial Splat LoRA with per-layer SVD."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()

    source_config = json.loads(args.config.read_text(encoding="utf-8"))
    source_state = safetensors.torch.load_file(str(args.input), device="cpu")
    compressed, report = compress_lora_state(source_state, args.rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "flow_lora.safetensors"
    config_path = args.output_dir / "flow_lora_config.json"
    safetensors.torch.save_file(compressed, str(weights_path))

    source_rank = int(source_config["rank"])
    source_alpha = float(source_config["alpha"])
    target_config = dict(source_config)
    target_config.update(
        rank=args.rank,
        alpha=source_alpha * args.rank / source_rank,
        trainable_parameters=report["target_parameters"],
        compression={
            "method": "per_layer_truncated_svd",
            "source_rank": source_rank,
            "retained_update_energy": report["retained_update_energy"],
            "source_adapter": str(args.input),
        },
    )
    config_path.write_text(
        json.dumps(target_config, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **report,
        "source_scaling": source_alpha / source_rank,
        "target_scaling": float(target_config["alpha"]) / args.rank,
        "weights": str(weights_path),
        "config": str(config_path),
    }
    (args.output_dir / "compression_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
