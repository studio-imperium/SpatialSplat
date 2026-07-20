from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.structure_metrics import aggregate_structure_metrics


def _improvement(before: dict, after: dict) -> float:
    return (before["structure_loss"] - after["structure_loss"]) / max(
        abs(before["structure_loss"]), 1e-8
    )


def refresh(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    training = summary.get("training_resolution_baseline")
    if training:
        training["aggregate"] = aggregate_structure_metrics(training["views"])
    for section_name in ("fixed_anchors", "fresh_anchors"):
        section = summary[section_name]
        base_views = {name: values["base"] for name, values in section["views"].items()}
        optimized_views = {
            name: values["optimized"] for name, values in section["views"].items()
        }
        section["base"] = aggregate_structure_metrics(base_views)
        section["optimized"] = aggregate_structure_metrics(optimized_views)
        section["relative_improvement"] = _improvement(
            section["base"], section["optimized"]
        )
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh derived structure aggregates.")
    parser.add_argument("--data-root", type=Path, default=Path("poc_data/diverse_train"))
    args = parser.parse_args()
    paths = sorted(args.data_root.glob("*/latent_optimization_summary.json"))
    for path in paths:
        refresh(path)
        print(path.parent.name)


if __name__ == "__main__":
    main()
