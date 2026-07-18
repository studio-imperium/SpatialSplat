import json
from pathlib import Path

from training.build_lora_dataset import REQUIRED_FILES, build_manifest


def _scene(root: Path, name: str, improvement: float) -> None:
    scene = root / name
    scene.mkdir()
    for filename in REQUIRED_FILES:
        (scene / filename).write_bytes(b"placeholder")
    summary = {
        "fresh_anchors": {
            "base": {"spatial_loss": 0.2, "soft_iou": 0.8},
            "optimized": {"spatial_loss": 0.2 * (1 - improvement), "soft_iou": 0.9},
            "relative_improvement": improvement,
        }
    }
    (scene / "latent_optimization_summary.json").write_text(json.dumps(summary))


def test_build_manifest_records_acceptance_and_relative_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _scene(data_root, "01_pass", 0.8)
    _scene(data_root, "02_fail", 0.05)
    output = data_root / "lora_dataset.json"

    manifest = build_manifest(data_root, output, min_improvement=0.1)

    assert manifest["num_items"] == 2
    assert manifest["num_accepted"] == 1
    assert manifest["items"][0]["accepted"] is True
    assert manifest["items"][1]["accepted"] is False
    assert manifest["items"][0]["conditioning"].startswith("01_pass/")
