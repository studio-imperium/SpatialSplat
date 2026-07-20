import json
from pathlib import Path

from training.build_lora_dataset import REQUIRED_FILES, build_manifest


def _scene(root: Path, name: str, improvement: float) -> None:
    scene = root / name
    scene.mkdir()
    for filename in REQUIRED_FILES:
        (scene / filename).write_bytes(b"placeholder")
    base = {
        "spatial_loss": 0.2,
        "soft_iou": 0.8,
        "median_normalized_depth_error": 0.12,
        "p95_normalized_depth_error": 0.3,
    }
    optimized = {
        "spatial_loss": 0.2 * (1 - improvement),
        "soft_iou": 0.9,
        "median_normalized_depth_error": 0.04,
        "p95_normalized_depth_error": 0.1,
    }
    summary = {
        "fresh_anchors": {
            "base": base,
            "optimized": optimized,
            "relative_improvement": improvement,
            "views": {
                name: {
                    "base": base,
                    "optimized": optimized,
                    "relative_improvement": improvement,
                }
                for name in (
                    "isometric",
                    "top",
                    "left",
                    "right",
                    "front",
                    "back",
                )
            },
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


def test_build_manifest_rejects_high_p95_depth_error(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _scene(data_root, "01_bad_tail", 0.8)
    summary_path = data_root / "01_bad_tail" / "latent_optimization_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["fresh_anchors"]["views"]["back"]["optimized"] = {
        **summary["fresh_anchors"]["optimized"],
        "p95_normalized_depth_error": 0.21,
    }
    summary_path.write_text(json.dumps(summary))

    manifest = build_manifest(data_root, data_root / "manifest.json")

    item = manifest["items"][0]
    assert item["accepted"] is False
    assert item["acceptance_checks"]["p95_depth"] is False


def test_build_manifest_rejects_missing_supervision_view(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _scene(data_root, "01_missing_side", 0.8)
    summary_path = data_root / "01_missing_side" / "latent_optimization_summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["fresh_anchors"]["views"]["back"]
    summary_path.write_text(json.dumps(summary))

    manifest = build_manifest(data_root, data_root / "manifest.json")

    item = manifest["items"][0]
    assert item["accepted"] is False
    assert item["acceptance_checks"]["required_views"] is False


def test_build_manifest_records_nonviable_candidate_without_target(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    scene = data_root / "01_nonviable"
    scene.mkdir(parents=True)
    (scene / "generated_image.png").write_bytes(b"placeholder")
    (scene / "candidate_selection.json").write_text(
        json.dumps({"selected_viable": False})
    )

    manifest = build_manifest(data_root, data_root / "manifest.json")

    item = manifest["items"][0]
    assert item["accepted"] is False
    assert item["acceptance_checks"]["candidate_orientation"] is False
    assert "target_latent.safetensors" in item["missing_files"]
