from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import sys

from training.create_data_split import split_scene_names


def scene_directories(
    data_root: Path, names: list[str], allowed: set[str] | None = None
) -> list[Path]:
    available = {
        path.parent.name: path.parent
        for path in data_root.glob("*/generated_image.png")
    }
    if names:
        missing = sorted(set(names) - available.keys())
        if missing:
            raise FileNotFoundError(f"unknown scene(s): {', '.join(missing)}")
        outside = sorted(set(names) - allowed) if allowed is not None else []
        if outside:
            raise ValueError(f"scene(s) outside selected split: {', '.join(outside)}")
        return [available[name] for name in names]
    selected = sorted(available) if allowed is None else sorted(set(available) & allowed)
    return [available[name] for name in selected]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run latent-only spatial optimization over POC scenes."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-viable-candidate", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=60)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--final-render-size", type=int, default=512)
    parser.add_argument(
        "--views",
        nargs="+",
        default=["isometric", "top", "left", "right", "front", "back"],
    )
    args = parser.parse_args()

    allowed = (
        split_scene_names(args.split_file, args.split)
        if args.split_file and args.split
        else None
    )
    scenes = scene_directories(args.data_root, args.scene, allowed)
    if args.require_viable_candidate:
        viable = []
        for scene in scenes:
            path = scene / "candidate_selection.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing candidate selection: {path}")
            selection = json.loads(path.read_text(encoding="utf-8"))
            if selection["selected_viable"]:
                viable.append(scene)
            else:
                print(f"excluding {scene.name}: no viable base candidate", flush=True)
        scenes = viable
    if not scenes:
        raise FileNotFoundError(f"no scenes found under {args.data_root}")

    for index, scene in enumerate(scenes, start=1):
        summary = scene / "latent_optimization_summary.json"
        target = scene / "target_latent.safetensors"
        if summary.is_file() and target.is_file() and not args.force:
            print(f"[{index}/{len(scenes)}] skipping {scene.name}: target exists")
            continue
        print(f"[{index}/{len(scenes)}] optimizing {scene.name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "training.optimize_one_latent",
            "--scene-dir",
            str(scene),
            "--checkpoint-root",
            str(args.checkpoint_root),
            "--optimization-steps",
            str(args.optimization_steps),
            "--render-size",
            str(args.render_size),
            "--final-render-size",
            str(args.final_render_size),
            "--views",
            *args.views,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
