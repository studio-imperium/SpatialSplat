from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def scene_directories(data_root: Path, names: list[str]) -> list[Path]:
    available = {
        path.parent.name: path.parent
        for path in data_root.glob("*/generated_image.png")
    }
    if names:
        missing = sorted(set(names) - available.keys())
        if missing:
            raise FileNotFoundError(f"unknown scene(s): {', '.join(missing)}")
        return [available[name] for name in names]
    return [available[name] for name in sorted(available)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run latent-only spatial optimization over POC scenes."
    )
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=60)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--final-render-size", type=int, default=512)
    args = parser.parse_args()

    scenes = scene_directories(args.data_root, args.scene)
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
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
