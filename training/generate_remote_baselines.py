from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from gradio_client import Client, handle_file


DEFAULT_SERVER = "http://148.153.245.160:17860"


def _copy_result(source: str | Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source), destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_directories(data_root: Path, names: Sequence[str]) -> list[Path]:
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
        description="Generate POC baseline splats with a remote Gradio server."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--data-root", type=Path, default=Path("poc_data"))
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene directory name to run. Repeat for multiple scenes; default is all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--num-gaussians",
        type=int,
        choices=(32768, 65536, 131072, 262144),
        default=32768,
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scene_dirs = _scene_directories(args.data_root, args.scene)
    if not scene_dirs:
        raise FileNotFoundError(
            f"no generated_image.png files found under {args.data_root}"
        )

    print(f"Connecting to {args.server}")
    client = Client(args.server, verbose=False)

    for index, scene_dir in enumerate(scene_dirs, start=1):
        condition_path = scene_dir / "generated_image.png"
        splat_path = scene_dir / "base_splat.ply"
        metadata_path = scene_dir / "base_metadata.json"
        if splat_path.is_file() and metadata_path.is_file() and not args.force:
            print(f"[{index}/{len(scene_dirs)}] skipping {scene_dir.name}: baseline exists")
            continue

        print(f"[{index}/{len(scene_dirs)}] generating {scene_dir.name}")
        result = client.predict(
            image=handle_file(str(condition_path.resolve())),
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_gaussians=args.num_gaussians,
            output_format="ply",
            preprocess=args.preprocess,
            api_name="/generate",
        )
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError(f"unexpected server result for {scene_dir.name}: {result!r}")

        prepared_file, viewer_ply_file, _download_file, info = result
        _copy_result(prepared_file, scene_dir / "prepared_condition.png")
        _copy_result(viewer_ply_file, splat_path)

        metadata = {
            "source": "remote_gradio",
            "server": args.server,
            "condition": str(condition_path),
            "condition_sha256": _sha256(condition_path),
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_gaussians": args.num_gaussians,
            "preprocess": args.preprocess,
            "output_format": "ply",
            "server_info": info,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{index}/{len(scene_dirs)}] completed {scene_dir.name}: {info}")


if __name__ == "__main__":
    main()
