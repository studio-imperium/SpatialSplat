from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from training.generate_diverse_data import build_recipes


def create_split(data_root: Path) -> dict:
    available = {
        path.parent.name for path in data_root.glob("*/generated_image.png")
    }
    recipes = [recipe for recipe in build_recipes() if recipe.scene_id in available]
    missing = sorted(available - {recipe.scene_id for recipe in recipes})
    if missing:
        raise ValueError(f"scenes have no recipe metadata: {', '.join(missing)}")

    by_category: dict[str, list] = defaultdict(list)
    for recipe in recipes:
        by_category[recipe.category].append(recipe)
    if len(recipes) != 50 or any(len(items) != 5 for items in by_category.values()):
        raise ValueError("expected 50 scenes arranged as 10 categories of 5")

    splits = {"train": [], "validation": [], "test": []}
    category_rows = []
    for category_index, (category, items) in enumerate(by_category.items()):
        items = sorted(items, key=lambda item: item.scene_id)
        splits["train"].extend(item.scene_id for item in items[:4])
        holdout = "validation" if category_index % 2 == 0 else "test"
        splits[holdout].append(items[4].scene_id)
        category_rows.append(
            {
                "category": category,
                "train": [item.scene_id for item in items[:4]],
                holdout: items[4].scene_id,
            }
        )

    recipe_by_id = {recipe.scene_id: recipe for recipe in recipes}
    stats = {}
    for name, scene_ids in splits.items():
        stats[name] = {
            "count": len(scene_ids),
            "with_ground": sum(
                recipe_by_id[scene_id].ground_material is not None
                for scene_id in scene_ids
            ),
            "without_ground": sum(
                recipe_by_id[scene_id].ground_material is None
                for scene_id in scene_ids
            ),
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "Four scenes per category train; fifth alternates validation/test by category.",
        "splits": splits,
        "stats": stats,
        "categories": category_rows,
    }


def split_scene_names(path: Path, names: tuple[str, ...] | list[str]) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(names) - set(data["splits"]))
    if unknown:
        raise ValueError(f"unknown split(s): {', '.join(unknown)}")
    return {
        scene
        for name in names
        for scene in data["splits"][name]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the immutable diverse POC split.")
    parser.add_argument("--data-root", type=Path, default=Path("poc_data/diverse_train"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data_root / "split.json"
    split = create_split(args.data_root)
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(split["stats"], indent=2))


if __name__ == "__main__":
    main()
