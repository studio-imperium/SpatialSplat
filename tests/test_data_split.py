from pathlib import Path

from training.create_data_split import create_split


def test_diverse_split_is_40_5_5_and_disjoint():
    split = create_split(Path("poc_data/diverse_train"))
    groups = split["splits"]
    assert {name: len(items) for name, items in groups.items()} == {
        "train": 40,
        "validation": 5,
        "test": 5,
    }
    assert len(set(groups["train"]) | set(groups["validation"]) | set(groups["test"])) == 50
    assert not (set(groups["train"]) & set(groups["validation"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["validation"]) & set(groups["test"]))
