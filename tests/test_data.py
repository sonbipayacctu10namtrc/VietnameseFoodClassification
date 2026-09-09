from pathlib import Path

from food_classifier.data import create_splits, load_classes

DATA_ROOT = Path("food-101")


def test_food101_official_split_produces_balanced_train_validation_and_test() -> None:
    classes = load_classes(DATA_ROOT)
    splits = create_splits(DATA_ROOT, validation_per_class=150, seed=42)

    assert len(classes) == 101
    assert len(splits["train"]) == 60_600
    assert len(splits["validation"]) == 15_150
    assert len(splits["test"]) == 25_250
    assert {sample.relative_path for sample in splits["train"]}.isdisjoint(sample.relative_path for sample in splits["validation"])
    assert {sample.relative_path for sample in splits["train"]}.isdisjoint(sample.relative_path for sample in splits["test"])
    assert {sample.relative_path for sample in splits["validation"]}.isdisjoint(sample.relative_path for sample in splits["test"])
