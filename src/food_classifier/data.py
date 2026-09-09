"""Food-101 split handling and PyTorch datasets."""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class Sample:
    """One Food-101 image and its integer class label."""

    relative_path: str
    class_name: str
    class_id: int
    split: str


def load_classes(data_root: Path) -> list[str]:
    """Return the canonical Food-101 class order from ``classes.txt``."""
    classes_path = data_root / "meta" / "classes.txt"
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_split_paths(data_root: Path, name: str) -> list[str]:
    path = data_root / "meta" / f"{name}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_splits(data_root: Path, *, validation_per_class: int = 150, seed: int = 42) -> dict[str, list[Sample]]:
    """Create train/validation from official train and retain official test unchanged.

    Food-101 contains 750 official-train images per class.  This function samples
    a fixed validation subset independently for each class, preserving balance.
    """
    classes = load_classes(data_root)
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    grouped: dict[str, list[str]] = {class_name: [] for class_name in classes}

    for relative_path in _load_split_paths(data_root, "train"):
        class_name = relative_path.split("/", 1)[0]
        grouped[class_name].append(relative_path)

    splits: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    for class_name in classes:
        paths = grouped[class_name]
        if len(paths) <= validation_per_class:
            raise ValueError(
                f"Class {class_name!r} has {len(paths)} train images; "
                f"cannot reserve {validation_per_class} validation images."
            )
        random.Random(f"{seed}:{class_name}").shuffle(paths)
        class_id = class_to_id[class_name]
        for relative_path in paths[:validation_per_class]:
            splits["validation"].append(Sample(relative_path, class_name, class_id, "validation"))
        for relative_path in paths[validation_per_class:]:
            splits["train"].append(Sample(relative_path, class_name, class_id, "train"))

    for relative_path in _load_split_paths(data_root, "test"):
        class_name = relative_path.split("/", 1)[0]
        splits["test"].append(Sample(relative_path, class_name, class_to_id[class_name], "test"))

    return splits


def write_manifests(splits: dict[str, Iterable[Sample]], output_dir: Path) -> None:
    """Persist split manifests without copying or changing original images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "class_name", "class_id", "split"])
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample.__dict__)


class Food101Dataset(Dataset[tuple[object, int]]):
    """Read Food-101 images from their original locations."""

    def __init__(self, data_root: Path, samples: list[Sample], transform: object) -> None:
        self.images_root = data_root / "images"
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        sample = self.samples[index]
        path = self.images_root / f"{sample.relative_path}.jpg"
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), sample.class_id


def build_transforms(image_size: int = 224) -> tuple[object, object]:
    """Return training and evaluation transforms compatible with ImageNet weights."""
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform
