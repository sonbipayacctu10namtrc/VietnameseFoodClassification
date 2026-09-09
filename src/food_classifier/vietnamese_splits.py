"""Leakage-aware train/validation/test manifests for crawled Vietnamese food."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _groups(rows: list[dict[str, Any]], threshold: int) -> list[list[dict[str, Any]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("source_url") and row.get("provider") != "web":
            by_source[row["source_url"]].append(index)
    for indices in by_source.values():
        for index in indices[1:]:
            union(indices[0], index)

    for left in range(len(rows)):
        left_hash = rows[left].get("perceptual_hash")
        if not left_hash:
            continue
        for right in range(left + 1, len(rows)):
            right_hash = rows[right].get("perceptual_hash")
            if right_hash and _hamming(left_hash, right_hash) <= threshold:
                union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return list(grouped.values())


def _assign(groups: list[list[dict[str, Any]]], ratios: tuple[float, float, float], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    names = ("train", "validation", "test")
    total = sum(len(group) for group in groups)
    targets = {
        "train": round(total * ratios[0]),
        "validation": round(total * ratios[1]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    assigned = {name: [] for name in names}
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    for group in groups:
        destination = max(names, key=lambda name: (targets[name] - len(assigned[name]), -len(assigned[name])))
        assigned[destination].extend(group)
    return assigned


def create_manifests(
    metadata_path: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
    near_duplicate_distance: int = 8,
    min_per_class: int = 30,
) -> dict[str, Any]:
    test_ratio = 1.0 - train_ratio - validation_ratio
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("train/validation/test ratios must all be positive")
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_path = metadata_path.parent / row["image_path"]
        if image_path.is_file():
            by_class[row["canonical_label"]].append(row)

    too_small = {label: len(items) for label, items in by_class.items() if len(items) < min_per_class}
    if too_small:
        details = ", ".join(f"{label}={count}" for label, count in sorted(too_small.items()))
        raise ValueError(f"Classes below min_per_class={min_per_class}: {details}")

    rng = random.Random(seed)
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    classes = sorted(by_class)
    for class_id, label in enumerate(classes):
        assigned = _assign(_groups(by_class[label], near_duplicate_distance), (train_ratio, validation_ratio, test_ratio), rng)
        for split, items in assigned.items():
            for row in items:
                split_rows[split].append({
                    "relative_path": row["image_path"], "class_name": label, "class_id": class_id,
                    "split": split, "provider": row.get("provider", "unknown"),
                    "source_url": row.get("source_url", ""), "sha256": row["sha256"],
                    "perceptual_hash": row.get("perceptual_hash", ""), "license": row.get("license", ""),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["relative_path", "class_name", "class_id", "split", "provider", "source_url", "sha256", "perceptual_hash", "license"]
    for split, items in split_rows.items():
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)

    summary = {
        "metadata": str(metadata_path), "seed": seed,
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "near_duplicate_distance": near_duplicate_distance, "classes": len(classes),
        "images": sum(len(items) for items in split_rows.values()),
        "split_counts": {name: len(items) for name, items in split_rows.items()},
        "class_counts": dict(sorted(Counter(row["canonical_label"] for row in rows).items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
