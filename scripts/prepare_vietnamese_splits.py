"""Create leakage-aware 70/15/15 manifests from crawled Vietnamese-food images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from food_classifier.vietnamese_splits import create_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/vietnamese_food/metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vietnamese_food/manifests"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=8)
    parser.add_argument("--min-per-class", type=int, default=30)
    args = parser.parse_args()
    summary = create_manifests(
        args.metadata, args.output_dir, train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio, seed=args.seed,
        near_duplicate_distance=args.near_duplicate_distance,
        min_per_class=args.min_per_class,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
