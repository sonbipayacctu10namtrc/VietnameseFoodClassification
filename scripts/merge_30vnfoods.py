"""Add 30VNFoods images to the TRAIN split of the 50-class Vietnamese-food dataset.

Rules that keep the comparison with the 50-class baseline valid:
- only 30VNFoods Train/Validate images are used; its Test split stays untouched so the
  paper benchmark remains usable;
- only classes that already exist in the 50-class dataset are added;
- candidates that are exact or near duplicates (average-hash distance) of ANY of our images,
  or of each other, are dropped, so nothing leaks into our validation/test;
- our validation/test manifests are copied byte-for-byte.

Images are written at most --max-side pixels (same as the compact Kaggle uploads).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("artifacts/vietnamese_food_expanded"))
    parser.add_argument("--extra", type=Path, default=Path("artifacts/30vnfoods"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vietnamese_food_merged"))
    parser.add_argument("--per-class", type=int, default=300, help="max 30VNFoods images added per class")
    parser.add_argument("--near-duplicate-distance", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def hash_bits(hashes: list[str]) -> np.ndarray:
    return np.array([[int(h[i:i + 16], 16) for i in range(0, 64, 16)] for h in hashes], dtype=np.uint64).reshape(-1, 4)


def min_distances(query: list[str], reference: list[str]) -> np.ndarray:
    ref = hash_bits(reference)
    return np.array([np.bitwise_count(ref ^ q).sum(axis=1).min() if len(ref) else 256 for q in hash_bits(query)])


def copy_image(job: tuple[str, str, int]) -> None:
    source, target, max_side = job
    data = Path(source).read_bytes()
    image = Image.open(io.BytesIO(data))
    image.load()
    if max_side and (max(image.size) > max_side or image.mode != "RGB"):
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        data = buffer.getvalue()
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_bytes(data)


def main() -> None:
    args = parse_args()
    base = {s: read_manifest(args.base / "manifests" / f"{s}.csv") for s in SPLITS}
    extra = [r for s in ("train", "validation") for r in read_manifest(args.extra / "manifests" / f"{s}.csv")]
    class_ids = {r["class_name"]: r["class_id"] for r in base["train"]}
    fields = list(base["train"][0].keys())

    ours = [r for s in SPLITS for r in base[s]]
    our_sha = {r["sha256"] for r in ours}
    our_hashes = [r["perceptual_hash"] for r in ours if r.get("perceptual_hash")]
    candidates = [r for r in extra if r["class_name"] in class_ids]
    dropped = Counter(skipped_class=len(extra) - len(candidates))

    distances = min_distances([r["perceptual_hash"] for r in candidates], our_hashes)
    clean = []
    for row, distance in zip(candidates, distances):
        if row["sha256"] in our_sha:
            dropped["exact_duplicate_of_ours"] += 1
        elif distance <= args.near_duplicate_distance:
            dropped["near_duplicate_of_ours"] += 1
        else:
            clean.append(row)

    rng = random.Random(args.seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clean:
        by_class[row["class_name"]].append(row)
    added: list[dict[str, str]] = []
    for name in sorted(by_class):
        pool = by_class[name][:]
        rng.shuffle(pool)
        kept: list[dict[str, str]] = []
        for row in pool:  # greedy de-dup inside 30VNFoods itself
            if len(kept) == args.per_class:
                break
            if kept and min_distances([row["perceptual_hash"]], [k["perceptual_hash"] for k in kept])[0] <= args.near_duplicate_distance:
                dropped["near_duplicate_within_30vnfoods"] += 1
                continue
            kept.append(row)
        for row in kept:
            added.append({**row, "relative_path": f"images/{name}/30vn_{Path(row['relative_path']).name}",
                          "class_id": class_ids[name], "split": "train", "_source": str(args.extra / row["relative_path"])})

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    jobs = [(str(args.base / r["relative_path"]), str(args.output_dir / r["relative_path"]), args.max_side) for r in ours]
    jobs += [(r["_source"], str(args.output_dir / r["relative_path"]), args.max_side) for r in added]
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(copy_image, jobs, chunksize=64), total=len(jobs), desc="copy"))

    manifests = args.output_dir / "manifests"
    manifests.mkdir(parents=True)
    for split in ("validation", "test"):
        shutil.copyfile(args.base / "manifests" / f"{split}.csv", manifests / f"{split}.csv")
    with (manifests / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(base["train"] + added)

    added_counts = Counter(r["class_name"] for r in added)
    base_counts = Counter(r["class_name"] for r in base["train"])
    summary = {
        "base": str(args.base), "extra": str(args.extra), "per_class_cap": args.per_class,
        "near_duplicate_distance": args.near_duplicate_distance, "max_side": args.max_side, "seed": args.seed,
        "split_counts": {"train": len(base["train"]) + len(added), "validation": len(base["validation"]), "test": len(base["test"])},
        "added_images": len(added), "dropped_candidates": dict(dropped),
        "train_per_class": {c: {"base": base_counts[c], "added": added_counts[c]} for c in sorted(class_ids)},
        "licenses": {"30vnfoods": "CC BY-NC-SA 4.0 (https://www.kaggle.com/datasets/quandang/vietnamese-foods)"},
    }
    (manifests / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if (args.extra / "LICENSE-30VNFoods.txt").exists():
        shutil.copyfile(args.extra / "LICENSE-30VNFoods.txt", args.output_dir / "LICENSE-30VNFoods.txt")
    if not args.no_zip:
        print("upload to Kaggle:", shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir))
    print(json.dumps({k: summary[k] for k in ("split_counts", "added_images", "dropped_candidates")}, indent=2))


if __name__ == "__main__":
    main()
