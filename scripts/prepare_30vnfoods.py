"""Convert the 30VNFoods Kaggle archive into this project's images/ + manifests/ layout.

The dataset's own Train/Validate/Test split is kept unchanged so results stay comparable
with the paper (77.54% top-1 / 96.07% top-5 on Test). Each row carries sha256 and the same
16x16 average hash the crawler uses, so the data can later be deduplicated against our own
splits. The script also reports near-duplicates that leak across the published splits.

Dataset: https://www.kaggle.com/datasets/quandang/vietnamese-foods (CC BY-NC-SA 4.0)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLITS = {"Train": "train", "Validate": "validation", "Test": "test"}
LICENSE = "CC BY-NC-SA 4.0"
FIELDS = ["relative_path", "class_name", "class_id", "split", "provider", "source_url", "sha256", "perceptual_hash", "license"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/30vnfoods"))
    parser.add_argument("--max-side", type=int, default=512,
                        help="downscale so the longer side is at most this (0 keeps original bytes)")
    parser.add_argument("--near-duplicate-distance", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-zip", action="store_true", help="skip writing <output-dir>.zip for Kaggle upload")
    return parser.parse_args()


def snake(name: str) -> str:
    return "_".join(name.lower().split())


def average_hash(image: Image.Image) -> str:
    """Same 16x16 mean hash as food_classifier.crawler._perceptual_hash."""
    pixels = np.asarray(image.convert("L").resize((16, 16)), dtype=np.float64).ravel()
    return f"{int(''.join('1' if p >= pixels.mean() else '0' for p in pixels), 2):064x}"


_open_archives: dict[str, zipfile.ZipFile] = {}  # one handle per worker process


def convert(job: tuple[str, str, str, int]) -> dict[str, str]:
    archive, member, target, max_side = job
    if archive not in _open_archives:
        _open_archives[archive] = zipfile.ZipFile(archive)
    data = _open_archives[archive].read(member)
    image = Image.open(io.BytesIO(data))
    image.load()
    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        data = buffer.getvalue()
    elif max_side and Image.open(io.BytesIO(data)).mode != "RGB":
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        data = buffer.getvalue()
    Path(target).write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "perceptual_hash": average_hash(image)}


def hash_bits(hashes: list[str]) -> np.ndarray:
    return np.array([[int(h[i:i + 16], 16) for i in range(0, 64, 16)] for h in hashes], dtype=np.uint64)


def cross_split_leaks(rows: list[dict[str, object]], threshold: int) -> dict[str, object]:
    """Count test/validation images with a near-duplicate (any class) in an earlier split."""
    report: dict[str, object] = {}
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    for query, reference in (("validation", ["train"]), ("test", ["train", "validation"])):
        ref_rows = [r for s in reference for r in by_split[s]]
        ref_bits, ref_sha = hash_bits([r["perceptual_hash"] for r in ref_rows]), {r["sha256"] for r in ref_rows}
        exact = near = cross_class = 0
        for row in by_split[query]:
            distances = np.bitwise_count(ref_bits ^ hash_bits([row["perceptual_hash"]])).sum(axis=1)
            hits = np.flatnonzero(distances <= threshold)
            exact += row["sha256"] in ref_sha
            if hits.size:
                near += 1
                cross_class += any(ref_rows[i]["class_name"] != row["class_name"] for i in hits)
        report[f"{query}_vs_{'+'.join(reference)}"] = {
            "images": len(by_split[query]), "exact_duplicates": exact,
            "near_duplicates": near, "near_duplicates_with_other_label": cross_class,
        }
    return report


def main() -> None:
    args = parse_args()
    with zipfile.ZipFile(args.archive) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg") and m.startswith("Images/")]
        license_text = zf.read("LICENSE").decode("utf-8", "replace") if "LICENSE" in zf.namelist() else ""
    source_classes = sorted({m.split("/")[2] for m in members})
    classes = sorted({snake(c) for c in source_classes})
    class_id = {c: i for i, c in enumerate(classes)}

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    rows, jobs = [], []
    for member in sorted(members):
        _, split_dir, source_class, filename = member.split("/", 3)
        name, split = snake(source_class), SPLITS[split_dir]
        relative = f"images/{name}/{split}_{Path(filename).stem}.jpg"
        (args.output_dir / "images" / name).mkdir(parents=True, exist_ok=True)
        rows.append({"relative_path": relative, "class_name": name, "class_id": class_id[name], "split": split,
                     "provider": "30vnfoods", "source_url": f"kaggle:quandang/vietnamese-foods/{member}",
                     "license": LICENSE})
        jobs.append((str(args.archive), member, str(args.output_dir / relative), args.max_side))

    with ProcessPoolExecutor(args.workers) as pool:
        for row, hashes in zip(rows, tqdm(pool.map(convert, jobs, chunksize=64), total=len(jobs), desc="extract")):
            row.update(hashes)

    manifests = args.output_dir / "manifests"
    manifests.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        with (manifests / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: r[k] for k in FIELDS} for r in rows if r["split"] == split)

    counts = Counter((r["class_name"], r["split"]) for r in rows)
    summary = {
        "source": "https://www.kaggle.com/datasets/quandang/vietnamese-foods", "license": LICENSE,
        "paper": "https://ieeexplore.ieee.org/abstract/document/9530774",
        "paper_test_top1": 0.7754, "paper_test_top5": 0.9607,
        "max_side": args.max_side, "classes": len(classes), "images": len(rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "class_counts": {c: {s: counts[(c, s)] for s in ("train", "validation", "test")} for c in classes},
        "near_duplicate_distance": args.near_duplicate_distance,
        "published_split_leakage": cross_split_leaks(rows, args.near_duplicate_distance),
    }
    (manifests / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "LICENSE-30VNFoods.txt").write_text(
        f"30VNFoods — Quan Dang et al. — {LICENSE}\n"
        "https://www.kaggle.com/datasets/quandang/vietnamese-foods\n"
        "Paper: https://ieeexplore.ieee.org/abstract/document/9530774\n\n" + license_text, encoding="utf-8")

    if not args.no_zip:
        archive = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
        print("upload to Kaggle:", archive)
    print(json.dumps({k: summary[k] for k in ("classes", "images", "split_counts", "published_split_leakage")}, indent=2))


if __name__ == "__main__":
    main()
