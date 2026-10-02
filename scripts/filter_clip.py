"""Filter crawled Vietnamese-food images with CLIP zero-shot scoring.

For every image under ``<images-dir>/<label>/*`` the script computes a CLIP
image embedding and compares it against text prompts built from the dish
catalog (``vietnamese_dishes.py``). Each image gets:

- ``own_prob``  : softmax probability of its own (folder) label,
- ``top1_label``: the highest-scoring catalog label,
- ``top1_prob`` : that label's probability,
- ``rank``      : rank of the own label among all catalog classes (1 = best).

Two things are always written to ``--report-dir``:
- ``clip_scores.csv``  : one row per image (review this to pick a threshold),
- ``clip_summary.json``: kept/dropped counts per class at the current threshold.

Images are only moved/copied when ``--apply`` is given, so you can inspect the
scores first. Originals are never deleted; kept images are copied to
``--output-dir`` (default: ``<images-dir>_clip``), preserving the class layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from food_classifier.vietnamese_dishes import DISHES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT_TEMPLATES = (
    "a photo of {name}, a Vietnamese dish",
    "a close-up food photo of {name}",
    "{english}",
)


def build_prompts() -> dict[str, list[str]]:
    """Map canonical label -> list of natural-language prompts."""
    prompts: dict[str, list[str]] = {}
    for dish in DISHES:
        english = dish.english_queries[0] if dish.english_queries else dish.vietnamese_name
        texts = [t.format(name=dish.vietnamese_name, english=english) for t in PROMPT_TEMPLATES]
        prompts[dish.canonical_label] = texts
    return prompts


def iter_images(images_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((class_dir.name, path))
    return items


@torch.no_grad()
def encode_text(model, tokenizer, prompts: dict[str, list[str]], labels: list[str], device: str) -> torch.Tensor:
    """Return an averaged, normalized text embedding per label (rows aligned to ``labels``)."""
    rows = []
    for label in labels:
        tokens = tokenizer(prompts[label]).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        rows.append(feats.mean(dim=0))
    matrix = torch.stack(rows)
    return matrix / matrix.norm(dim=-1, keepdim=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=Path("artifacts/vietnamese_food_expanded/images"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Where kept images go with --apply (default: <images-dir>_clip)")
    parser.add_argument("--report-dir", type=Path, default=None, help="Where CSV/JSON reports go (default: parent of images-dir)")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--min-prob", type=float, default=0.30, help="Keep image if own-label softmax prob >= this")
    parser.add_argument("--max-rank", type=int, default=3, help="...OR own label is within this rank among all classes")
    parser.add_argument("--restrict-present", action="store_true",
                        help="Score only against labels present as folders (fewer look-alike competitors)")
    parser.add_argument("--outlier-only", action="store_true",
                        help="Conservative mode: DROP only confident-wrong images "
                             "(own_prob < --floor AND top1_prob > --ceil AND top1 != own); keep everything else")
    parser.add_argument("--floor", type=float, default=0.02, help="outlier-only: own_prob below this counts as very low")
    parser.add_argument("--ceil", type=float, default=0.50, help="outlier-only: top1_prob above this counts as confident")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--apply", action="store_true", help="Actually copy kept images to --output-dir (otherwise report only)")
    parser.add_argument("--move", action="store_true", help="With --apply, move instead of copy")
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("open_clip not installed: pip install open_clip_torch") from exc

    images_dir: Path = args.images_dir
    if not images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {images_dir}")
    output_dir = args.output_dir or images_dir.with_name(images_dir.name + "_clip")
    report_dir = args.report_dir or images_dir.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] device={device} model={args.model}/{args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    prompts = build_prompts()
    items = iter_images(images_dir)
    folder_labels = sorted({label for label, _ in items})
    unknown = [l for l in folder_labels if l not in prompts]
    if unknown:
        raise SystemExit(f"No catalog prompt for folders: {', '.join(unknown)}")
    # Score against ALL catalog classes (catch out-of-class photos), or only the
    # labels present as folders when --restrict-present reduces look-alike dilution.
    if args.restrict_present:
        class_labels = [d.canonical_label for d in DISHES if d.canonical_label in set(folder_labels)]
    else:
        class_labels = [d.canonical_label for d in DISHES]
    label_index = {label: i for i, label in enumerate(class_labels)}
    text_features = encode_text(model, tokenizer, prompts, class_labels, device)

    print(f"[clip] scoring {len(items)} images across {len(folder_labels)} folders...")
    rows: list[dict] = []
    batch_imgs: list[torch.Tensor] = []
    batch_meta: list[tuple[str, Path]] = []

    @torch.no_grad()
    def flush() -> None:
        if not batch_imgs:
            return
        pixel = torch.stack(batch_imgs).to(device)
        feats = model.encode_image(pixel)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = (100.0 * feats @ text_features.T).softmax(dim=-1).cpu()
        for (label, path), prob in zip(batch_meta, logits):
            own_i = label_index[label]
            own_prob = float(prob[own_i])
            top1_i = int(prob.argmax())
            order = torch.argsort(prob, descending=True)
            rank = int((order == own_i).nonzero(as_tuple=True)[0]) + 1
            rows.append({
                "label": label,
                "path": str(path),
                "own_prob": round(own_prob, 4),
                "top1_label": class_labels[top1_i],
                "top1_prob": round(float(prob[top1_i]), 4),
                "rank": rank,
            })
        batch_imgs.clear()
        batch_meta.clear()

    for i, (label, path) in enumerate(items, 1):
        try:
            with Image.open(path) as img:
                batch_imgs.append(preprocess(img.convert("RGB")))
            batch_meta.append((label, path))
        except (UnidentifiedImageError, OSError):
            rows.append({"label": label, "path": str(path), "own_prob": 0.0, "top1_label": "", "top1_prob": 0.0, "rank": 9999})
        if len(batch_imgs) >= args.batch_size:
            flush()
        if i % 500 == 0:
            print(f"  {i}/{len(items)}")
    flush()

    def keep(row: dict) -> bool:
        if args.outlier_only:
            confident_wrong = (
                row["own_prob"] < args.floor
                and row["top1_prob"] > args.ceil
                and row["top1_label"] != row["label"]
            )
            return not confident_wrong
        return row["own_prob"] >= args.min_prob or row["rank"] <= args.max_rank

    # Reports
    scores_csv = report_dir / "clip_scores.csv"
    with scores_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "path", "own_prob", "top1_label", "top1_prob", "rank", "kept"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "kept": int(keep(row))})

    kept_by_class: Counter[str] = Counter()
    dropped_by_class: Counter[str] = Counter()
    for row in rows:
        (kept_by_class if keep(row) else dropped_by_class)[row["label"]] += 1

    summary = {
        "images_dir": str(images_dir),
        "model": f"{args.model}/{args.pretrained}",
        "min_prob": args.min_prob,
        "max_rank": args.max_rank,
        "total": len(rows),
        "kept": sum(kept_by_class.values()),
        "dropped": sum(dropped_by_class.values()),
        "per_class": {
            label: {"kept": kept_by_class.get(label, 0), "dropped": dropped_by_class.get(label, 0)}
            for label in folder_labels
        },
    }
    (report_dir / "clip_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[clip] total={summary['total']} kept={summary['kept']} dropped={summary['dropped']}")
    print(f"[clip] scores : {scores_csv}")
    print(f"[clip] summary: {report_dir / 'clip_summary.json'}")

    if args.apply:
        output_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for row in rows:
            if not keep(row):
                continue
            src = Path(row["path"])
            dst = output_dir / row["label"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(src, dst)
            moved += 1
        print(f"[clip] {'moved' if args.move else 'copied'} {moved} kept images -> {output_dir}")
    else:
        print("[clip] report-only run. Review clip_scores.csv, then re-run with --apply.")


if __name__ == "__main__":
    main()
