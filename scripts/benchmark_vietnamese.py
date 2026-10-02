"""Benchmark Vietnamese-food classifiers against reproducible label baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from food_classifier.data import build_transforms
from food_classifier.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/vietnamese_food_expanded"))
    parser.add_argument("--manifests", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vietnamese_benchmark"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def classes_from_rows(rows: list[dict[str, str]]) -> list[str]:
    labels = {int(row["class_id"]): row["class_name"] for row in rows}
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("class_id values must be contiguous and start at zero.")
    return [labels[index] for index in range(len(labels))]


class ManifestDataset(Dataset):
    def __init__(self, data_root: Path, rows: list[dict[str, str]], transform: object) -> None:
        self.data_root, self.rows, self.transform = data_root, rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[object, int]:
        row = self.rows[index]
        with Image.open(self.data_root / row["relative_path"]) as image:
            return self.transform(image.convert("RGB")), int(row["class_id"])


def metrics(targets: np.ndarray, predictions: np.ndarray, num_classes: int, *, top5: float) -> dict[str, float]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    true_count, predicted_count, diagonal = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    precision = np.divide(diagonal, predicted_count, out=np.zeros(num_classes), where=predicted_count != 0)
    recall = np.divide(diagonal, true_count, out=np.zeros(num_classes), where=true_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes), where=(precision + recall) != 0)
    return {
        "examples": int(len(targets)), "top1_accuracy": float(np.mean(targets == predictions)),
        "top5_accuracy": float(top5), "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def label_baseline(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]], num_classes: int) -> dict[str, object]:
    counts = Counter(int(row["class_id"]) for row in train_rows)
    ranked = sorted(range(num_classes), key=lambda class_id: (-counts[class_id], class_id))
    targets = np.array([int(row["class_id"]) for row in eval_rows])
    result: dict[str, object] = {
        "name": "majority_class", "predicted_class_id": ranked[0],
        "top5_class_ids": ranked[: min(5, num_classes)],
        "train_class_counts": {str(key): counts[key] for key in range(num_classes)},
    }
    result.update(metrics(targets, np.full(len(targets), ranked[0]), num_classes, top5=float(np.isin(targets, ranked[:5]).mean())))
    return result


def evaluate_checkpoint(checkpoint_path: Path, loader: DataLoader, classes: list[str], device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("classes") != classes:
        raise ValueError("Checkpoint class list does not exactly match the benchmark manifest. Train a new model with the same manifests before comparing it.")
    model = build_model(checkpoint["architecture"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    targets_all, predictions_all = [], []
    top5_correct = examples = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=checkpoint_path.parent.name, leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions_all.append(logits.argmax(dim=1).cpu().numpy())
            targets_all.append(targets.numpy())
            top5 = logits.topk(k=min(5, len(classes)), dim=1).indices
            top5_correct += int(top5.eq(targets.to(device)[:, None]).any(dim=1).sum().item())
            examples += len(targets)
    result: dict[str, object] = {
        "name": checkpoint_path.parent.name, "checkpoint": str(checkpoint_path),
        "architecture": checkpoint["architecture"],
        "inference_ms_per_image": 1000 * (time.perf_counter() - started) / examples,
    }
    result.update(metrics(np.concatenate(targets_all), np.concatenate(predictions_all), len(classes), top5=top5_correct / examples))
    return result


def main() -> None:
    args = parse_args()
    manifests = args.manifests or args.data_root / "manifests"
    train_rows, eval_rows = load_manifest(manifests / "train.csv"), load_manifest(manifests / f"{args.split}.csv")
    classes, device = classes_from_rows(train_rows), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_transform = build_transforms()
    options: dict[str, object] = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers:
        options["persistent_workers"] = True
    loader = DataLoader(ManifestDataset(args.data_root, eval_rows, eval_transform), batch_size=args.batch_size, shuffle=False, **options)
    results, failures = [label_baseline(train_rows, eval_rows, len(classes))], []
    for checkpoint in args.checkpoint:
        try:
            results.append(evaluate_checkpoint(checkpoint, loader, classes, device))
        except (KeyError, RuntimeError, ValueError, FileNotFoundError) as error:
            failures.append({"checkpoint": str(checkpoint), "error": str(error)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": str(args.data_root), "split": args.split, "device": str(device), "classes": classes, "results": results, "failures": failures}
    (args.output_dir / f"{args.split}_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["name", "architecture", "examples", "top1_accuracy", "top5_accuracy", "macro_precision", "macro_recall", "macro_f1", "inference_ms_per_image"]
    with (args.output_dir / f"{args.split}_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: result.get(key, "") for key in fields} for result in results)
    console_report = {
        **report,
        "results": [
            {key: value for key, value in result.items() if key != "confusion_matrix"}
            for result in results
        ],
    }
    print(json.dumps(console_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
