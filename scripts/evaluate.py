"""Evaluate a trained Food-101 checkpoint on the untouched official test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from food_classifier.data import Food101Dataset, build_transforms, create_splits
from food_classifier.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("food-101"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes: list[str] = checkpoint["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["architecture"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    splits = create_splits(args.data_root, seed=checkpoint.get("seed", 42))
    test_samples = [sample for sample in splits["test"] if sample.class_id < len(classes)]
    _, evaluation_transform = build_transforms(checkpoint["image_size"])
    dataset = Food101Dataset(args.data_root, test_samples, evaluation_transform)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    top5_correct = 0
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="test"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            top5 = logits.topk(k=min(5, len(classes)), dim=1).indices
            top5_correct += top5.eq(targets.to(device)[:, None]).any(dim=1).sum().item()
            targets_all.append(targets.numpy())
            predictions_all.append(predictions.cpu().numpy())

    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(len(classes)), average="macro", zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(classes)))
    metrics = {
        "split": "official_test",
        "examples": len(targets),
        "top1_accuracy": float((targets == predictions).mean()),
        "top5_accuracy": float(top5_correct / len(targets)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "checkpoint": str(args.checkpoint),
        "classes": classes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(targets, predictions, target_names=classes, zero_division=0)
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(title="ResNet18 — Official Test Confusion Matrix", xlabel="Predicted class", ylabel="True class")
    axis.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(classes)), labels=classes)
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
