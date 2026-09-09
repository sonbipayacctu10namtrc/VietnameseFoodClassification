"""Train a ResNet18 transfer-learning baseline on the crawled Vietnamese-food manifests.

Unlike ``train.py`` (Food-101), this reads the leakage-aware CSV manifests produced by
``prepare_vietnamese_splits.py`` and loads images by their manifest ``relative_path``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from food_classifier.data import build_transforms
from food_classifier.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/vietnamese_food_clean"))
    parser.add_argument("--manifests", type=Path, default=None, help="Defaults to <data-root>/manifests")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vietnamese_resnet18"))
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    """Read images listed in a split manifest by their ``relative_path``."""

    def __init__(self, data_root: Path, rows: list[dict], transform) -> None:
        self.data_root = data_root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.data_root / row["relative_path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["class_id"])


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def class_names(rows: list[dict]) -> list[str]:
    pairs = {int(r["class_id"]): r["class_name"] for r in rows}
    return [pairs[i] for i in range(len(pairs))]


def accuracy_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    top1 = logits.argmax(dim=1).eq(targets).sum().item()
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(targets[:, None]).any(dim=1).sum().item()
    return int(top1), int(top5)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    classifier = model.fc if hasattr(model, "fc") else model.classifier
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def run_epoch(model, loader, criterion, device, *, optimizer, scaler, num_classes, collect_cm=False):
    training = optimizer is not None
    model.train(training)
    total_loss = total = top1 = top5 = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long) if collect_cm else None
    for images, targets in tqdm(loader, desc="train" if training else "eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        count = targets.shape[0]
        b1, b5 = accuracy_counts(logits, targets)
        total += count
        total_loss += loss.detach().item() * count
        top1 += b1
        top5 += b5
        if cm is not None:
            for t, p in zip(targets.cpu(), logits.argmax(dim=1).cpu()):
                cm[t, p] += 1
    metrics = {"loss": total_loss / total, "top1": top1 / total, "top5": top5 / total, "examples": float(total)}
    return (metrics, cm) if collect_cm else metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    manifests = args.manifests or (args.data_root / "manifests")
    train_rows = load_manifest(manifests / "train.csv")
    val_rows = load_manifest(manifests / "validation.csv")
    test_rows = load_manifest(manifests / "test.csv")
    classes = class_names(train_rows)
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()

    train_tf, eval_tf = build_transforms(args.image_size)
    loader_opts = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_opts["persistent_workers"] = True
    train_loader = DataLoader(ManifestDataset(args.data_root, train_rows, train_tf), batch_size=args.batch_size, shuffle=True, **loader_opts)
    val_loader = DataLoader(ManifestDataset(args.data_root, val_rows, eval_tf), batch_size=args.batch_size, shuffle=False, **loader_opts)
    test_loader = DataLoader(ManifestDataset(args.data_root, test_rows, eval_tf), batch_size=args.batch_size, shuffle=False, **loader_opts)

    # Class-weighted loss to offset the imbalance from thin Commons classes.
    counts = Counter(int(r["class_id"]) for r in train_rows)
    weights = torch.tensor([len(train_rows) / (num_classes * counts[i]) for i in range(num_classes)], dtype=torch.float32, device=device)

    model = build_model(args.model, num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict] = []
    best_acc = -1.0
    optimizer = None
    scheduler = None
    started = time.perf_counter()
    print(f"device={device} classes={num_classes} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_trainable(model, trainable=not frozen)
        if optimizer is None or epoch == args.freeze_epochs + 1:
            lr = 1e-3 if frozen else 1e-4
            optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
            scheduler = None if frozen else CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.freeze_epochs, 1), eta_min=1e-6)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | phase={'head' if frozen else 'fine-tune'} | lr={lr:.2e}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, num_classes=num_classes)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, scaler=scaler, num_classes=num_classes)
        if scheduler is not None:
            scheduler.step()
        record = {"epoch": epoch, "lr": lr,
                  **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()},
                  # dashboard-compatible aliases (apps/training_dashboard.py, frontend/)
                  **{f"validation_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.output_dir / "status.json").write_text(json.dumps({
            "state": "running", "started_at": started_at, "current_epoch": epoch,
            "total_epochs": args.epochs, "best_validation_top1": best_acc, "device": str(device),
        }, indent=2), encoding="utf-8")
        print(f"  train top1={train_metrics['top1']:.3f} | val top1={val_metrics['top1']:.3f} top5={val_metrics['top5']:.3f}")
        if val_metrics["top1"] > best_acc:
            best_acc = val_metrics["top1"]
            torch.save({"architecture": args.model, "model_state_dict": model.state_dict(), "classes": classes,
                        "image_size": args.image_size, "val_top1": best_acc, "seed": args.seed}, args.output_dir / "best.pt")

    # Final test evaluation with confusion matrix using the best checkpoint.
    ckpt = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, cm = run_epoch(model, test_loader, criterion, device, optimizer=None, scaler=scaler, num_classes=num_classes, collect_cm=True)
    cm_f = cm.float()
    recall = torch.diag(cm_f) / cm_f.sum(1).clamp(min=1)
    precision = torch.diag(cm_f) / cm_f.sum(0).clamp(min=1)
    f1 = (2 * precision * recall / (precision + recall).clamp(min=1e-9))
    per_class = {classes[i]: recall[i].item() for i in range(num_classes)}
    summary = {
        "device": str(device), "architecture": args.model, "classes": num_classes,
        "best_val_top1": best_acc, "best_validation_top1": best_acc,  # alias for dashboards
        "test_top1": test_metrics["top1"], "test_top5": test_metrics["top5"],
        "elapsed_minutes": (time.perf_counter() - started) / 60, "epochs": args.epochs,
        "started_at": started_at, "completed_at": datetime.now(UTC).isoformat(),
        "per_class_test_recall": dict(sorted(per_class.items(), key=lambda kv: kv[1])),
        "train_class_counts": {classes[i]: counts[i] for i in range(num_classes)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "confusion_matrix.json").write_text(json.dumps({"classes": classes, "matrix": cm.tolist()}, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "status.json").write_text(json.dumps({
        "state": "completed", "started_at": started_at, "completed_at": summary["completed_at"],
        "current_epoch": args.epochs, "total_epochs": args.epochs,
        "best_validation_top1": best_acc, "device": str(device),
    }, indent=2), encoding="utf-8")

    # Dashboard-compatible evaluation artifacts (manifests copy, metrics.json, confusion png).
    import shutil
    dst_manifests = args.output_dir / "manifests"
    if not dst_manifests.exists():
        shutil.copytree(manifests, dst_manifests)
    evaluation = args.output_dir / "evaluation"
    evaluation.mkdir(exist_ok=True)
    (evaluation / "metrics.json").write_text(json.dumps({
        "split": "vietnamese_test", "examples": int(cm.sum().item()),
        "top1_accuracy": test_metrics["top1"], "top5_accuracy": test_metrics["top5"],
        "macro_precision": precision.mean().item(), "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(), "checkpoint": str(args.output_dir / "best.pt"), "classes": classes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_confusion(cm.numpy(), classes, test_metrics, evaluation / "confusion_matrix.png")
    print(json.dumps({k: summary[k] for k in ["best_val_top1", "test_top1", "test_top5", "elapsed_minutes"]}, indent=2))


def _render_confusion(matrix, classes, test_metrics, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    norm = matrix / matrix.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Vietnamese-food test confusion (row-normalized)\ntop1={test_metrics['top1']:.3f} top5={test_metrics['top5']:.3f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
