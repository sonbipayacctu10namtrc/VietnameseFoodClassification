"""Train a reproducible ResNet18 baseline on Food-101."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from food_classifier.data import (
    Food101Dataset,
    build_transforms,
    create_splits,
    load_classes,
    write_manifests,
)
from food_classifier.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("food-101"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/resnet18_baseline"))
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-per-class", type=int, default=150)
    parser.add_argument(
        "--class-limit",
        type=int,
        default=None,
        help="Train only the first N canonical Food-101 classes for a fast experiment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--smoke-test", action="store_true", help="Train only 10 batches and validate 5 batches.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    top1 = logits.argmax(dim=1).eq(targets).sum().item()
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(targets[:, None]).any(dim=1).sum().item()
    return int(top1), int(top5)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_examples = top1 = top5 = 0
    progress = tqdm(loader, desc="train" if training else "validation", leave=False)

    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
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
        batch_top1, batch_top5 = accuracy_counts(logits, targets)
        total_examples += count
        total_loss += loss.detach().item() * count
        top1 += batch_top1
        top5 += batch_top5
        progress.set_postfix(loss=f"{total_loss / total_examples:.3f}", top1=f"{top1 / total_examples:.3f}")

    return {
        "loss": total_loss / total_examples,
        "top1": top1 / total_examples,
        "top5": top5 / total_examples,
        "examples": float(total_examples),
    }


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    classifier = model.fc if hasattr(model, "fc") else model.classifier
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def write_json(path: Path, value: object) -> None:
    """Persist a small run artifact for the dashboard."""
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.data_root.exists():
        raise FileNotFoundError(f"Food-101 dataset not found: {args.data_root}")
    if args.epochs < 1 or not 0 <= args.freeze_epochs < args.epochs:
        raise ValueError("epochs must be >= 1 and freeze-epochs must be in [0, epochs).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.json"
    status_path = args.output_dir / "status.json"
    started_at = datetime.now(UTC).isoformat()
    write_json(
        status_path,
        {
            "state": "running",
            "started_at": started_at,
            "current_epoch": 0,
            "total_epochs": args.epochs,
            "device": str(device),
        },
    )

    splits = create_splits(args.data_root, validation_per_class=args.validation_per_class, seed=args.seed)
    classes = load_classes(args.data_root)
    if args.class_limit is not None:
        if not 2 <= args.class_limit <= len(classes):
            raise ValueError(f"class-limit must be between 2 and {len(classes)}.")
        classes = classes[: args.class_limit]
        splits = {
            split_name: [sample for sample in samples if sample.class_id < args.class_limit]
            for split_name, samples in splits.items()
        }
    write_manifests(splits, args.output_dir / "manifests")
    train_transform, evaluation_transform = build_transforms(args.image_size)
    train_dataset = Food101Dataset(args.data_root, splits["train"], train_transform)
    validation_dataset = Food101Dataset(args.data_root, splits["validation"], evaluation_transform)
    loader_options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = build_model(args.model, len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    started = time.perf_counter()
    optimizer: AdamW | None = None
    scheduler: CosineAnnealingLR | None = None

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_trainable(model, trainable=not frozen)
        initial_learning_rate = 1e-3 if frozen else 1e-4
        if optimizer is None or epoch == args.freeze_epochs + 1:
            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=initial_learning_rate,
                weight_decay=1e-4,
            )
            scheduler = (
                None
                if frozen
                else CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.freeze_epochs, 1), eta_min=1e-6)
            )
        learning_rate = optimizer.param_groups[0]["lr"]
        max_train_batches = 10 if args.smoke_test else None
        max_validation_batches = 5 if args.smoke_test else None
        print(f"Epoch {epoch}/{args.epochs} | phase={'head' if frozen else 'fine-tune'} | lr={learning_rate}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=max_train_batches,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            max_batches=max_validation_batches,
        )
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": float(epoch),
            "learning_rate": learning_rate,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        write_json(history_path, history)
        write_json(
            status_path,
            {
                "state": "running",
                "started_at": started_at,
                "updated_at": datetime.now(UTC).isoformat(),
                "current_epoch": epoch,
                "total_epochs": args.epochs,
                "best_validation_top1": best_accuracy,
                "device": str(device),
            },
        )
        print(json.dumps(record, indent=2))

        if validation_metrics["top1"] > best_accuracy:
            best_accuracy = validation_metrics["top1"]
            torch.save(
                {
                    "architecture": args.model,
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                    "validation_top1": best_accuracy,
                    "seed": args.seed,
                },
                args.output_dir / "best.pt",
            )

    summary = {
        "device": str(device),
        "best_validation_top1": best_accuracy,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "epochs": args.epochs,
        "smoke_test": args.smoke_test,
        "architecture": args.model,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        status_path,
        {
            "state": "completed",
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "current_epoch": args.epochs,
            "total_epochs": args.epochs,
            "best_validation_top1": best_accuracy,
            "device": str(device),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
