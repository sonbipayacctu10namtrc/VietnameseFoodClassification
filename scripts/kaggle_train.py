"""Kaggle training script for the Vietnamese-food dataset.

USAGE ON KAGGLE
---------------
1. Create a Kaggle Dataset from the zip of `artifacts/vietnamese_food_clean/`
   (it must contain `images/<class>/*.jpg` and `manifests/{train,validation,test}.csv`).
2. New Notebook -> Add Data -> your dataset. Settings: Accelerator = GPU T4 x2 (or P100),
   Internet = ON (needed once to download pretrained weights).
3. Paste this whole file into one cell, set DATA_ROOT to the mounted path
   (e.g. /kaggle/input/vietnamese-food-clean), and Run. Outputs go to /kaggle/working.

It reads the leakage-aware manifests so the split matches local experiments. If the
manifests are absent it falls back to a stratified per-class random split.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ----------------------- config (edit these on Kaggle) -----------------------
DATA_ROOT = Path("/kaggle/input/vietnamese-food-clean")  # folder with images/ and manifests/
OUTPUT_DIR = Path("/kaggle/working")
MODEL = "efficientnet_b3"      # "resnet50" | "efficientnet_b0" | "efficientnet_b3" | "convnext_tiny"
IMAGE_SIZE = 300               # 224 for resnet/effb0; 300 suits effb3
EPOCHS = 40
FREEZE_EPOCHS = 3
BATCH_SIZE = 32
NUM_WORKERS = 2
PATIENCE = 8                   # early-stop if val top1 does not improve for this many epochs
SEED = 42
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int) -> nn.Module:
    import torchvision.models as m
    if name == "resnet50":
        net = m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        net = m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.IMAGENET1K_V1)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b3":
        net = m.efficientnet_b3(weights=m.EfficientNet_B3_Weights.IMAGENET1K_V1)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        net = m.convnext_tiny(weights=m.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    else:
        raise ValueError(name)
    return net


def classifier_params(model: nn.Module):
    for attr in ("fc", "classifier"):
        if hasattr(model, attr):
            return getattr(model, attr).parameters()
    raise AttributeError("no classifier head found")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    for p in classifier_params(model):
        p.requires_grad = True


def build_transforms(size: int):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm,
        transforms.RandomErasing(p=0.25),
    ])
    evaluation = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(), norm,
    ])
    return train, evaluation


class ManifestDataset(Dataset):
    def __init__(self, data_root: Path, rows: list[dict], transform):
        self.data_root = data_root; self.rows = rows; self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.data_root / row["relative_path"]) as im:
            im = im.convert("RGB")
        return self.transform(im), int(row["class_id"])


def load_splits(data_root: Path):
    """Read manifests if present; else stratified per-class 70/15/15 split of images/."""
    manifests = data_root / "manifests"
    if (manifests / "train.csv").exists():
        def read(name):
            with (manifests / f"{name}.csv").open(encoding="utf-8") as h:
                return list(csv.DictReader(h))
        train, val, test = read("train"), read("validation"), read("test")
        classes = [None] * (max(int(r["class_id"]) for r in train) + 1)
        for r in train:
            classes[int(r["class_id"])] = r["class_name"]
        return train, val, test, classes

    import random
    rng = random.Random(SEED)
    classes = sorted(p.name for p in (data_root / "images").iterdir() if p.is_dir())
    cid = {c: i for i, c in enumerate(classes)}
    train, val, test = [], [], []
    for c in classes:
        files = sorted((data_root / "images" / c).glob("*.jpg")); rng.shuffle(files)
        n = len(files); n_tr = int(n * 0.7); n_va = int(n * 0.15)
        for split, chunk in (("train", files[:n_tr]), ("validation", files[n_tr:n_tr + n_va]), ("test", files[n_tr + n_va:])):
            for f in chunk:
                row = {"relative_path": f"images/{c}/{f.name}", "class_name": c, "class_id": cid[c]}
                {"train": train, "validation": val, "test": test}[split].append(row)
    return train, val, test, classes


def run_epoch(model, loader, criterion, device, optimizer, scaler, num_classes, cm=False):
    training = optimizer is not None
    model.train(training)
    tot = loss_sum = top1 = top5 = 0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long) if cm else None
    for images, targets in loader:
        images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images); loss = criterion(logits, targets)
        if training:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        n = targets.size(0); tot += n; loss_sum += loss.item() * n
        top1 += logits.argmax(1).eq(targets).sum().item()
        top5 += logits.topk(min(5, logits.size(1)), 1).indices.eq(targets[:, None]).any(1).sum().item()
        if conf is not None:
            for t, p in zip(targets.cpu(), logits.argmax(1).cpu()):
                conf[t, p] += 1
    m = {"loss": loss_sum / tot, "top1": top1 / tot, "top5": top5 / tot}
    return (m, conf) if cm else m


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows, classes = load_splits(DATA_ROOT)
    num_classes = len(classes)
    train_tf, eval_tf = build_transforms(IMAGE_SIZE)
    opts = {"num_workers": NUM_WORKERS, "pin_memory": device.type == "cuda"}
    if NUM_WORKERS > 0:
        opts["persistent_workers"] = True
    tl = DataLoader(ManifestDataset(DATA_ROOT, train_rows, train_tf), BATCH_SIZE, shuffle=True, drop_last=True, **opts)
    vl = DataLoader(ManifestDataset(DATA_ROOT, val_rows, eval_tf), BATCH_SIZE, shuffle=False, **opts)
    tsl = DataLoader(ManifestDataset(DATA_ROOT, test_rows, eval_tf), BATCH_SIZE, shuffle=False, **opts)

    counts = Counter(int(r["class_id"]) for r in train_rows)
    weights = torch.tensor([len(train_rows) / (num_classes * max(counts[i], 1)) for i in range(num_classes)],
                           dtype=torch.float32, device=device)
    model = build_model(MODEL, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"device={device} model={MODEL} classes={num_classes} "
          f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    history, best, best_epoch, optimizer, scheduler = [], -1.0, 0, None, None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        frozen = epoch <= FREEZE_EPOCHS
        set_backbone_trainable(model, not frozen)
        if optimizer is None or epoch == FREEZE_EPOCHS + 1:
            lr = 1e-3 if frozen else 3e-4
            optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
            scheduler = None if frozen else CosineAnnealingLR(optimizer, T_max=max(EPOCHS - FREEZE_EPOCHS, 1), eta_min=1e-6)
        tr = run_epoch(model, tl, criterion, device, optimizer, scaler, num_classes)
        va = run_epoch(model, vl, criterion, device, None, scaler, num_classes)
        if scheduler:
            scheduler.step()
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"validation_{k}": v for k, v in va.items()}})
        (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
        print(f"ep{epoch:02d} train_top1={tr['top1']:.3f} val_top1={va['top1']:.3f} val_top5={va['top5']:.3f}")
        if va["top1"] > best:
            best, best_epoch = va["top1"], epoch
            torch.save({"architecture": MODEL, "model_state_dict": model.state_dict(),
                        "classes": classes, "image_size": IMAGE_SIZE, "val_top1": best, "seed": SEED},
                       OUTPUT_DIR / "best.pt")
        elif epoch - best_epoch >= PATIENCE:
            print(f"early stop (no val improvement for {PATIENCE} epochs)"); break

    ckpt = torch.load(OUTPUT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    tm, cm = run_epoch(model, tsl, criterion, device, None, scaler, num_classes, cm=True)
    cmf = cm.float(); rec = torch.diag(cmf) / cmf.sum(1).clamp(min=1)
    prec = torch.diag(cmf) / cmf.sum(0).clamp(min=1); f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    summary = {"model": MODEL, "classes": num_classes, "best_val_top1": best,
               "test_top1": tm["top1"], "test_top5": tm["top5"], "macro_f1": f1.mean().item(),
               "elapsed_minutes": (time.perf_counter() - started) / 60,
               "per_class_test_recall": dict(sorted({classes[i]: rec[i].item() for i in range(num_classes)}.items(),
                                                     key=lambda kv: kv[1]))}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["best_val_top1", "test_top1", "test_top5", "macro_f1", "elapsed_minutes"]}, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        norm = cm.numpy() / cm.numpy().sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(11, 10)); im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(classes, fontsize=6)
        ax.set_title(f"{MODEL} test top1={tm['top1']:.3f}"); fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion.png", dpi=120)
    except Exception as exc:
        print("confusion plot skipped:", exc)


if __name__ == "__main__":
    main()
