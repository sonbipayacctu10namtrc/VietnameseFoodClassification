"""Write per-image top-5 predictions of a checkpoint for manifest splits.

Uses the same evaluation transform as scripts/kaggle_train.py (Resize(1.14*size) +
CenterCrop(size)), so the top-1 printed here should reproduce the Kaggle summary.json.

    .venv/Scripts/python scripts/predict_splits.py \
        --checkpoint artifacts/merged50_resnet50/best.pt --data-root artifacts/vietnamese_food_merged
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from food_classifier.models import build_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/vietnamese_food_merged"))
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the checkpoint's folder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class Rows(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], transform: object) -> None:
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.rows[index]["relative_path"]) as image:
            return self.transform(image.convert("RGB")), index


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes: list[str] = checkpoint["classes"]
    size = int(checkpoint.get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["architecture"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    output_dir = args.output_dir or args.checkpoint.parent
    report = {"checkpoint": str(args.checkpoint), "architecture": checkpoint["architecture"], "image_size": size}
    for split in args.splits:
        with (args.data_root / "manifests" / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if any(classes[int(r["class_id"])] != r["class_name"] for r in rows):
            raise ValueError(f"{split}: manifest class ids do not match the checkpoint class list")
        loader = DataLoader(Rows(args.data_root, rows, transform), batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        out, top1 = [], 0
        with torch.inference_mode():
            for images, indices in tqdm(loader, desc=split):
                probs = model(images.to(device)).float().softmax(dim=1).cpu()
                values, ids = probs.topk(5, dim=1)
                for position, index in enumerate(indices.tolist()):
                    row, label = rows[index], int(rows[index]["class_id"])
                    predicted = ids[position, 0].item()
                    top1 += predicted == label
                    out.append({
                        "relative_path": row["relative_path"], "split": split, "label": row["class_name"],
                        "pred": classes[predicted], "pred_prob": round(values[position, 0].item(), 4),
                        "label_prob": round(probs[position, label].item(), 4), "correct": int(predicted == label),
                        "top5": ";".join(f"{classes[c]}:{p:.4f}" for c, p in zip(ids[position].tolist(), values[position].tolist())),
                        "provider": row.get("provider", ""), "source_url": row.get("source_url", ""),
                    })
        path = output_dir / f"predictions_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)
        report[f"{split}_top1"] = top1 / len(rows)
        report[f"{split}_examples"] = len(rows)
        print(f"{split}: top1={top1 / len(rows):.4f} ({top1}/{len(rows)}) -> {path}")
    (output_dir / "predictions_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
