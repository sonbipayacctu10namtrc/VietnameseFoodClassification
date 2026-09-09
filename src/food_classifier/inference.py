"""Load a trained checkpoint and predict Vietnamese-food classes for an image."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from food_classifier.data import build_transforms
from food_classifier.models import build_model
from food_classifier.vietnamese_dishes import DISHES

_VN_NAME = {dish.canonical_label: dish.vietnamese_name for dish in DISHES}


class FoodPredictor:
    """Wraps a trained model checkpoint for top-k image classification."""

    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes: list[str] = checkpoint["classes"]
        self.image_size: int = checkpoint.get("image_size", 224)
        self.model = build_model(checkpoint["architecture"], len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        _, self.transform = build_transforms(self.image_size)

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict[str, object]]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        k = min(top_k, probabilities.numel())
        scores, indices = probabilities.topk(k)
        return [
            {
                "label": self.classes[index],
                "name": _VN_NAME.get(self.classes[index], self.classes[index]),
                "probability": float(score),
            }
            for score, index in zip(scores.tolist(), indices.tolist())
        ]


@lru_cache(maxsize=4)
def get_predictor(checkpoint_path: str) -> FoodPredictor:
    """Cache a predictor per checkpoint path so the model loads once."""
    return FoodPredictor(Path(checkpoint_path))
