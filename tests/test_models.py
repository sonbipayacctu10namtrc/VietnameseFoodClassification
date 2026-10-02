import torch

from food_classifier.models import build_model


def test_supported_models_return_expected_number_of_logits() -> None:
    images = torch.zeros((1, 3, 224, 224))
    for architecture in ("resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"):
        model = build_model(architecture, 101, pretrained=False).eval()
        assert model(images).shape == (1, 101)
