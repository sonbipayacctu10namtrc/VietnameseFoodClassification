"""Model constructors used by training and inference."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


def build_resnet18(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create a ResNet18 classifier for Food-101."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet-B0 classifier for Food-101."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_model(architecture: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Construct a supported Food-101 classifier."""
    builders = {"resnet18": build_resnet18, "efficientnet_b0": build_efficientnet_b0}
    try:
        return builders[architecture](num_classes, pretrained=pretrained)
    except KeyError as error:
        raise ValueError(f"Unsupported architecture: {architecture}") from error
