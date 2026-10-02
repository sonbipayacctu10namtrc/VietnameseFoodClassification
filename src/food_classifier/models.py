"""Model constructors used by training and inference."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    efficientnet_b3,
    resnet18,
    resnet50,
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


def build_resnet50(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create a ResNet50 classifier (head layout matches scripts/kaggle_train.py)."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b3(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet-B3 classifier (head layout matches scripts/kaggle_train.py)."""
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_convnext_tiny(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create a ConvNeXt-Tiny classifier (head layout matches scripts/kaggle_train.py)."""
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def build_model(architecture: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Construct a supported Food-101 classifier."""
    builders = {
        "resnet18": build_resnet18, "resnet50": build_resnet50, "efficientnet_b0": build_efficientnet_b0,
        "efficientnet_b3": build_efficientnet_b3, "convnext_tiny": build_convnext_tiny,
    }
    try:
        return builders[architecture](num_classes, pretrained=pretrained)
    except KeyError as error:
        raise ValueError(f"Unsupported architecture: {architecture}") from error
