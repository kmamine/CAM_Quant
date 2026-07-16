"""Model registry, loading, and CAM target-layer selection."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models

# Supported architectures: registry name -> torchvision constructor.
MODEL_REGISTRY = {
    "vgg16": models.vgg16,
    "resnet50": models.resnet50,
    "densenet121": models.densenet121,
    "mobilenet_v2": models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
    "squeezenet1_1": models.squeezenet1_1,
}


def load_model(name: str) -> nn.Module:
    """Load a pretrained architecture in train mode.

    Train mode is required both for QAT preparation and for the gradient
    computation used by GradCAM++.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](weights="DEFAULT")
    model.train()
    return model


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Return the layer used as the CAM target for a given architecture.

    The mapping is architecture-specific (the last convolutional stage that
    still carries spatial resolution) and must be extended when adding models
    to ``MODEL_REGISTRY``.
    """
    if name == "vgg16":
        return model.features[-2]
    if name == "resnet50":
        return model.layer4[-1].conv3
    if name == "densenet121":
        return model.features[-2]
    if name == "mobilenet_v2":
        return model.features[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    if name == "squeezenet1_1":
        return model.features[-1]
    raise KeyError(f"No target layer defined for model '{name}'.")
