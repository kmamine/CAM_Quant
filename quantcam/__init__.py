"""QuantCAM — quantization effects on CNN class activation maps.

Reference implementation for the IPTA 2024 paper
*"Quantization Effects on Neural Networks Perception: How Would Quantization
Change the Perceptual Field of Vision Models?"* (Kerkouri, Tliba, Chetouani,
Bruno).

The package generates GradCAM++ heatmaps for full-precision (fp32) and
fake-quantized (int8 / int16) CNNs so that the effect of quantization on a
model's perceptual field can be compared against visual saliency.
"""

from __future__ import annotations

from .cam import generate_heatmap
from .data import build_preprocess, list_images, load_image, preprocess_image
from .models import MODEL_REGISTRY, get_target_layer, load_model
from .quantization import PRECISIONS, QCONFIGS, quantize_model
from .visualize import color_heatmap, overlay_heatmap, resize_heatmap, save_heatmap

__version__ = "0.1.0"

__all__ = [
    "MODEL_REGISTRY",
    "load_model",
    "get_target_layer",
    "PRECISIONS",
    "QCONFIGS",
    "quantize_model",
    "generate_heatmap",
    "build_preprocess",
    "list_images",
    "load_image",
    "preprocess_image",
    "resize_heatmap",
    "color_heatmap",
    "overlay_heatmap",
    "save_heatmap",
]
