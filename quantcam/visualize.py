"""Heatmap post-processing: resize, colorize, overlay, and save."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a heatmap to ``size`` (width, height) and normalize to ``[0, 1]``.

    A zero value-range (constant heatmap) is left unscaled to avoid a
    division-by-zero.
    """
    resized = Image.fromarray(heatmap).resize(size, Image.BICUBIC)
    resized = np.asarray(resized, dtype=np.float32)
    value_range = resized.max() - resized.min()
    if value_range > 0:
        resized = (resized - resized.min()) / value_range
    return resized


def color_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Map a grayscale heatmap through the 'jet' colormap (RGB, alpha dropped)."""
    cmap = cm.get_cmap("jet")
    return cmap(heatmap)[:, :, :3]


def overlay_heatmap(image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Alpha-blend a colored heatmap over the original image."""
    blended = alpha * np.asarray(image) + (1 - alpha) * heatmap * 255
    return Image.fromarray(blended.astype(np.uint8))


def save_heatmap(
    image_path: str,
    heatmap: np.ndarray,
    model_name: str,
    precision: str,
    output_dir: str,
) -> str:
    """Save a grayscale heatmap PNG under ``output_dir/model_name/precision/``.

    Returns the output path.
    """
    save_dir = os.path.join(output_dir, model_name, precision)
    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    output_path = os.path.join(save_dir, image_name)
    plt.imsave(output_path, heatmap, cmap="gray", pil_kwargs={"compress_level": 0})
    return output_path
