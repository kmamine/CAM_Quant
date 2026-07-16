"""Image discovery, loading, and preprocessing."""

from __future__ import annotations

import os

import torch
import torchvision.transforms as transforms
from PIL import Image

# ImageNet channel statistics used to normalize inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_preprocess(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet preprocessing: resize -> tensor -> normalize."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def list_images(image_dir: str, limit: int | None = None) -> list[str]:
    """Return a sorted list of image file paths under ``image_dir``.

    Sorting makes the processing order deterministic; ``limit`` truncates the
    list (e.g. to the first 10,000 validation images used in the paper).
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    files = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, name))
    ]
    files = sorted(files)
    if limit is not None:
        files = files[:limit]
    return files


def load_image(path: str) -> Image.Image:
    """Open an image and ensure it is in RGB mode."""
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_image(image: Image.Image, preprocess: transforms.Compose) -> torch.Tensor:
    """Apply the preprocessing transform and add a batch dimension."""
    return preprocess(image).unsqueeze(0)
