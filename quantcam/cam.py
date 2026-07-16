"""GradCAM++ heatmap generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchcam.methods import GradCAMpp


def generate_heatmap(
    model: nn.Module,
    input_batch: torch.Tensor,
    target_layer: nn.Module,
) -> np.ndarray:
    """Generate a GradCAM++ heatmap for the model's top-1 prediction.

    Parameters
    ----------
    model:
        Model in train mode (GradCAM++ requires gradients).
    input_batch:
        Preprocessed input of shape ``(1, 3, H, W)`` with ``requires_grad=True``.
    target_layer:
        Layer whose activations and gradients drive the CAM.

    Returns
    -------
    np.ndarray
        A 2D heatmap at the target layer's feature-map resolution, before
        resizing and normalization.

    Notes
    -----
    A fresh extractor is created per call and its forward/backward hooks are
    removed afterwards, so repeated calls in a loop do not accumulate hooks on
    the model.
    """
    cam_extractor = GradCAMpp(model, target_layer=target_layer)
    try:
        out = model(input_batch)
        class_idx = out.squeeze(0).argmax().item()
        cams = cam_extractor(class_idx, out)
        heatmap = cams[0].squeeze(0).detach().cpu().numpy()
    finally:
        cam_extractor.remove_hooks()
    return heatmap
