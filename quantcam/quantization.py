"""Fake-quantization configurations and QAT-preparation helpers.

The study compares full-precision (fp32) models against int8 and int16
fake-quantized variants. Quantization is applied through QAT *preparation*
(:func:`torch.quantization.prepare_qat`), which inserts fake-quant/observer
modules so the forward pass simulates the target precision while remaining
differentiable — a requirement for gradient-based CAM methods such as
GradCAM++.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    default_fake_quant,
    default_weight_fake_quant,
)
from torch.quantization import QConfig

# --- int16 fake-quant (custom observers) ---
_int16_activation = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-32768,
    quant_max=32767,
    dtype=torch.int16,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
_int16_weight = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-32768,
    quant_max=32767,
    dtype=torch.int16,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
)

QCONFIG_INT16 = QConfig(activation=_int16_activation, weight=_int16_weight)

# --- int8 fake-quant (PyTorch defaults) ---
QCONFIG_INT8 = QConfig(activation=default_fake_quant, weight=default_weight_fake_quant)

# Precision name -> QConfig. "fp32" is handled separately (no fake-quant).
QCONFIGS = {
    "int16": QCONFIG_INT16,
    "int8": QCONFIG_INT8,
}

# Canonical ordering used as the default across the CLI.
PRECISIONS = ["fp32", "int16", "int8"]


def quantize_model(
    model: nn.Module,
    name: str,
    precisions: list[str] | None = None,
) -> list[dict]:
    """Return one model record per requested precision.

    Parameters
    ----------
    model:
        A model in train mode.
    name:
        Architecture name, propagated to each returned record.
    precisions:
        Subset of :data:`PRECISIONS`. Defaults to all. ``"fp32"`` yields the
        original model unchanged; every other precision yields a QAT-prepared
        copy (the original is left untouched by ``inplace=False``).

    Returns
    -------
    list of dict
        Records of the form ``{"name", "precision", "model"}``.
    """
    if precisions is None:
        precisions = PRECISIONS

    variants: list[dict] = []
    for precision in precisions:
        if precision == "fp32":
            variants.append({"name": name, "precision": "fp32", "model": model})
            continue
        if precision not in QCONFIGS:
            raise KeyError(f"Unknown precision '{precision}'. Available: {PRECISIONS}")
        model.qconfig = QCONFIGS[precision]
        prepared = torch.quantization.prepare_qat(model, inplace=False)
        variants.append({"name": name, "precision": precision, "model": prepared})
    return variants
