# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reference implementation for the IPTA 2024 paper *"Quantization Effects on
Neural Networks Perception"* (Kerkouri, Tliba, Chetouani, Bruno; see
`QuantCAM___IPTA_2024.pdf`). It studies how quantization affects CNN class
activation maps (CAMs) — the alignment between GradCAM++ heatmaps and visual
saliency across six architectures and three precisions (fp32 / int16 / int8).

## Commands

```bash
pip install -r requirements.txt          # dependencies (torch, torchvision, torchcam, ...)

python run.py                            # full study: all models x precisions, 10k images
python run.py --help                     # all options
python run.py --models resnet50 --precisions fp32 int8 --limit 20   # quick smoke test
python -m quantcam                       # equivalent to run.py

python -m py_compile quantcam/*.py run.py   # syntax check (there is no test suite)
```

The dataset (ImageNet ILSVRC-2012 validation split) is **not** in the repo and
is git-ignored; pass its path via `--data-dir` (default `./ILSVRC/Data/CLS-LOC/val`).
Outputs go to `--output-dir` (default `./results_heatmaps/<model>/<precision>/<image>.png`),
and each run writes a timestamped `quantcam_<timestamp>.log`. All three are
git-ignored.

## Architecture

`quantcam/` is a small package; the pipeline is a triple-nested loop (model ×
precision × image) driven by [quantcam/cli.py](quantcam/cli.py):

- [models.py](quantcam/models.py) — `MODEL_REGISTRY` (name → torchvision constructor)
  and `get_target_layer(model, name)`, which maps each architecture to the final
  conv stage GradCAM++ hooks. **Adding a model requires updating both.** Models
  are loaded in `.train()` mode (required for QAT prep and for GradCAM++ gradients).
- [quantization.py](quantcam/quantization.py) — int8/int16 fake-quant `QConfig`s
  and `quantize_model()`, which returns one record `{name, precision, model}` per
  requested precision. `fp32` returns the original model untouched; other
  precisions return a `prepare_qat(inplace=False)` **copy**. Quantization is QAT
  *preparation* (fake-quant/observer modules), not full conversion — this keeps
  the forward pass differentiable so GradCAM++ works.
- [cam.py](quantcam/cam.py) — `generate_heatmap()` runs GradCAM++ for the top-1
  prediction. A fresh extractor is created per call and its hooks removed in a
  `finally` block, so the 10k-image loop does not leak hooks.
- [data.py](quantcam/data.py) — image discovery/loading and ImageNet preprocessing.
- [visualize.py](quantcam/visualize.py) — `resize_heatmap` (bicubic + min-max
  normalize, with a zero-range guard), plus colorize/overlay/save helpers.

## Conventions and gotchas

- Precision is a string in `{"fp32", "int16", "int8"}` (`quantization.PRECISIONS`),
  used as both a CLI choice and an output-subdirectory name — keep those aligned.
- The int16 path uses hand-defined `FakeQuantize` observers; int8 uses PyTorch
  defaults. `torch.int16` fake-quant is intentional (it is the paper's method) —
  do not "simplify" it to a standard int8-only config.
- Model loading uses the modern `weights="DEFAULT"` API (torchvision ≥ 0.13),
  not the deprecated `pretrained=True`.
- The per-image loop catches and logs exceptions so one bad image doesn't abort
  a multi-hour run — check the log for skipped images rather than assuming a
  clean run processed everything.
- torch/torchvision/torchcam may not be installed in every environment. To
  validate logic without them, load a torch-free module (e.g. `visualize.py`) or
  a module with no intra-package imports (`quantization.py`) directly via
  `importlib` with stubbed `torch`, rather than importing the package (its
  `__init__` eagerly imports every submodule).
