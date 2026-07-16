"""Command-line entry point: generate CAM heatmaps across models and precisions."""

from __future__ import annotations

import argparse
import datetime
import logging
import warnings

import torch
from tqdm import tqdm

from .cam import generate_heatmap
from .data import build_preprocess, list_images, load_image, preprocess_image
from .models import MODEL_REGISTRY, get_target_layer, load_model
from .quantization import PRECISIONS, quantize_model
from .visualize import resize_heatmap, save_heatmap

LOGGER = logging.getLogger("quantcam")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quantcam",
        description=(
            "Generate GradCAM++ heatmaps for full-precision and quantized "
            "(int8/int16) CNNs over an image set."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="./ILSVRC/Data/CLS-LOC/val",
        help="Directory of input images (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="./results_heatmaps",
        help="Directory to write heatmaps (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY),
        metavar="MODEL",
        help="Architectures to evaluate (default: all registered models).",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=PRECISIONS,
        choices=PRECISIONS,
        metavar="PRECISION",
        help="Precisions to evaluate (default: fp32 int16 int8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max images to process; use 0 for no limit (default: %(default)s).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square resize resolution for model input (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'cpu', or 'cuda' (default: auto).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: quantcam_<timestamp>.log).",
    )
    return parser.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def configure_logging(log_file: str | None) -> str:
    if log_file is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"quantcam_{stamp}.log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return log_file


def run(args: argparse.Namespace) -> None:
    log_file = configure_logging(args.log_file)
    device = resolve_device(args.device)
    limit = None if args.limit in (0, None) else args.limit

    LOGGER.info("Starting QuantCAM run on device=%s, log=%s", device, log_file)

    preprocess = build_preprocess(args.image_size)
    image_files = list_images(args.data_dir, limit=limit)
    LOGGER.info("Found %d images in %s", len(image_files), args.data_dir)
    print(
        f"Device: {device} | images: {len(image_files)} | "
        f"models: {args.models} | precisions: {args.precisions}"
    )

    for name in args.models:
        LOGGER.info("Loading model %s", name)
        print(f"\n=== {name} ===")
        model = load_model(name)
        variants = quantize_model(model, name, precisions=args.precisions)

        for variant in variants:
            precision = variant["precision"]
            net = variant["model"].to(device)
            target_layer = get_target_layer(net, name)
            LOGGER.info("Inference: %s / %s", name, precision)
            print(f"  {name} [{precision}]")

            pbar = tqdm(
                image_files,
                colour="green",
                total=len(image_files),
                desc=f"{name}/{precision}",
            )
            for i, image_file in enumerate(pbar):
                try:
                    image = load_image(image_file)
                    input_batch = preprocess_image(image, preprocess).to(device)
                    input_batch.requires_grad_(True)
                    heatmap = generate_heatmap(net, input_batch, target_layer)
                    heatmap = resize_heatmap(heatmap, image.size)
                    save_heatmap(image_file, heatmap, name, precision, args.output_dir)
                except Exception as exc:  # keep the batch running on a bad image
                    LOGGER.error(
                        "Failed on %s (%s/%s): %s", image_file, name, precision, exc
                    )
                    continue
                if (i + 1) % 1000 == 0:
                    LOGGER.info(
                        "Processed %d/%d images for %s/%s",
                        i + 1,
                        len(image_files),
                        name,
                        precision,
                    )
            LOGGER.info("Finished %s / %s", name, precision)
        LOGGER.info("Finished model %s", name)

    print(f"\nDone. Heatmaps written to {args.output_dir}")
    LOGGER.info("Run complete.")


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
