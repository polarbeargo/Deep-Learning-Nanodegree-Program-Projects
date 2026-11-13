"""Compose four generated digits; this is a visual demo, not a secure CAPTCHA."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch

from utils.checkpoint import OUTPUT_DIR
from utils.generation import generate_digits, load_generator
from utils.runtime import setup


def create_captcha(text, manifest_path, output_path, device="cpu", seed=42):
    if len(text) != 4 or any(character not in "0123456789" for character in text):
        raise ValueError("Provide exactly four ASCII digits; quote strings with leading zeroes.")
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("selection_basis") != "synthetic_classifier_real_train_validation_accuracy":
        raise ValueError("Expected a model-selection manifest from 03_evaluation.ipynb.")
    canvas = Image.new("L", (4 * 36 + 8, 44), color=255)
    models = {}
    for position, character in enumerate(text):
        record = manifest["digits"][character]
        checkpoint = (manifest_path.parent / record["checkpoint"]).resolve()
        key = (str(checkpoint), record["kind"])
        if key not in models:
            models[key] = load_generator(checkpoint, record["kind"], device)
        images, _ = generate_digits(
            models[key], record["kind"], [int(character)], device,
            seed=seed + position, sampling_steps=manifest["sampling_steps"],
        )
        # MNIST has light foreground on dark background; invert for a white page.
        pixels = ((images[0, 0] + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
        tile = ImageOps.invert(Image.fromarray(pixels))
        canvas.paste(tile, (8 + 36 * position, 8))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("digits")
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "captcha_models.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "captcha.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_captcha(args.digits, args.manifest, args.output, setup(args.seed), args.seed)
    print(f"Saved demonstration image: {args.output}")


if __name__ == "__main__":
    main()
