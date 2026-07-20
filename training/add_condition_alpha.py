from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def border_connected_alpha(
    image: Image.Image,
    min_background_luma: float = 145.0,
    max_background_chroma: float = 24.0,
    feather_radius: float = 0.6,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = rgb.mean(axis=-1)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    candidate = (luma >= min_background_luma) & (
        chroma <= max_background_chroma
    )

    flood = Image.fromarray(candidate.astype(np.uint8) * 255, mode="L")
    draw = ImageDraw.Draw(flood)
    width, height = flood.size
    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if flood.getpixel(seed) == 255:
            ImageDraw.floodfill(flood, seed, 128, thresh=0)

    background = np.asarray(flood) == 128
    alpha = Image.fromarray((~background).astype(np.uint8) * 255, mode="L")
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather_radius))
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preserve a rendered subject and platform while removing its border-connected neutral background."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-background-luma", type=float, default=145.0)
    parser.add_argument("--max-background-chroma", type=float, default=24.0)
    parser.add_argument("--feather-radius", type=float, default=0.6)
    args = parser.parse_args()

    with Image.open(args.input) as image:
        result = border_connected_alpha(
            image,
            min_background_luma=args.min_background_luma,
            max_background_chroma=args.max_background_chroma,
            feather_radius=args.feather_radius,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    alpha = np.asarray(result.getchannel("A"))
    print(
        f"{args.output}: foreground={(alpha > 127).mean():.3f}, "
        f"transparent={(alpha == 0).mean():.3f}"
    )


if __name__ == "__main__":
    main()
