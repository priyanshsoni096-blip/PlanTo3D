"""Measure what generalises beyond the drawings the pipeline was tuned on.

The geometry layer's constants -- wall and room grey levels, the working
resolution, the colour signals -- were all measured from one drawing set.
This runs the pipeline's stages over CubiCasa5K samples, which are drafted
by other people in other conventions, and reports what survives.

Predictions were recorded before the first run so they could not be
rationalised afterwards: the U-Net and the geometry hold up, the classical
baseline collapses, and the colour-driven windows and planting find nothing.
All three held.

Usage:
    python scripts/generalisation_test.py <samples_dir> [--checkpoint model.pt]

``samples_dir`` holds one folder per sample, each containing an
``F1_scaled.png`` -- the layout CubiCasa5K ships. Extract a handful from the
archive rather than the whole 5 GB.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import WALL
from planto3d.classical import classical_mask, vegetation_regions, window_mask
from planto3d.extract import extract_rooms, extract_walls
from planto3d.segment import load_segmenter

# A sample counts as reconstructable if it yields enough geometry to build
# something -- roughly a closed room or two, not a scattering of fragments.
MIN_WALLS = 8
MIN_ROOMS = 3


def measure(image, segmenter) -> dict:
    predicted = segmenter(image)
    baseline = classical_mask(image)

    return {
        "unet_wall_pct": float((predicted == WALL).mean()) * 100,
        "unet_walls": len(extract_walls(predicted, min_wall_length=25)),
        "unet_rooms": len(extract_rooms(predicted, min_area=2000)),
        "classical_wall_pct": float((baseline == WALL).mean()) * 100,
        "classical_walls": len(extract_walls(baseline, min_wall_length=25)),
        "has_colour_windows": bool(window_mask(image).any()),
        "planting_regions": len(vegetation_regions(image)),
    }


def main(samples_dir: Path, checkpoint: Path | None) -> None:
    logging.basicConfig(level=logging.WARNING)
    segmenter = load_segmenter(checkpoint)

    rows = []
    for folder in sorted(Path(samples_dir).iterdir()):
        page = folder / "F1_scaled.png"
        if not page.is_file():
            continue
        image = cv2.imread(str(page))
        if image is None:
            continue
        rows.append({"id": folder.name, "size": image.shape, **measure(image, segmenter)})

    if not rows:
        raise SystemExit(f"no samples found under {samples_dir}")

    print(f"{'sample':10} {'size':12} {'U-Net':>22}   {'classical':>16}   colour")
    print(f"{'':10} {'':12} {'wall%':>7}{'walls':>7}{'rooms':>8}   {'wall%':>7}{'walls':>9}   win plant")
    print("-" * 88)
    for row in rows:
        height, width = row["size"][:2]
        print(
            f"{row['id']:10} {f'{width}x{height}':12} "
            f"{row['unet_wall_pct']:6.1f}%{row['unet_walls']:7d}{row['unet_rooms']:8d}   "
            f"{row['classical_wall_pct']:6.1f}%{row['classical_walls']:9d}   "
            f"{int(row['has_colour_windows']):3d} {row['planting_regions']:5d}"
        )

    usable = sum(
        1 for r in rows if r["unet_walls"] >= MIN_WALLS and r["unet_rooms"] >= MIN_ROOMS
    )
    print("-" * 88)
    print(f"U-Net produced usable geometry on {usable}/{len(rows)} samples")
    print(
        f"classical baseline found no walls at all on "
        f"{sum(1 for r in rows if r['classical_walls'] == 0)}/{len(rows)}"
    )
    print(
        f"colour found windows on {sum(1 for r in rows if r['has_colour_windows'])}"
        f"/{len(rows)}, planting on {sum(1 for r in rows if r['planting_regions'])}/{len(rows)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    arguments = parser.parse_args()
    main(arguments.samples_dir, arguments.checkpoint)
