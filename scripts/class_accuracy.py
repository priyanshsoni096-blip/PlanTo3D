"""Per-class accuracy against CubiCasa's own annotations.

The headline Dice from training is scored on the validation split during
the run; this scores the checkpoint that is actually installed, on whole
sheets at their own resolution, which is how the pipeline sees them.

Reported per class because the average hides the thing that matters: the
room classes carry most of the pixels and the classes the geometry needs
most -- wall, door, window -- carry the fewest.

Recall is printed beside IoU deliberately. IoU is brutal on a thin class:
a window is about four pixels wide, so a one-pixel offset costs a quarter
of it, and a low IoU can mean "found, drawn slightly wide" rather than
"missed". Recall separates the two.

Two averages are printed, because they disagree a lot and picking one
quietly would be a way of choosing an answer. Pooling over pixels asks
what share of all the bath in the corpus was found, and one large sheet
can carry the figure; the per-sheet median asks how bath does on a
typical drawing, and counts a small sheet equally. Bath scores 0.60
pooled and 0.94 per sheet. **The pooled column is the one quoted in
`docs/AUDIT.md` and the README**, because it is the less flattering of
the two and because everything else there is pooled.

    python scripts/class_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
"""

import argparse
import logging
import statistics
import warnings
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import CLASS_NAMES, NUM_CLASSES
from planto3d.cubicasa import svg_to_mask
from planto3d.segment import load_segmenter

# A class has to appear on at least this many sheets to be worth a row;
# below it the median is one or two drawings rather than a measurement.
MIN_SHEETS = 3


def main(root: str, checkpoint: Path | None, limit: int) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    segmenter = load_segmenter(checkpoint)

    intersections = np.zeros(NUM_CLASSES, dtype=float)
    unions = np.zeros(NUM_CLASSES, dtype=float)
    actuals = np.zeros(NUM_CLASSES, dtype=float)
    hits = np.zeros(NUM_CLASSES, dtype=float)
    ious: dict[int, list[float]] = {index: [] for index in range(NUM_CLASSES)}
    recalls: dict[int, list[float]] = {index: [] for index in range(NUM_CLASSES)}
    share: dict[int, list[float]] = {index: [] for index in range(NUM_CLASSES)}
    sheets = 0

    for path in sorted(Path(root).glob("*/*/F1_scaled.png"))[:limit]:
        image = cv2.imread(str(path))
        annotation = path.parent / "model.svg"
        if image is None or not annotation.is_file():
            continue

        truth = svg_to_mask(annotation, image.shape[:2])
        predicted = segmenter(image)
        sheets += 1

        for index in range(NUM_CLASSES):
            actual = truth == index
            if not actual.any():
                continue
            guessed = predicted == index
            union = np.logical_or(actual, guessed).sum()
            hit = np.logical_and(actual, guessed).sum()
            intersections[index] += hit
            unions[index] += union
            actuals[index] += actual.sum()
            hits[index] += hit
            ious[index].append(float(hit / union) if union else 0.0)
            recalls[index].append(float(hit / actual.sum()))
            share[index].append(float(actual.mean()))

    if not sheets:
        print("no sheets scored")
        return

    print(f"{sheets} sheets")
    print()
    print(
        f"{'class':14}{'sheets':>7}{'share':>8}"
        f"{'IoU pooled':>12}{'recall':>9}{'IoU per sheet':>15}"
    )
    print("-" * 65)

    rows = [
        (
            CLASS_NAMES[index],
            len(ious[index]),
            statistics.median(share[index]),
            float(intersections[index] / unions[index]) if unions[index] else 0.0,
            float(hits[index] / actuals[index]) if actuals[index] else 0.0,
            statistics.median(ious[index]),
        )
        for index in range(NUM_CLASSES)
        if len(ious[index]) >= MIN_SHEETS
    ]
    for name, count, portion, pooled, recall, per_sheet in sorted(
        rows, key=lambda row: -row[3]
    ):
        print(
            f"{name:14}{count:>7}{portion:>8.2%}{pooled:>12.3f}"
            f"{recall:>9.1%}{per_sheet:>15.3f}"
        )

    print()
    print(
        "median across classes, pooled: "
        f"{statistics.median(row[3] for row in rows):.3f}"
    )
    print("Pooled is the figure quoted in docs/AUDIT.md and the README.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.limit)
