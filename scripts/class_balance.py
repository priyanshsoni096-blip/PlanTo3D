"""Measure how much of a drawing each class occupies.

The loss weights in ``training.train`` are derived from these numbers, so
re-run this after any change to the class scheme rather than adjusting the
weights by feel. Prints both the mean share of pixels and how many plans
carry the class at all -- a class present on every plan but thin, like a
window, needs different handling from one that is absent from most.

    python scripts/class_balance.py path/to/cubicasa
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import CLASS_NAMES, NUM_CLASSES
from planto3d.cubicasa import svg_to_mask

# Below this share, a class is treated as absent rather than present in
# trace amounts -- a few stray pixels from a neighbouring polygon should not
# count as a plan having a sauna.
PRESENCE_THRESHOLD = 0.0005


def measure(root: Path, limit: int | None = None) -> dict[int, tuple[float, int]]:
    """Mean pixel share and plan count per class, over every annotation found."""
    annotations = sorted(root.glob("*/*/model.svg"))[:limit]
    if not annotations:
        raise SystemExit(f"no model.svg found under {root}")

    shares = np.zeros(NUM_CLASSES)
    present = np.zeros(NUM_CLASSES, dtype=int)

    counted = 0
    for annotation in annotations:
        image = cv2.imread(str(annotation.parent / "F1_scaled.png"))
        if image is None:
            continue

        mask = svg_to_mask(annotation, image.shape[:2])
        for index in range(NUM_CLASSES):
            share = float((mask == index).mean())
            shares[index] += share
            present[index] += share > PRESENCE_THRESHOLD
        counted += 1

    return {
        index: (shares[index] / counted, int(present[index]))
        for index in range(NUM_CLASSES)
    }, counted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="folder of CubiCasa sample directories")
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    results, counted = measure(arguments.root, arguments.limit)

    print(f"{counted} plan(s)\n")
    print(f"{'class':14}{'share':>9}{'on N plans':>13}")
    for index, (share, plans) in results.items():
        print(f"{CLASS_NAMES[index]:14}{share * 100:8.2f}%{plans:9}/{counted}")

    print("\nCLASS_FREQUENCY = {")
    for index, (share, _) in results.items():
        print(f"    {CLASS_NAMES[index].upper()}: {share:.4f},")
    print("}")


if __name__ == "__main__":
    main()
