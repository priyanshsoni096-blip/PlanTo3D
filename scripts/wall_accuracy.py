"""Do the extracted walls land where the drawing puts its walls?

Everything downstream rests on this. The per-class IoU says how well the
*segmenter* finds wall pixels; this asks the next question, which is
whether the geometry stage turns those pixels into wall segments in the
right places.

Measured two ways, because they fail differently:

    coverage   how much annotated wall has an extracted wall lying on it
               -- what the model is missing
    agreement  how much extracted wall lies on annotated wall
               -- how much of what it builds is invented

Both are computed by painting the wall segments back onto a canvas at
their own thickness and comparing with the annotation, so a wall roughly
in the right place with roughly the right thickness scores well and one
off by a room does not.

Usage:
    python scripts/wall_accuracy.py <cubicasa-root> [--checkpoint model.pt]
                                    [--limit 30]
"""

import argparse
import logging
import statistics
import warnings
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import WALL
from planto3d.cubicasa import svg_to_mask
from planto3d.extract import extract_walls, wall_gauge
from planto3d.segment import load_segmenter

# How far a built wall may sit from an annotated one and still count, as a
# share of the drawing's own wall thickness. Half a wall is generous; more
# would let a wall match its neighbour across a doorway.
TOLERANCE = 0.5

# Below this a plan is reported individually, since the median hides the
# handful of plans where the geometry actually went wrong.
POOR = 0.70


def painted(walls, shape: tuple[int, int]) -> np.ndarray:
    """The wall segments drawn back onto a canvas at their own thickness."""
    canvas = np.zeros(shape, np.uint8)
    for wall in walls:
        cv2.line(
            canvas,
            (int(round(wall.start[0])), int(round(wall.start[1]))),
            (int(round(wall.end[0])), int(round(wall.end[1]))),
            255,
            max(int(round(wall.thickness)), 1),
        )
    return canvas > 0


def score(mask_truth: np.ndarray, walls) -> tuple[float, float] | None:
    """Coverage and agreement for one plan, or None if it has no wall."""
    truth = mask_truth == WALL
    if not truth.any():
        return None

    built = painted(walls, truth.shape)
    if not built.any():
        return 0.0, 0.0

    # A wall a pixel off is not a wrong wall.
    slack = max(int(wall_gauge(mask_truth) * TOLERANCE), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (slack * 2 + 1,) * 2)
    truth_near = cv2.dilate(truth.astype(np.uint8), kernel) > 0
    built_near = cv2.dilate(built.astype(np.uint8), kernel) > 0

    coverage = float(np.logical_and(truth, built_near).sum() / truth.sum())
    agreement = float(np.logical_and(built, truth_near).sum() / built.sum())
    return coverage, agreement


def main(root: str, checkpoint: Path | None, limit: int) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    segmenter = load_segmenter(checkpoint)
    rows: list[tuple[str, float, float]] = []

    for path in sorted(Path(root).glob("*/*/F1_scaled.png"))[:limit]:
        image = cv2.imread(str(path))
        annotation = path.parent / "model.svg"
        if image is None or not annotation.is_file():
            continue

        truth = svg_to_mask(annotation, image.shape[:2])
        result = score(truth, extract_walls(segmenter(image)))
        if result is None:
            continue
        rows.append((path.parent.name, result[0], result[1]))

    if not rows:
        print("no plans scored")
        return

    coverage = [row[1] for row in rows]
    agreement = [row[2] for row in rows]

    print(f"{len(rows)} plans\n")
    print(f"{'':44}{'median':>9}{'mean':>9}")
    print(f"{'coverage  (annotated wall that was built)':44}"
          f"{statistics.median(coverage):>9.1%}{statistics.mean(coverage):>9.1%}")
    print(f"{'agreement (built wall that is really wall)':44}"
          f"{statistics.median(agreement):>9.1%}{statistics.mean(agreement):>9.1%}")

    print(f"\nplans below {POOR:.0%} coverage:  "
          f"{sum(1 for c in coverage if c < POOR)} of {len(rows)}")
    print(f"plans below {POOR:.0%} agreement: "
          f"{sum(1 for a in agreement if a < POOR)} of {len(rows)}")

    print(f"\nworst by agreement\n{'plan':12}{'coverage':>11}{'agreement':>12}")
    for name, cov, agr in sorted(rows, key=lambda r: r[2])[:8]:
        print(f"{name:12}{cov:>11.1%}{agr:>12.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="directory of CubiCasa plan folders")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.limit)
