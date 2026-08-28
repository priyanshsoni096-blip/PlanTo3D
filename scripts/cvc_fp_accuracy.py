"""Wall, room and opening accuracy against CVC-FP's own annotations.

CVC-FP is a second, unrelated drafting tradition -- Spanish/Catalan
architectural office plans, not CubiCasa's Finnish residential set -- so
this is the generalization check: does geometry extraction still work on
a convention the model has never been tuned against?

Modeled directly on scripts/wall_accuracy.py (coverage/agreement) and
scripts/class_accuracy.py (per-class IoU, pooled and per-sheet median).
Read those first if this script's shape is unclear.

CVC-FP carries no room-type labels -- every space is annotated "Room" --
so only wall/room/window/door are scored; there is no bath/kitchen/storage
comparison to make here the way there is on CubiCasa.

Wall coverage reproduces the documented 96.7% (measured 96.3%) and all four
class IoUs reproduce exactly (wall 0.636, room 0.530, window 0.239, door
0.136). Wall agreement measures 69.0% against a documented 77.5%; the
difference is in the painting/dilation method of the original throwaway code,
which was not recorded. This script's figure is the reproducible one.

    python scripts/cvc_fp_accuracy.py <cvc-fp-root> [--checkpoint model.pt] [--limit 30]
"""

import argparse
import logging
import statistics
import warnings
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import CLASS_NAMES, DOOR, NUM_CLASSES, ROOM, WALL, WINDOW
from planto3d.cvc_fp import sample_paths, svg_to_mask
from planto3d.extract import extract_walls, wall_gauge
from planto3d.segment import load_segmenter

# Same tolerance as scripts/wall_accuracy.py -- half a wall's own thickness,
# so this script scores identically to its sibling and the two numbers are
# comparable side by side.
MATCH_RATIO = 0.5

MIN_SHEETS = 3


def _iou(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    if union == 0:
        return None
    return float(intersection) / float(union)


def _paint_walls(walls, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for wall in walls:
        cv2.line(
            canvas,
            (int(wall.start[0]), int(wall.start[1])),
            (int(wall.end[0]), int(wall.end[1])),
            1,
            max(1, int(wall.thickness)),
        )
    return canvas.astype(bool)


def _wall_score(truth: np.ndarray, walls) -> tuple[float, float] | None:
    """Coverage and agreement, painted and compared exactly as wall_accuracy.py does."""
    truth_wall = truth == WALL
    if not truth_wall.any():
        return None
    built = _paint_walls(walls, truth.shape)
    if not built.any():
        return 0.0, 0.0

    gauge = wall_gauge(truth)
    thickness = max(1, int(gauge * MATCH_RATIO))
    kernel = np.ones((thickness, thickness), np.uint8)
    truth_dilated = cv2.dilate(truth_wall.astype(np.uint8), kernel).astype(bool)
    built_dilated = cv2.dilate(built.astype(np.uint8), kernel).astype(bool)

    coverage = np.logical_and(truth_wall, built_dilated).sum() / truth_wall.sum()
    agreement = np.logical_and(built, truth_dilated).sum() / built.sum()
    return float(coverage), float(agreement)


def main(root: str, checkpoint: Path | None, limit: int) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    segmenter = load_segmenter(checkpoint)

    coverages, agreements = [], []
    per_class_pooled = {index: [0, 0] for index in range(NUM_CLASSES)}  # [intersection, union]
    per_class_ious: dict[int, list[float]] = {index: [] for index in range(NUM_CLASSES)}
    usable = 0

    pairs = sample_paths(Path(root))[:limit]
    for image_path, svg_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        truth = svg_to_mask(svg_path, image.shape[:2])
        if not (truth == WALL).any():
            continue

        predicted_classes = segmenter(image)
        walls = extract_walls(predicted_classes, gauge=wall_gauge(predicted_classes))
        if walls:
            usable += 1

        wall_result = _wall_score(truth, walls)
        if wall_result:
            coverages.append(wall_result[0])
            agreements.append(wall_result[1])

        for index in range(NUM_CLASSES):
            predicted_mask = predicted_classes == index
            truth_mask = truth == index
            if not truth_mask.any():
                continue
            intersection = int(np.logical_and(predicted_mask, truth_mask).sum())
            union = int(np.logical_or(predicted_mask, truth_mask).sum())
            per_class_pooled[index][0] += intersection
            per_class_pooled[index][1] += union
            iou = _iou(predicted_mask, truth_mask)
            if iou is not None:
                per_class_ious[index].append(iou)

    print(f"{len(pairs)} plans, {usable} yielding usable geometry\n")
    if coverages:
        print(f"wall coverage:  {statistics.median(coverages):.1%}")
        print(f"wall agreement: {statistics.median(agreements):.1%}")
    print()
    print(f"{'class':10} {'pooled IoU':>10} {'median IoU':>10} {'sheets':>7}")
    for index in (WALL, ROOM, DOOR, WINDOW):
        sheets = len(per_class_ious[index])
        if sheets < MIN_SHEETS:
            continue
        name = CLASS_NAMES[index]
        intersection, union = per_class_pooled[index]
        pooled = intersection / union if union else 0.0
        median = statistics.median(per_class_ious[index])
        print(f"{name:10} {pooled:>10.3f} {median:>10.3f} {sheets:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.limit)
