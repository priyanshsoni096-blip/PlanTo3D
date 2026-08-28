"""Window detection recall and precision, distinct from IoU.

class_accuracy.py's window IoU (0.089) is brutal on a class this thin --
a window is about four pixels wide, so a one-pixel offset costs a quarter
of it. This asks a different, more forgiving question: of the windows the
drawing actually has, how many did the model find at all (recall), and of
what it called a window, how much really was one (precision)? A window is
"found" when a predicted window component and an annotated one overlap by
any amount -- this is a detection count, not a shape-accuracy score.

    python scripts/window_detection_accuracy.py <cubicasa-root> [--checkpoint model.pt] [--limit 60]
"""

import argparse
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import WINDOW
from planto3d.cubicasa import svg_to_mask
from planto3d.segment import load_segmenter

# A component under this many pixels is segmentation noise, not a window --
# same floor output_scorecard.py uses for openings.
MIN_COMPONENT_AREA = 16


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Each connected blob in a binary mask, as its own boolean mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    return [
        labels == index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
    ]


def main(root: str, checkpoint: Path | None, limit: int) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    segmenter = load_segmenter(checkpoint)

    found_of_annotated = 0
    total_annotated = 0
    real_of_predicted = 0
    total_predicted = 0
    scored_sheets = 0

    for path in sorted(Path(root).glob("*/*/F1_scaled.png"))[:limit]:
        annotation = path.parent / "model.svg"
        image = cv2.imread(str(path))
        if image is None or not annotation.is_file():
            continue

        truth = svg_to_mask(annotation, image.shape[:2])
        truth_windows = _components(truth == WINDOW)
        if not truth_windows:
            continue

        predicted_classes = segmenter(image)
        predicted_windows = _components(predicted_classes == WINDOW)

        scored_sheets += 1
        total_annotated += len(truth_windows)
        total_predicted += len(predicted_windows)

        for truth_window in truth_windows:
            if any(np.logical_and(truth_window, p).any() for p in predicted_windows):
                found_of_annotated += 1
        for predicted_window in predicted_windows:
            if any(np.logical_and(predicted_window, t).any() for t in truth_windows):
                real_of_predicted += 1

    if not scored_sheets:
        print("no plans with annotated windows found")
        return

    recall = found_of_annotated / total_annotated if total_annotated else 0.0
    precision = real_of_predicted / total_predicted if total_predicted else 0.0
    print(f"{scored_sheets} plans with annotated windows")
    print(f"annotated windows:  {total_annotated}")
    print(f"predicted windows:  {total_predicted}")
    print(f"recall:    {recall:.1%}  (of annotated windows, how many were found)")
    print(f"precision: {precision:.1%}  (of predicted windows, how many are real)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=60)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.limit)
