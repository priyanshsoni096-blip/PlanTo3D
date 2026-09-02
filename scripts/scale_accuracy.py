"""Score the inferred scale against the sizes CubiCasa recorded.

Scale decides how big the finished house is. Until now it could not be
checked: the only drawing with printed dimensions was the one the project
started with, and everything else fell back to inferring from door widths or
wall thickness with no way of knowing whether the answer was right.

CubiCasa records each room's real size inside the annotation as a hidden
label, so every plan in the dataset carries an answer. This runs the
calibration stage over as many as asked for and reports the error.

    python scripts/scale_accuracy.py path/to/cubicasa --checkpoint models/unet.pt

Read the median absolute error rather than the mean: one plan that lands
three times out of true says more about that plan than about the method,
and the mean hides how the bulk behaves.
"""

import argparse
import logging
import statistics
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import cv2

from planto3d.cubicasa import ground_truth_scale
from planto3d.pipeline import run
from planto3d.segment import load_segmenter

warnings.filterwarnings("ignore")

# Beyond this the model is a different building rather than an imprecise
# one: a fifth off turns a 12 ft room into 10 ft, which reads as wrong.
ACCEPTABLE_ERROR = 0.20


def true_scale(svg_path: Path, image_path: Path) -> float | None:
    """Ground truth in the image's pixels rather than the annotation's.

    The rendered PNG is not quite the SVG's size -- it carries a footer the
    geometry does not -- so the width ratio converts between them. Width
    rather than height because the footer only adds height.
    """
    svg_scale = ground_truth_scale(svg_path)
    if svg_scale is None:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    header = svg_path.read_text(encoding="utf-8", errors="ignore")[:600]
    marker = 'width="'
    start = header.find(marker)
    if start < 0:
        return svg_scale
    svg_width = float(header[start + len(marker) : header.find('"', start + len(marker))])

    return svg_scale * (image.shape[1] / svg_width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=24)
    arguments = parser.parse_args()

    logging.disable(logging.WARNING)
    segmenter = load_segmenter(arguments.checkpoint)

    annotations = sorted(arguments.root.glob("*/*/model.svg"))[: arguments.limit]
    print(f"scoring {len(annotations)} plan(s)\n")
    print(f"{'plan':10}{'inferred':>10}{'true':>8}{'error':>9}  source")
    print("-" * 50)

    errors: list[float] = []
    by_source: dict[str, list[float]] = {}

    for annotation in annotations:
        image_path = annotation.parent / "F1_scaled.png"
        expected = true_scale(annotation, image_path)
        if expected is None:
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="planto3d_scale_") as workdir:
                result = run(image_path, Path(workdir), segmenter=segmenter, crop=False)
        except Exception as error:
            print(f"{annotation.parent.name:10}  FAILED  {type(error).__name__}")
            continue

        if not result.scale:
            continue

        error = (result.scale - expected) / expected
        errors.append(error)
        by_source.setdefault(result.scale_source, []).append(error)

        flag = "" if abs(error) <= ACCEPTABLE_ERROR else "  <-- off"
        print(
            f"{annotation.parent.name:10}{result.scale:10.1f}{expected:8.1f}"
            f"{error:+8.0%}  {result.scale_source}{flag}"
        )

    if not errors:
        print("\nnothing scored")
        return

    absolute = sorted(abs(e) for e in errors)
    within = sum(1 for e in absolute if e <= ACCEPTABLE_ERROR)

    print("-" * 50)
    print(f"scored          {len(errors)}")
    print(f"median error    {statistics.median(absolute):.1%}")
    print(f"worst           {absolute[-1]:.1%}")
    print(f"within {ACCEPTABLE_ERROR:.0%}      {within}/{len(errors)}")

    # Sources are reported separately because they fail in opposite
    # directions and a pooled median hides it: measured over 30 sheets the
    # split is an even 15 walls to 15 doors, and correcting the wall
    # constant alone moved the pooled figure from 17.7% to 9.9% while
    # correcting the door constant alone moved it to 20.1%. A change that
    # helps one source and harms the other nets out to nothing here.
    print()
    print(f"{'source':12}{'plans':>7}{'median err':>12}{'within 20%':>12}")
    for source in sorted(by_source):
        magnitudes = sorted(abs(error) for error in by_source[source])
        within = sum(1 for error in magnitudes if error <= ACCEPTABLE_ERROR)
        print(
            f"{source:12}{len(magnitudes):>7}"
            f"{statistics.median(magnitudes):>11.1%}"
            f"{within:>8}/{len(magnitudes)}"
        )


if __name__ == "__main__":
    main()
