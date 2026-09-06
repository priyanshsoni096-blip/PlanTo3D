"""Score which spaces get built open to the sky, against the annotation.

A balcony sealed under a roof slab is the most visible failure this
pipeline still has, and it is the one `scripts/output_scorecard.py` says
outright it cannot see. This is the instrument that can: it runs the
pipeline as far as room labelling, asks `features.is_open_to_sky` of every
room it found, paints the ones that say yes, and compares that painting
against CubiCasa's OUTDOOR class.

    python scripts/open_air_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt

Ground truth comes from `planto3d.cubicasa.svg_to_mask`, the same parser
`training/dataset.py` builds its training masks with. Nothing is parsed
twice: OUTDOOR is a class in that mask already.

**Scored in pixels, not in rooms.** A terrace found at half its true
extent is a quarter of a roof left standing over it, and a room-level hit
rate calls that a hit. `iou` is the headline; `recall` and `precision` are
reported beside it because the two failures are not equivalent -- a roof
wrongly removed reads worse in a render than one wrongly left on, so a
candidate that buys recall with precision has not obviously earned its
place.

Two deliberate choices about how the pipeline is run, both so that the
predicted painting and the annotation share one frame of reference:

* `crop=False`, because the annotation is drawn on the uncropped sheet;
* `split=1`, because once a sheet is split each piece carries its own
  origin and painting them all back onto the original annotation compares
  nothing. `output_scorecard.py` sidesteps the same problem by declining
  to judge walls on a split sheet.

The pipeline may still enlarge a sheet whose walls are too thin to
measure, which moves the geometry into a larger frame. That is caught
rather than assumed: the segmenter is wrapped so the harness knows the
exact shape of the last image the geometry was read from, and the
prediction is resampled back to the annotation's frame before anything is
counted.
"""

import argparse
import logging
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from planto3d.classes import OUTDOOR  # noqa: E402
from planto3d.cubicasa import svg_to_mask  # noqa: E402
from planto3d.features import is_open_to_sky  # noqa: E402
from planto3d.pipeline import extract  # noqa: E402
from planto3d.segment import load_segmenter  # noqa: E402


@dataclass
class PlanScore:
    """One sheet's open-to-sky agreement, in pixels of the sheet.

    `rooms` and `undecidable` are carried alongside because the ceiling on
    this work is not accuracy but evidence: a room with neither a printed
    label nor a predicted type cannot be decided by any rule, and the
    share of those bounds what any candidate can reach.
    """

    plan: str
    true_open_px: int
    predicted_open_px: int
    intersection_px: int
    rooms: int
    undecidable: int


class _FrameRecorder:
    """A segmenter that remembers the shape of the last image it read.

    `pipeline._extract_floor` re-segments an enlarged copy when the walls
    are too thin to measure, and the room polygons then live in that
    enlarged frame with nothing on the result to say so. Recording the
    shape here is exact and free, where inferring it afterwards from
    polygon extents would be a guess.
    """

    def __init__(self, segmenter):
        self._segmenter = segmenter
        self.shape: tuple[int, int] | None = None

    def __call__(self, image):
        self.shape = (image.shape[0], image.shape[1])
        return self._segmenter(image)


def _painted(rooms, shape: tuple[int, int]) -> np.ndarray:
    """The open-to-sky rooms, filled onto a blank sheet."""
    canvas = np.zeros(shape, dtype=np.uint8)
    for room in rooms:
        polygon = np.array(
            [[int(round(x)), int(round(y))] for x, y in room.polygon], dtype=np.int32
        )
        cv2.fillPoly(canvas, [polygon], 1)
    return canvas


def score_plan(image_path: Path, svg_path: Path, segmenter) -> PlanScore | None:
    """Score one sheet. None when it cannot be read or built at all.

    A sheet that crashes the pipeline is not scored rather than scored
    zero: it is a different defect, and folding it in here would make this
    number move for reasons that have nothing to do with open air.
    """
    image_path, svg_path = Path(image_path), Path(svg_path)
    image = cv2.imread(str(image_path))
    if image is None or not svg_path.is_file():
        return None

    truth = svg_to_mask(svg_path, image.shape[:2])
    true_open = (truth == OUTDOOR).astype(np.uint8)

    recorder = _FrameRecorder(segmenter)
    try:
        with tempfile.TemporaryDirectory(prefix="planto3d_openair_") as workdir:
            result = extract(
                image_path, Path(workdir), segmenter=recorder, crop=False, split=1
            )
    except Exception as error:
        logging.getLogger(__name__).info(
            "%s did not extract: %s", image_path.parent.name, error
        )
        return None

    rooms = [room for floor in result.floors for room in floor.plan.rooms]
    open_rooms = [room for room in rooms if is_open_to_sky(room)]
    undecidable = sum(
        1 for room in rooms if not getattr(room, "label", "")
        and not getattr(room, "category", "")
    )

    frame = recorder.shape or image.shape[:2]
    predicted = _painted(open_rooms, frame)
    if frame != true_open.shape:
        # Nearest, not linear: this is a membership mask and an interpolated
        # edge pixel is not half open to the sky.
        predicted = cv2.resize(
            predicted,
            (true_open.shape[1], true_open.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return PlanScore(
        plan=image_path.parent.name,
        true_open_px=int(true_open.sum()),
        predicted_open_px=int(predicted.sum()),
        intersection_px=int(np.logical_and(true_open, predicted).sum()),
        rooms=len(rooms),
        undecidable=undecidable,
    )


def summarise(scores: list[PlanScore]) -> dict[str, float]:
    """Pool the sheets into one set of figures.

    Pooled over pixels rather than averaged over plans: a plan with one
    small balcony and a plan that is half courtyard are not equally
    informative about how much roof is wrong, and averaging their per-plan
    IoUs pretends they are.

    Every denominator is guarded. A corpus with no OUTDOOR annotation
    anywhere in it is a legitimate corpus -- it simply has nothing to say
    -- and it returns zeros rather than raising.
    """
    true_total = sum(score.true_open_px for score in scores)
    predicted_total = sum(score.predicted_open_px for score in scores)
    intersection = sum(score.intersection_px for score in scores)
    union = true_total + predicted_total - intersection

    rooms = sum(score.rooms for score in scores)
    undecidable = sum(score.undecidable for score in scores)

    return {
        "recall": intersection / max(true_total, 1),
        "precision": intersection / max(predicted_total, 1),
        "iou": intersection / max(union, 1),
        "plans": float(len(scores)),
        "rooms": float(rooms),
        "undecidable_share": undecidable / max(rooms, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=60)
    arguments = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    segmenter = load_segmenter(arguments.checkpoint)

    images = sorted(arguments.root.glob("*/*/F1_scaled.png"))[: arguments.limit]
    print(f"scoring {len(images)} plan(s)\n")
    print(f"{'plan':10}{'true px':>10}{'pred px':>10}{'IoU':>8}"
          f"{'recall':>9}{'rooms':>7}{'undec':>7}")
    print("-" * 61)

    scores: list[PlanScore] = []
    for image_path in images:
        score = score_plan(image_path, image_path.parent / "model.svg", segmenter)
        if score is None:
            print(f"{image_path.parent.name:10}  not scored")
            continue
        scores.append(score)

        one = summarise([score])
        # A sheet the annotation says has no open air is not a failure and
        # not a success; its IoU is undefined and printing 0% would read
        # as one. Only its false positives matter, and those pool upward.
        marker = "" if score.true_open_px else "  <-- no open air drawn"
        print(
            f"{score.plan:10}{score.true_open_px:>10}{score.predicted_open_px:>10}"
            f"{one['iou']:>8.0%}{one['recall']:>9.0%}"
            f"{score.rooms:>7}{score.undecidable:>7}{marker}"
        )

    if not scores:
        print("\nnothing scored")
        return

    pooled = summarise(scores)
    with_truth = [score for score in scores if score.true_open_px]

    print("-" * 61)
    print(f"plans scored        {int(pooled['plans'])}")
    print(f"  with open air     {len(with_truth)}")
    print(f"rooms               {int(pooled['rooms'])}")
    print()
    print(f"IoU                 {pooled['iou']:.1%}   <-- the headline")
    print(f"recall              {pooled['recall']:.1%}")
    print(f"precision           {pooled['precision']:.1%}")
    print()
    # The ceiling, restated every run so no candidate is judged against a
    # target it could never reach: a room with neither a printed label nor
    # a predicted type has no evidence in it for any rule to read.
    print(f"undecidable rooms   {pooled['undecidable_share']:.1%}  "
          f"(neither label nor predicted type)")


if __name__ == "__main__":
    main()
