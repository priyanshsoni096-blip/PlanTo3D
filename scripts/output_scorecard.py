"""Is the finished model right? Not "is each stage right".

Every other script here scores one stage against ground truth -- the mask,
the wall segments, the scale, the split. All of them can pass and still
leave a model nobody would accept, because a plan that is 80% right on six
separate things is not 80% of a house. This asks the question none of them
ask: **how many plans come out right on every count at once.**

That headline is deliberately harsh, and it is the point. A per-check
failure count comes with it, and that is what should order the work: the
check that fails most is the binding constraint on the output, whatever
the individual stage metrics say about it.

Everything is scored against CubiCasa's own annotations rather than by
eye, so it runs over sixty sheets and gives the same answer twice. The
cost of that is real and worth stating: it cannot see a balcony sealed
under a slab, geometry floating in mid-air, or a roof with a hole in it.
Those need looking at, and looking at them is a separate exercise -- not
one to fold in here, where it would make the number unrepeatable.

    python scripts/output_scorecard.py <corpus> --checkpoint models/unet_cubicasa.pt
"""

import argparse
import logging
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from planto3d.classes import ROOM_CLASSES, WALL  # noqa: E402
from planto3d.cubicasa import svg_to_mask  # noqa: E402
from planto3d.extract import MIN_ROOM_AREA_RATIO, wall_gauge  # noqa: E402
from planto3d.pipeline import run  # noqa: E402
from planto3d.segment import load_segmenter  # noqa: E402
from scale_accuracy import true_scale  # noqa: E402
from split_accuracy import true_floors  # noqa: E402
from wall_accuracy import score as wall_score  # noqa: E402

# Thresholds. These are first guesses, not derived from anything, and they
# are the weakest part of this script -- a check is only as meaningful as
# the line it draws. Revise them once the distribution is visible, and do
# not treat them as settled because they appear in code.
SCALE_TOLERANCE = 0.20        # a fifth, matching scripts/scale_accuracy.py
ROOM_COUNT_TOLERANCE = 0.25   # rooms are merged and split freely by both sides
MIN_WALL_COVERAGE = 0.85
MIN_WALL_AGREEMENT = 0.80
MIN_OPENING_RATIO = 0.6       # a band: too few is a bare facade,
MAX_OPENING_RATIO = 1.5       # too many is holes that are not there

CHECKS = ("built", "storeys", "size", "rooms", "walls", "openings")


def annotated_rooms(truth: np.ndarray, gauge: float) -> int:
    """How many rooms the annotation draws, counted the way we count ours.

    Per class, not over their union. Taking components of every room class
    at once merges an open-plan floor into a single region -- it scored a
    four-room flat as two -- and the pipeline does not count that way
    either: ``extract_rooms`` walks one class at a time.
    """
    minimum = max(int(MIN_ROOM_AREA_RATIO * gauge**2), 16)
    rooms = 0
    for class_index in sorted(ROOM_CLASSES):
        binary = (truth == class_index).astype(np.uint8)
        if not binary.any():
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        rooms += sum(
            1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum
        )
    return rooms


def judge(path: Path, segmenter) -> dict[str, bool] | None:
    """Every check for one sheet. None when the sheet cannot be scored."""
    annotation = path.parent / "model.svg"
    image = cv2.imread(str(path))
    if image is None or not annotation.is_file():
        return None

    truth = svg_to_mask(annotation, image.shape[:2])
    if not (truth == WALL).any():
        return None

    verdict = dict.fromkeys(CHECKS, False)

    try:
        with tempfile.TemporaryDirectory() as workdir:
            result = run(path, Path(workdir), segmenter=segmenter)
            verdict["built"] = result.model_path is not None
    except Exception as error:  # a crash is a failed plan, not a failed run
        logging.getLogger(__name__).info("%s did not build: %s", path.parent.name, error)
        return verdict

    verdict["storeys"] = len(result.floors) == true_floors(annotation)

    wanted = true_scale(annotation, path)
    if wanted and result.scale:
        verdict["size"] = abs(result.scale - wanted) / wanted <= SCALE_TOLERANCE

    expected = annotated_rooms(truth, wall_gauge(truth))
    got = result.room_count
    if expected:
        verdict["rooms"] = abs(got - expected) / expected <= ROOM_COUNT_TOLERANCE

    # Only on a sheet that stayed whole. Once it is split, each piece's
    # geometry is in its own frame with its own origin, and painting them
    # all onto the original sheet's annotation compares nothing.
    if len(result.floors) == 1:
        measured = wall_score(truth, result.floors[0].plan.walls)
        if measured:
            coverage, agreement = measured
            verdict["walls"] = (
                coverage >= MIN_WALL_COVERAGE and agreement >= MIN_WALL_AGREEMENT
            )
    else:
        verdict["walls"] = True  # not judged rather than failed

    # Roughly as many openings as the drawing draws -- a band, not a floor.
    # A floor is passed by over-reporting, and openings are over-reported:
    # precision on windows is 43%, so "at least as many as drawn" rewards
    # exactly the failure it should catch.
    from planto3d.classes import DOOR, WINDOW

    drawn = 0
    for class_index in (DOOR, WINDOW):
        binary = (truth == class_index).astype(np.uint8)
        if binary.any():
            drawn += cv2.connectedComponentsWithStats(binary, 8)[0] - 1
    if drawn:
        ratio = result.opening_count / drawn
        verdict["openings"] = MIN_OPENING_RATIO <= ratio <= MAX_OPENING_RATIO

    return verdict


def main(root: str, checkpoint: Path | None, limit: int) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    segmenter = load_segmenter(checkpoint)

    rows: list[tuple[str, dict[str, bool]]] = []
    for path in sorted(Path(root).glob("*/*/F1_scaled.png"))[:limit]:
        verdict = judge(path, segmenter)
        if verdict is not None:
            rows.append((path.parent.name, verdict))

    if not rows:
        print("no plans scored")
        return

    print(f"{len(rows)} plans\n")
    header = "".join(f"{name:>10}" for name in CHECKS)
    print(f"{'plan':10}{header}{'all':>7}")
    print("-" * (10 + 10 * len(CHECKS) + 7))
    for name, verdict in rows:
        marks = "".join(f"{('pass' if verdict[c] else '.'):>10}" for c in CHECKS)
        print(f"{name:10}{marks}{('yes' if all(verdict.values()) else 'no'):>7}")

    passed = sum(1 for _, v in rows if all(v.values()))
    print()
    print(f"correct on every count: {passed}/{len(rows)}  ({passed/len(rows):.0%})")
    print()
    print("what fails, worst first — this is what should order the work:")
    failures = sorted(
        ((c, sum(1 for _, v in rows if not v[c])) for c in CHECKS),
        key=lambda item: -item[1],
    )
    for check, count in failures:
        print(f"   {check:12}{count:>4} of {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.limit)
