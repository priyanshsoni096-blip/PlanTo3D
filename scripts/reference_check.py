"""Check a built model against a real building's measured dimensions.

Every other score in this project is against an annotator's drawing. This is
the one that is not: measurements taken off an actual building, compared with
what the pipeline built from that building's plan. The spec's second gate
lives here -- overall size within 5%, each room within 10%.

Measurements go in a plain text file, in the same syntax the correction
flags already use, so there is nothing new to learn:

    # my house, measured with a tape
    overall=40x60        # the ground-floor building outline, in feet
    1:5=12.5x14          # floor 1, room 5, as correct_and_build.py --list numbers it
    2:0=10x10

Feet as decimals: 12'6" is 12.5. Which side is the width does not matter.
Take room numbers from a --list run on the same plan with the same options.

Keep that file somewhere git ignores -- for the reference house,
data/soni_residence/truth.txt. Nothing it contains is ever printed to a
commit; only this script is.

    python scripts/reference_check.py plan.pdf data/soni_residence/truth.txt --checkpoint models/unet_cubicasa.pt
"""

import argparse
import logging
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

from scripts.correct_and_build import parse_scale_room

# The spec's Gate 2, verbatim: "within 5% on overall size and within 10% on
# individual rooms".
OVERALL_TOLERANCE = 0.05
ROOM_TOLERANCE = 0.10


@dataclass
class Truth:
    overall: tuple[float, float] | None
    rooms: dict[tuple[int, int], tuple[float, float]]


@dataclass
class Row:
    name: str
    truth: tuple[float, float]
    measured: tuple[float, float] | None
    error: float | None
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.error is not None and self.error <= self.tolerance


def read_truth(lines: list[str]) -> Truth:
    """Read measurements, refusing any line it cannot understand.

    A measurement that silently does nothing would report a pass on a room
    nobody checked, so a bad line stops the run rather than being skipped.
    """
    overall = None
    rooms = {}
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if text.lower().startswith("overall="):
            size = text.split("=", 1)[1]
            try:
                width, depth = (float(part) for part in size.lower().split("x", 1))
            except ValueError:
                raise ValueError(
                    f"could not read {text!r}; expected overall=WxD in feet, "
                    "for example overall=40x60"
                ) from None
            overall = (width, depth)
            continue
        where, width, height = parse_scale_room(text)
        rooms[where] = (width, height)
    return Truth(overall=overall, rooms=rooms)


def size_error(measured: tuple[float, float], truth: tuple[float, float]) -> float:
    """Worst relative error of the two sides, ignoring orientation."""
    pairs = zip(sorted(measured), sorted(truth))
    return max(abs(m - t) / t for m, t in pairs)


def measured_size(room, scale: float) -> tuple[float, float]:
    left, top, right, bottom = room.bounds()
    return ((right - left) / scale, (bottom - top) / scale)


def _outline_size(points, scale: float) -> tuple[float, float] | None:
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return ((max(xs) - min(xs)) / scale, (max(ys) - min(ys)) / scale)


def evaluate(result, truth: Truth) -> list[Row]:
    rows = []
    if truth.overall is not None:
        measured = _outline_size(result.floors[0].plan.footprint, result.scale)
        rows.append(Row(
            name="overall",
            truth=truth.overall,
            measured=measured,
            error=None if measured is None else size_error(measured, truth.overall),
            tolerance=OVERALL_TOLERANCE,
        ))

    for (floor, index), size in sorted(truth.rooms.items()):
        measured = None
        if floor < len(result.floors) and index < len(result.floors[floor].plan.rooms):
            measured = measured_size(result.floors[floor].plan.rooms[index], result.scale)
        rows.append(Row(
            name=f"{floor + 1}:{index}",
            truth=size,
            measured=measured,
            error=None if measured is None else size_error(measured, size),
            tolerance=ROOM_TOLERANCE,
        ))
    return rows


def _size(pair) -> str:
    return "--" if pair is None else f"{pair[0]:.1f} x {pair[1]:.1f}"


def main(source: str, truth_path: str, checkpoint: Path | None, crop: bool) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    from planto3d.pipeline import extract
    from planto3d.segment import load_segmenter

    truth = read_truth(Path(truth_path).read_text(encoding="utf-8").splitlines())
    with tempfile.TemporaryDirectory(prefix="planto3d_reference_") as workdir:
        result = extract(Path(source), Path(workdir), segmenter=load_segmenter(checkpoint), crop=crop)

    rows = evaluate(result, truth)
    print(f"scale {result.scale:.2f} px/ft, from {result.scale_source}\n")
    print(f"{'item':8} {'real (ft)':>14} {'model (ft)':>14} {'error':>7} {'gate':>6}  result")
    print("-" * 62)
    for row in rows:
        error = "--" if row.error is None else f"{row.error:.1%}"
        verdict = "pass" if row.passed else ("missing" if row.measured is None else "FAIL")
        print(f"{row.name:8} {_size(row.truth):>14} {_size(row.measured):>14} "
              f"{error:>7} {row.tolerance:>6.0%}  {verdict}")
    print("-" * 62)
    passed = sum(row.passed for row in rows)
    print(f"{passed} of {len(rows)} within the spec's gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="the plan: PDF, image, or directory of images")
    parser.add_argument("truth", help="measurements file; keep it git-ignored")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-crop", action="store_true")
    arguments = parser.parse_args()
    main(arguments.source, arguments.truth, arguments.checkpoint, crop=not arguments.no_crop)
