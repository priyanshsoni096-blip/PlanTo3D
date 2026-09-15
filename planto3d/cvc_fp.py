"""Read the CVC-FP floor plan database into the pipeline's class scheme.

A second drafting tradition, which is the single biggest thing every
figure in `docs/AUDIT.md` was missing. CVC-FP is 122 scanned plans in four
subsets that differ deliberately in origin, drawing style, quality and
resolution -- against CubiCasa5K's 5,000 sheets from one Finnish source.
It is forty times smaller and much harder, which is the point.

What it can and cannot settle, stated up front because the difference
decides what it is worth running:

    can    walls, rooms, doors and windows, scored against a real second
           convention rather than a transform of the first
    cannot **scale** -- there is no metric ground truth anywhere in the
           122 annotations, so the largest failure on the end-to-end
           scorecard is exactly the one this corpus cannot judge
    cannot room function -- every space is labelled "Room" with no type
    cannot storey splitting -- sheets carry no floor grouping

The annotation is flat SVG: polygons carrying a ``class`` attribute, with
separate ``<relation>`` elements recording how the objects connect, which
nothing here uses.

    from planto3d.cvc_fp import sample_paths, svg_to_mask
"""

import logging
import random
import re
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

from planto3d.classes import BACKGROUND, DOOR, OUTDOOR, ROOM, WALL, WINDOW

logger = logging.getLogger(__name__)

# CVC-FP's vocabulary onto ours. Its room class carries no type, so every
# space becomes the generic ROOM -- which is honest: the drawing set says
# nothing about what its rooms are for, and inventing a type would make
# this corpus look like it validates something it does not.
#
# "Parking" is the one space it does name, and maps to OUTDOOR the same way
# a parking bay does everywhere else in this pipeline.
#
# "Separation" is deliberately absent. It marks where one room gives onto
# another with nothing built between them -- an opening in the plan's
# logic rather than an object on the page. Painting it as wall would
# invent walls that are not there and painting it as a door would invent
# doors; leaving it unpainted lets the rooms meet, which is what it means.
CLASS_MAP = {
    "Room": ROOM,
    "Parking": OUTDOOR,
    "Wall": WALL,
    "Door": DOOR,
    "Window": WINDOW,
}

# Rooms first and openings last, so a door reads as a hole in its wall
# rather than the wall swallowing it. Matches `cubicasa.PAINT_ORDER`.
PAINT_ORDER = [ROOM, OUTDOOR, WALL, DOOR, WINDOW]

_POINT = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
_GT_SUFFIX = re.compile(r"_gt_\d+$")


def _polygons(root) -> dict[int, list[np.ndarray]]:
    """Every polygon, grouped by the class index it maps onto."""
    found: dict[int, list[np.ndarray]] = {}
    for element in root.iter():
        if not element.tag.endswith("polygon"):
            continue
        index = CLASS_MAP.get(element.get("class", ""))
        if index is None:
            continue
        points = [
            (float(x), float(y)) for x, y in _POINT.findall(element.get("points", ""))
        ]
        if len(points) >= 3:
            found.setdefault(index, []).append(np.asarray(points, dtype=np.float32))
    return found


def annotation_size(svg_path: Path) -> tuple[int, int] | None:
    """The (height, width) the annotation was drawn against, if it says.

    CVC-FP records these as plain ``<width>`` and ``<height>`` elements
    rather than on the ``svg`` tag, and they are worth checking: a mask
    rasterized at a different size than the coordinates assume is silently
    wrong everywhere rather than obviously wrong somewhere.
    """
    try:
        root = ElementTree.parse(str(svg_path)).getroot()
    except ElementTree.ParseError as error:
        logger.warning("could not parse %s: %s", svg_path.name, error)
        return None

    found = {}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in ("width", "height") and element.text:
            try:
                found[tag] = int(float(element.text))
            except ValueError:
                return None
    if "width" in found and "height" in found:
        return found["height"], found["width"]
    return None


def svg_to_mask(svg_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a CVC-FP annotation into a class-index mask.

    ``shape`` is (height, width) and should match the plan image. Where the
    annotation states a different size the polygons are scaled onto the
    image's, since the two disagree on a handful of sheets.
    """
    root = ElementTree.parse(str(svg_path)).getroot()
    shapes = _polygons(root)

    stated = annotation_size(svg_path)
    if stated and stated != shape and stated[0] > 0 and stated[1] > 0:
        scale_y = shape[0] / stated[0]
        scale_x = shape[1] / stated[1]
        for polygons in shapes.values():
            for polygon in polygons:
                polygon[:, 0] *= scale_x
                polygon[:, 1] *= scale_y

    canvas = np.full(shape, BACKGROUND, dtype=np.uint8)
    for class_index in PAINT_ORDER:
        # One call per polygon. fillPoly treats polygons passed together as
        # the contours of one shape, so where two overlap the overlap is
        # left unfilled. CVC-FP has no repeated polygons, but wall pieces
        # overlap at corners: filled together, 468,747 pixels of wall across
        # the 122 plans came out as background, 1.04% of all wall.
        # `cubicasa.svg_to_mask` had the same fault and a far worse result.
        for polygon in shapes.get(class_index, []):
            cv2.fillPoly(canvas, [polygon.round().astype(np.int32)], color=int(class_index))

    return canvas.astype(np.int64)


def sample_paths(root: Path) -> list[tuple[Path, Path]]:
    """Pair each plan image with its annotation.

    CVC-FP names them ``10.png`` beside ``10_gt_9.svg``, so the stem up to
    ``_gt_`` is the image it belongs to. Extensions vary: most plans are
    PNG and a few are JPEG.
    """
    root = Path(root)
    folder = root / "ImagesGT" if (root / "ImagesGT").is_dir() else root

    images = {
        path.stem: path
        for path in folder.iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg")
    }

    pairs, missing = [], 0
    for svg in sorted(folder.glob("*.svg")):
        stem = _GT_SUFFIX.sub("", svg.stem)
        image = images.get(stem)
        if image is None:
            missing += 1
            continue
        pairs.append((image, svg))

    if missing:
        logger.warning("%d annotation(s) had no matching image", missing)
    logger.info("found %d CVC-FP sample(s)", len(pairs))
    return pairs


# --- training on it, and keeping a test set back ------------------------------
#
# Training on CVC-FP spends the only second convention there is, so a share of
# it is held back and never trained on. The held-back plans are committed as a
# list, like data/cubicasa_test60.txt, so every later figure is measured on the
# same sheets and training can refuse them by name.
TEST_LIST = Path(__file__).resolve().parents[1] / "data" / "cvc_fp_test.txt"
SPLIT_SEED = 20260915
TEST_SHARE = 1 / 3

_SUBSET = re.compile(r"^(I{1,3}[a-d])_")


def subset(name: str) -> str:
    """Which of CVC-FP's drawing sets a plan belongs to, from its file name.

    The sets differ deliberately in origin and style, so a test set has to
    draw from every one of them.
    """
    match = _SUBSET.match(name)
    if match:
        return match.group(1)
    if name.startswith("image"):
        return "image"
    if re.fullmatch(r"p\d+", name):
        return "p"
    if name.isdigit():
        return "numbered"
    return "other"


def split_names(
    names: list[str], test_share: float = TEST_SHARE, seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """(train, test), a seeded share of every drawing set held back for test.

    Sets are drawn in sorted order from one generator, and each set's names
    are sorted first, so the same names and seed give the same split in any
    order. A set of one plan goes to training: it cannot be both taught and
    tested, and one sheet is not a measurement.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(subset(name), []).append(name)

    rng = random.Random(seed)
    train, test = [], []
    for key in sorted(groups):
        members = sorted(groups[key])
        count = max(1, round(len(members) * test_share)) if len(members) > 1 else 0
        held = set(rng.sample(members, count))
        test += [name for name in members if name in held]
        train += [name for name in members if name not in held]
    return sorted(train), sorted(test)


def read_names(path: Path) -> list[str]:
    """Plan names from a list file, one per line."""
    return sorted(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
