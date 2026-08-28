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
        polygons = [p.round().astype(np.int32) for p in shapes.get(class_index, [])]
        if polygons:
            cv2.fillPoly(canvas, polygons, color=int(class_index))

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
