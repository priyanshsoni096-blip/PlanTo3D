"""Read CubiCasa5K's SVG annotations as training masks.

Each sample folder holds a floor plan image beside a ``model.svg`` whose
groups carry a class attribute -- ``Wall``, ``Door``, ``Window``, and
``Space <RoomType>`` for interiors. Furniture and fixtures are discarded.

The room type is kept rather than collapsed. CubiCasa distinguishes some
forty types and this pipeline needs seven, so ``SPACE_MAP`` groups them by
what they change in the model: a bath and a sauna both want a tiled, wet
floor, so both arrive as BATH. A type absent from the map becomes the
generic ROOM rather than background, because an unrecognised room is still
a room and dropping it would punch a hole in the floor.

Getting this mapping wrong degrades the model in a way that looks like a
training problem rather than a labelling one, which is why it lives here with
tests rather than inside a notebook.

Rasterization order matters: rooms, then walls over them, then openings last.
A door interrupts the wall it sits in, so it has to win where they overlap.
"""

import logging
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import (
    BACKGROUND,
    BATH,
    BEDROOM,
    CIRCULATION,
    DOOR,
    KITCHEN,
    NUM_CLASSES,
    OUTDOOR,
    ROOM,
    STORAGE,
    WALL,
    WINDOW,
)

logger = logging.getLogger(__name__)

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"

# Annotation group class -> our class. Anything absent is background, which
# discards furniture, fixtures, stairs and dimension marks.
CLASS_MAP = {
    "Wall": WALL,
    "Door": DOOR,
    "Window": WINDOW,
    "Space": ROOM,
}

# CubiCasa's room type -> our class, keyed on the first token of the class
# attribute, which is always the primary type: "Outdoor Balcony Glazed" is
# an Outdoor, "Bath Shower" a Bath, "Closet WalkIn" a Closet.
#
# Grouped by what the type changes downstream rather than by architectural
# category. Rooms that take the same floor and the same fittings share a
# class, however different they are to live in -- a study and a living room
# are both ROOM. Where a type genuinely could go either way the habitable
# reading wins, since an over-plain floor is a smaller error than a tiled
# bedroom.
SPACE_MAP = {
    # Habitable rooms of no particular requirement.
    "Undefined": ROOM,
    "UserDefined": ROOM,
    "Room": ROOM,
    "LivingRoom": ROOM,
    "Dining": ROOM,
    "Den": ROOM,
    "RecreationRoom": ROOM,
    "Office": ROOM,
    "Alcove": ROOM,
    "Elevated": ROOM,
    "Basement": ROOM,
    # Sleeping.
    "Bedroom": BEDROOM,
    # Anything with a hob or a sink counts as a kitchen, kitchenettes and
    # sculleries included -- they are tiled the same way.
    "Kitchen": KITCHEN,
    # Wet rooms. A sauna and a pool are not bathrooms, but they share the
    # thing that matters here: a floor built to get wet.
    "Bath": BATH,
    "Sauna": BATH,
    "SwimmingPool": BATH,
    # Service space: unheated, hard-floored, not lived in.
    "Closet": STORAGE,
    "Storage": STORAGE,
    "DressingRoom": STORAGE,
    "Utility": STORAGE,
    "TechnicalRoom": STORAGE,
    "Garage": STORAGE,
    "CarPort": STORAGE,
    # Moving through rather than staying in.
    "Entry": CIRCULATION,
    "Hall": CIRCULATION,
    "DraughtLobby": CIRCULATION,
    # Outside the envelope: balconies, terraces, porches, covered areas.
    # These drive railings and paving, so they matter more than their share
    # of the drawing suggests.
    "Outdoor": OUTDOOR,
}

# Painted in this order so openings survive where they cross a wall. Room
# types go down first and in a fixed order, so that where two spaces overlap
# the result is at least deterministic.
PAINT_ORDER = [
    ROOM,
    BEDROOM,
    KITCHEN,
    BATH,
    STORAGE,
    CIRCULATION,
    OUTDOOR,
    WALL,
    DOOR,
    WINDOW,
]

_TRANSLATE = re.compile(r"translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)")


def _group_class(element: ElementTree.Element) -> int | None:
    """Map an SVG group's class attribute to one of our classes."""
    attribute = element.get("class")
    if not attribute:
        return None

    tokens = attribute.split()
    base = CLASS_MAP.get(tokens[0])
    if base != ROOM:
        return base

    # A space: keep its type. An unrecognised one stays a generic room, so a
    # vocabulary CubiCasa adds later degrades to plain rather than to a hole.
    return SPACE_MAP.get(tokens[1], ROOM) if len(tokens) > 1 else ROOM


def _translation(element: ElementTree.Element) -> tuple[float, float]:
    match = _TRANSLATE.search(element.get("transform", ""))
    return (float(match.group(1)), float(match.group(2))) if match else (0.0, 0.0)


def _points(polygon: ElementTree.Element) -> np.ndarray | None:
    """Parse an SVG polygon's point list into an array."""
    raw = polygon.get("points", "").strip()
    if not raw:
        return None

    numbers = [float(value) for value in re.split(r"[\s,]+", raw) if value]
    if len(numbers) < 6:  # fewer than three points cannot enclose an area
        return None
    return np.array(numbers[: len(numbers) // 2 * 2]).reshape(-1, 2)


def _collect(root: ElementTree.Element) -> dict[int, list[np.ndarray]]:
    """Gather polygons per class, applying any group translation."""
    shapes: dict[int, list[np.ndarray]] = {value: [] for value in PAINT_ORDER}

    for group in root.iter():
        class_index = _group_class(group)
        if class_index is None:
            continue

        offset = np.array(_translation(group))
        for polygon in group.iter(f"{SVG_NAMESPACE}polygon"):
            points = _points(polygon)
            if points is not None:
                shapes[class_index].append(points + offset)
        # Some releases use <path> for spaces; those are skipped rather than
        # guessed at, since a wrong outline is worse than a missing one.

    return shapes


def svg_to_mask(svg_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a CubiCasa ``model.svg`` into a class-index mask.

    ``shape`` is (height, width) and should match the sample's image.
    """
    root = ElementTree.parse(str(svg_path)).getroot()
    shapes = _collect(root)

    # Rasterize into uint8 -- OpenCV cannot fill an int64 array -- then widen
    # to the index dtype the rest of the pipeline expects.
    canvas = np.full(shape, BACKGROUND, dtype=np.uint8)
    for class_index in PAINT_ORDER:
        polygons = [p.round().astype(np.int32) for p in shapes[class_index]]
        if polygons:
            cv2.fillPoly(canvas, polygons, color=int(class_index))

    return canvas.astype(np.int64)


def sample_paths(root: Path, split_file: Path) -> list[tuple[Path, Path]]:
    """Read a CubiCasa split list into (image, annotation) pairs.

    Split files hold one folder per line, such as
    ``/high_quality_architectural/2003/``. Samples missing either file are
    logged and skipped so one bad folder does not abort a training run.
    """
    pairs = []
    missing = 0

    for line in Path(split_file).read_text().splitlines():
        folder = line.strip().strip("/")
        if not folder:
            continue

        directory = Path(root) / folder
        svg = directory / "model.svg"
        image = next(
            (directory / name for name in ("F1_scaled.png", "F1_original.png")
             if (directory / name).is_file()),
            None,
        )
        if image is None or not svg.is_file():
            missing += 1
            continue
        pairs.append((image, svg))

    if missing:
        logger.warning("skipped %d sample(s) missing an image or annotation", missing)
    logger.info("found %d usable sample(s)", len(pairs))
    return pairs


def class_distribution(mask: np.ndarray) -> dict[int, float]:
    """Share of each class in a mask, for spotting a broken remapping."""
    total = mask.size
    return {
        index: float((mask == index).sum()) / total for index in range(NUM_CLASSES)
    }


# --- Ground truth for scale ---------------------------------------------------
#
# CubiCasa records each space's real size inside the annotation, as a hidden
# label reading like ``12'4" x 9'6"``. It is marked ``display: none`` so it
# never renders, which means it is not a shortcut the pipeline could take on
# a real drawing -- but it does let the inferred scale be scored rather than
# assumed correct, which nothing else here allows.
#
# Every plan carries it, so scale accuracy can be measured across the whole
# dataset instead of against the one drawing that prints its dimensions.

# "12'4\"" or "12'" or "12' 4\"". The inches are optional and drafters are
# inconsistent about the space between the two parts.
_FEET_INCHES = re.compile(r"(\d+)\s*'\s*(?:(\d+)\s*\")?")

# A room's stated size is two measurements with a separator between them.
_DIMENSION_PAIR = re.compile(
    r"(\d+\s*'[^x×]*?)\s*[x×]\s*(\d+\s*'.*)", re.IGNORECASE
)

# Ratios this far apart mean the two sides disagree, and the room is not
# usable evidence -- usually an L-shaped space whose bounding box is much
# bigger than the room, or a label belonging to something else.
SCALE_AGREEMENT = 0.12

# Rooms smaller than this in either direction are too small to measure
# reliably: a foot of rounding on a 3 ft cupboard is a third of the answer.
MIN_MEASURABLE_FEET = 4.0


def parse_feet(text: str) -> float | None:
    """Feet as a decimal from a drafting measurement like ``12'4"``."""
    match = _FEET_INCHES.search(text)
    if not match:
        return None
    feet = float(match.group(1))
    return feet + float(match.group(2)) / 12.0 if match.group(2) else feet


def _space_measurements(root: ElementTree.Element) -> list[tuple[np.ndarray, float, float]]:
    """Each space's polygon beside the two lengths its label states."""
    found = []

    for group in root.iter():
        attribute = group.get("class", "")
        if not attribute.startswith("Space"):
            continue

        polygon = next(
            (
                points
                for element in group.iter(f"{SVG_NAMESPACE}polygon")
                if (points := _points(element)) is not None
            ),
            None,
        )
        if polygon is None:
            continue

        # The measurement is nested a few levels down, inside the space's
        # own Dimension group.
        for label in group.iter():
            if "DimensionMeasureLabel" not in label.get("class", ""):
                continue

            text = "".join(label.itertext())
            pair = _DIMENSION_PAIR.search(text)
            if not pair:
                continue

            first, second = parse_feet(pair.group(1)), parse_feet(pair.group(2))
            if first and second:
                found.append((polygon, first, second))
            break

    return found


def ground_truth_scale(svg_path: Path) -> float | None:
    """Pixels per foot, read from the sizes CubiCasa recorded for each room.

    Returns the median over every room that measures consistently, or None
    where the annotation states no sizes. The median rather than the mean
    because one L-shaped room whose bounding box overstates it would drag an
    average well off.
    """
    root = ElementTree.parse(str(svg_path)).getroot()

    estimates = []
    for polygon, first, second in _space_measurements(root):
        if min(first, second) < MIN_MEASURABLE_FEET:
            continue

        width = float(polygon[:, 0].max() - polygon[:, 0].min())
        height = float(polygon[:, 1].max() - polygon[:, 1].min())
        if not (width and height):
            continue

        # The label does not say which measurement is which way round, so
        # both pairings are tried and the consistent one is believed. A room
        # that agrees on neither is not a rectangle and is left out.
        for across, down in ((first, second), (second, first)):
            horizontal, vertical = width / across, height / down
            if abs(horizontal - vertical) / max(horizontal, vertical) < SCALE_AGREEMENT:
                estimates.append((horizontal + vertical) / 2)
                break

    if not estimates:
        return None
    return float(np.median(estimates))
