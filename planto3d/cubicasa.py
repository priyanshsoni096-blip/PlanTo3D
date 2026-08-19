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
