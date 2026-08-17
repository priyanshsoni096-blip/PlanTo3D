"""Read CubiCasa5K's SVG annotations as five-class training masks.

CubiCasa5K labels over 80 categories; this pipeline needs five. Each sample
folder holds a floor plan image beside a ``model.svg`` whose groups carry a
class attribute -- ``Wall``, ``Door``, ``Window``, and ``Space <RoomType>``
for interiors. Collapsing on the attribute's first token maps every room type
to one ROOM class and discards furniture and fixtures.

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

from planto3d.classes import BACKGROUND, DOOR, NUM_CLASSES, ROOM, WALL, WINDOW

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

# Painted in this order so openings survive where they cross a wall.
PAINT_ORDER = [ROOM, WALL, DOOR, WINDOW]

_TRANSLATE = re.compile(r"translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)")


def _group_class(element: ElementTree.Element) -> int | None:
    """Map an SVG group's class attribute to one of our classes."""
    attribute = element.get("class")
    if not attribute:
        return None
    # "Space Bedroom" and "Wall" alike are keyed on the first token.
    return CLASS_MAP.get(attribute.split()[0])


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
