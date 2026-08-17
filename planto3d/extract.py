"""Turn a segmentation mask into measurable geometry.

Walls are recovered by orientation rather than by contour. Walls in a floor
plan meet at corners, so the wall class forms one connected region per
enclosure -- tracing its contour yields the outline of the whole building,
not individual walls. Opening the mask with a long horizontal kernel keeps
only pixels belonging to a horizontal run, and likewise for vertical, which
separates a closed loop into its constituent segments. Architectural walls
are overwhelmingly axis-aligned, so the two passes cover them; the small
overlap where segments meet at corners is harmless once extruded.

Rooms are contours, simplified so a rectangle comes back as four corners
rather than a few hundred boundary pixels.
"""

import logging

import cv2
import numpy as np

from planto3d.classes import ROOM, WALL
from planto3d.geometry_types import Room, Wall

logger = logging.getLogger(__name__)

# Shortest run of pixels that counts as a wall rather than speckle.
MIN_WALL_LENGTH = 12
# Smallest region that counts as a room.
MIN_ROOM_AREA = 100
# Contour simplification, as a fraction of the contour's perimeter.
SIMPLIFY_TOLERANCE = 0.02


def _segments_along(binary: np.ndarray, horizontal: bool, min_length: int) -> list[Wall]:
    """Extract wall segments running in one direction."""
    kernel_shape = (min_length, 1) if horizontal else (1, min_length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    runs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, labels = cv2.connectedComponents(runs)

    walls: list[Wall] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue

        if horizontal:
            length = xs.max() - xs.min() + 1
            thickness = ys.max() - ys.min() + 1
            centre = (ys.min() + ys.max()) / 2.0
            start = (float(xs.min()), centre)
            end = (float(xs.max()), centre)
        else:
            length = ys.max() - ys.min() + 1
            thickness = xs.max() - xs.min() + 1
            centre = (xs.min() + xs.max()) / 2.0
            start = (centre, float(ys.min()))
            end = (centre, float(ys.max()))

        if length < min_length:
            continue

        walls.append(Wall(start=start, end=end, thickness=float(thickness)))

    return walls


def extract_walls(
    mask: np.ndarray,
    wall_class: int = WALL,
    min_wall_length: int = MIN_WALL_LENGTH,
) -> list[Wall]:
    """Recover wall segments from a class mask, in the mask's pixel coordinates."""
    binary = (mask == wall_class).astype(np.uint8)
    if not binary.any():
        return []

    walls = _segments_along(binary, horizontal=True, min_length=min_wall_length)
    walls += _segments_along(binary, horizontal=False, min_length=min_wall_length)

    logger.info("extracted %d wall segment(s)", len(walls))
    return walls


def extract_rooms(
    mask: np.ndarray,
    room_class: int = ROOM,
    min_area: int = MIN_ROOM_AREA,
    simplify_tolerance: float = SIMPLIFY_TOLERANCE,
) -> list[Room]:
    """Recover room polygons from a class mask, in the mask's pixel coordinates.

    Regions that cannot form a valid polygon are logged and skipped rather
    than raised, so one malformed room does not abort a floor.
    """
    binary = (mask == room_class).astype(np.uint8)
    if not binary.any():
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms: list[Room] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = simplify_tolerance * cv2.arcLength(contour, closed=True)
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        polygon = [(float(p[0][0]), float(p[0][1])) for p in simplified]

        try:
            rooms.append(Room(polygon=polygon))
        except ValueError as error:
            logger.warning("skipping region of area %.0f: %s", area, error)

    logger.info("extracted %d room polygon(s)", len(rooms))
    return rooms
