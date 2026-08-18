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

from planto3d.classes import DOOR, ROOM, WALL, WINDOW
from planto3d.geometry_types import Opening, Room, Wall

logger = logging.getLogger(__name__)

# Shortest run of pixels that counts as a wall rather than speckle.
MIN_WALL_LENGTH = 12
# Smallest region that counts as a room.
MIN_ROOM_AREA = 100
# Contour simplification, in pixels of allowed deviation.
#
# Deliberately absolute rather than a fraction of perimeter. A relative
# tolerance scales with the outline's size, so the building footprint -- by
# far the longest contour -- gets the coarsest treatment exactly where
# precision matters most: at 2% of perimeter it cut diagonal shortcuts across
# whole corners, turning a rectilinear building into a jagged wedge.
SIMPLIFY_PIXELS = 4.0
# Smallest door or window blob worth trusting.
MIN_OPENING_AREA = 40
# An opening further than this from any wall is dropped rather than bound to
# a distant one -- a misplaced opening cuts a hole through solid geometry.
MAX_OPENING_DISTANCE = 40.0
# Footprint cleanup: bridge doorways, then erase spurs narrower than this.
CLOSE_SPAN = 9
OPEN_SPAN = 25


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
    """Recover wall segments from a class mask, in the mask's pixel coordinates.

    Walls come out exactly axis-aligned: each run's endpoints are placed on
    its bounding box centreline, so segmentation noise along an edge cannot
    tilt the result. The cost is that a genuinely diagonal wall is not
    recovered at all -- the orientation filters erase it -- which is
    acceptable for the rectilinear plans this targets.
    """
    binary = (mask == wall_class).astype(np.uint8)
    if not binary.any():
        return []

    walls = _segments_along(binary, horizontal=True, min_length=min_wall_length)
    walls += _segments_along(binary, horizontal=False, min_length=min_wall_length)

    logger.info("extracted %d wall segment(s)", len(walls))
    return walls


def _project_onto_wall(point: tuple[float, float], wall: Wall) -> tuple[float, float]:
    """Distance along the wall to the point's projection, and distance from it."""
    start = np.array(wall.start)
    direction = np.array(wall.end) - start
    length = float(np.linalg.norm(direction))
    if length == 0:
        return 0.0, float(np.linalg.norm(np.array(point) - start))

    unit = direction / length
    offset = np.array(point) - start
    along = float(np.clip(np.dot(offset, unit), 0.0, length))
    perpendicular = float(np.linalg.norm(offset - along * unit))
    return along, perpendicular


def extract_openings(
    mask: np.ndarray,
    walls: list[Wall],
    min_area: int = MIN_OPENING_AREA,
    max_distance: float = MAX_OPENING_DISTANCE,
) -> list[Opening]:
    """Find doors and windows and bind each to the wall it interrupts.

    An opening is only meaningful relative to its wall, so a component that
    sits too far from any wall is discarded rather than attached to a distant
    one -- a misplaced opening cuts a hole through solid geometry.
    """
    if not walls:
        return []

    openings: list[Opening] = []

    for class_index, opening_type in ((DOOR, "door"), (WINDOW, "window")):
        binary = (mask == class_index).astype(np.uint8)
        if not binary.any():
            continue

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < min_area:
                continue

            centre = (float(centroids[index][0]), float(centroids[index][1]))
            projections = [
                (*_project_onto_wall(centre, wall), wall_id)
                for wall_id, wall in enumerate(walls)
            ]
            along, distance, wall_id = min(projections, key=lambda item: item[1])

            if distance > max_distance:
                logger.debug("dropping %s %.0fpx from any wall", opening_type, distance)
                continue

            # Width along the wall, from the component's larger extent -- an
            # opening is drawn as a strip across the wall's thickness.
            width = float(
                max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT])
            )
            openings.append(
                Opening(wall_id=wall_id, position=along, width=width, type=opening_type)
            )

    logger.info(
        "extracted %d opening(s): %d door(s), %d window(s)",
        len(openings),
        sum(1 for o in openings if o.type == "door"),
        sum(1 for o in openings if o.type == "window"),
    )
    return openings


def extract_footprint(
    mask: np.ndarray,
    simplify_pixels: float = SIMPLIFY_PIXELS,
) -> list[tuple[float, float]]:
    """Outline of the built area, for generating floor slabs and a roof.

    Walls and rooms together make up the storey's extent, so the outer
    contour of both classes traces its footprint. Returns an empty list when
    the mask holds no building.
    """
    built = ((mask == WALL) | (mask == ROOM)).astype(np.uint8)
    if not built.any():
        return []

    # Bridge doorways and joints so the storey reads as one region.
    closed = cv2.morphologyEx(built, cv2.MORPH_CLOSE, np.ones((CLOSE_SPAN, CLOSE_SPAN), np.uint8))

    # Erode away thin spurs, then restore the body. Segmentation bleeds into
    # landscaping and paving around the building, and those tendrils otherwise
    # end up in the outline -- producing slabs with jagged fingers reaching
    # out over open ground.
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((OPEN_SPAN, OPEN_SPAN), np.uint8))
    if not opened.any():
        opened = closed

    # Keep only the main mass, so a detached patch of paving cannot drag the
    # outline out across the site.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        opened = (labels == largest).astype(np.uint8)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    outer = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(outer, simplify_pixels, closed=True)
    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


def extract_rooms(
    mask: np.ndarray,
    room_class: int = ROOM,
    min_area: int = MIN_ROOM_AREA,
    simplify_pixels: float = SIMPLIFY_PIXELS,
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

        simplified = cv2.approxPolyDP(contour, simplify_pixels, closed=True)
        polygon = [(float(p[0][0]), float(p[0][1])) for p in simplified]

        try:
            rooms.append(Room(polygon=polygon))
        except ValueError as error:
            logger.warning("skipping region of area %.0f: %s", area, error)

    logger.info("extracted %d room polygon(s)", len(rooms))
    return rooms
