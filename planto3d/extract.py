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
# Closing the building envelope. The perimeter is walked at this interval,
# a wall counts as present within this distance, and the mask is probed this
# far inside to ask whether a room lies behind the gap.
ENVELOPE_SAMPLE_FT = 1.0
ENVELOPE_NEAR_FT = 2.0
ENVELOPE_PROBE_FT = 2.5
# Shorter gaps than this are doorways and reveals, not missing wall.
ENVELOPE_MIN_GAP_FT = 3.0
ENVELOPE_WALL_FT = 0.75
# Footprint cleanup: bridge doorways, then erase spurs narrower than this.
CLOSE_SPAN = 9
OPEN_SPAN = 25
# Collinear runs closer than this along their line are one wall. Sized to
# span a doorway, which is what usually splits a wall in two.
MERGE_GAP = 90.0
# Runs whose shared coordinate differs by less than this are on one line.
MERGE_OFFSET = 12.0


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


def _merge_collinear(walls: list[Wall], gap: float, offset: float) -> list[Wall]:
    """Join wall segments lying on the same line into single runs.

    Segmentation breaks one continuous wall into several runs wherever a
    doorway, a dimension leader or a patch of noise interrupts it. Extruded
    separately those become abutting boxes with a visible seam at every
    joint, and openings measured along a stub sit at the wrong distance.
    """
    merged: list[Wall] = []

    for horizontal in (True, False):
        axis, along = (1, 0) if horizontal else (0, 1)

        candidates = [
            wall
            for wall in walls
            if (abs(wall.end[1] - wall.start[1]) < abs(wall.end[0] - wall.start[0]))
            == horizontal
        ]

        # Group by the shared coordinate, so only walls on one line combine.
        lanes: dict[int, list[Wall]] = {}
        for wall in candidates:
            key = int(round(wall.start[axis] / max(offset, 1e-6)))
            lanes.setdefault(key, []).append(wall)

        for lane in lanes.values():
            lane.sort(key=lambda w: min(w.start[along], w.end[along]))

            current = lane[0]
            for wall in lane[1:]:
                current_end = max(current.start[along], current.end[along])
                next_start = min(wall.start[along], wall.end[along])

                if next_start - current_end <= gap:
                    next_end = max(wall.start[along], wall.end[along])
                    far = max(current_end, next_end)
                    near = min(current.start[along], current.end[along])
                    shared = (current.start[axis] + wall.start[axis]) / 2

                    start = (near, shared) if horizontal else (shared, near)
                    end = (far, shared) if horizontal else (shared, far)
                    current = Wall(
                        start=start,
                        end=end,
                        thickness=max(current.thickness, wall.thickness),
                    )
                else:
                    merged.append(current)
                    current = wall
            merged.append(current)

    return merged


def extract_walls(
    mask: np.ndarray,
    wall_class: int = WALL,
    min_wall_length: int = MIN_WALL_LENGTH,
    merge: bool = True,
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

    if merge and walls:
        before = len(walls)
        walls = _merge_collinear(walls, gap=MERGE_GAP, offset=MERGE_OFFSET)
        logger.info("merged %d wall run(s) into %d", before, len(walls))
    else:
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


def close_envelope(
    mask: np.ndarray,
    walls: list[Wall],
    footprint: list[tuple[float, float]],
    scale: float,
) -> list[Wall]:
    """Add the exterior walls segmentation missed, closing the building.

    Segmentation loses stretches of wall wherever a drawing is busy, and the
    building comes out with holes in its facade -- on the reference set, only
    two thirds of each storey's perimeter had a wall on it, with gaps as long
    as 43 ft. Those holes also cost every window that would have been bound
    to the missing wall.

    A gap is only filled where the space just inside it is a room. A terrace
    or courtyard edge is legitimately open, and walling it in would enclose
    the very spaces that make the plan what it is.
    """
    if not footprint or scale <= 0:
        return []

    height, width = mask.shape
    step_px = ENVELOPE_SAMPLE_FT * scale
    near_px = ENVELOPE_NEAR_FT * scale
    probe_px = ENVELOPE_PROBE_FT * scale
    centre = np.array(
        [sum(p[0] for p in footprint), sum(p[1] for p in footprint)], dtype=float
    ) / len(footprint)

    added: list[Wall] = []

    for index in range(len(footprint)):
        start = np.array(footprint[index], dtype=float)
        end = np.array(footprint[(index + 1) % len(footprint)], dtype=float)
        span = float(np.linalg.norm(end - start))
        if span < ENVELOPE_MIN_GAP_FT * scale:
            continue

        steps = max(int(span / step_px), 1)
        run_start = None

        for step in range(steps + 1):
            point = start + (end - start) * (step / steps)

            has_wall = walls and min(
                _project_onto_wall(tuple(point), wall)[1] for wall in walls
            ) <= near_px

            # Probe inward: a wall is only wanted where a room lies behind.
            inward = point + (centre - point) / max(
                np.linalg.norm(centre - point), 1e-6
            ) * probe_px
            column, row = int(round(inward[0])), int(round(inward[1]))
            encloses_room = (
                0 <= row < height
                and 0 <= column < width
                and mask[row, column] in (ROOM, WALL)
            )

            needs_wall = not has_wall and encloses_room

            if needs_wall and run_start is None:
                run_start = point
            elif not needs_wall and run_start is not None:
                if float(np.linalg.norm(point - run_start)) >= ENVELOPE_MIN_GAP_FT * scale:
                    added.append(
                        Wall(
                            start=tuple(run_start),
                            end=tuple(point),
                            thickness=ENVELOPE_WALL_FT * scale,
                        )
                    )
                run_start = None

        if run_start is not None:
            if float(np.linalg.norm(end - run_start)) >= ENVELOPE_MIN_GAP_FT * scale:
                added.append(
                    Wall(
                        start=tuple(run_start),
                        end=tuple(end),
                        thickness=ENVELOPE_WALL_FT * scale,
                    )
                )

    if added:
        logger.info("closed %d gap(s) in the building envelope", len(added))
    return added


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
