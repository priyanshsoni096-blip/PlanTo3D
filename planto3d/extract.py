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

from planto3d.classes import (
    CLASS_NAMES,
    DOOR,
    ROOM,
    ROOM_CLASSES,
    WALL,
    WINDOW,
)
from planto3d.geometry_types import Opening, Room, Wall

logger = logging.getLogger(__name__)

# --- how big things are, relative to the drawing itself ----------------------
#
# These were absolute pixel counts, measured on drawings around 28-30 pixels
# per foot. That is a resolution, not a property of any building, and it made
# the pipeline quietly dependent on one: the same CubiCasa plan reconstructed
# with 15 walls at half size and 40 at double, and its inferred scale went
# from 8% out to 48% out. A plan is the same building whatever size it is
# rendered at, and it now reconstructs that way.
#
# Everything is a multiple of the drawing's own wall thickness -- the one
# length a floor plan always contains, is drawn to scale, and can be
# measured from before anything else is known. A doorway is about three and
# a half wall thicknesses wide whatever the drawing's resolution.
#
# The ratios are the old constants divided by the gauge they were measured
# at, so behaviour at that resolution is unchanged.
REFERENCE_GAUGE = 24.0

# Shortest run that counts as a wall rather than speckle. This one also
# sizes the morphological kernel that separates horizontal runs from
# vertical ones, so it cannot simply shrink with the drawing: below about
# a dozen pixels the opening stops decomposing a corner at all and a
# four-sided room comes back as one ring-shaped "wall". The floor is what
# that stage needs to work; the ratio is what carries it up to larger
# sheets, which is the direction that was actually broken.
MIN_WALL_LENGTH_RATIO = 0.5
MIN_WALL_LENGTH_FLOOR = 12
# How much longer than a wall is thick the severing kernel must be. Above
# one it erases the perpendicular walls; comfortably above it also survives
# the gauge reading a little high at junctions, which it does.
SEVER_RATIO = 2.0
# Smallest region that counts as a room, as a multiple of the gauge squared.
MIN_ROOM_AREA_RATIO = 100 / REFERENCE_GAUGE**2
# Contour simplification, in allowed deviation.
#
# Deliberately not a fraction of perimeter. A relative tolerance scales with
# the outline's size, so the building footprint -- by far the longest
# contour -- gets the coarsest treatment exactly where precision matters
# most: at 2% of perimeter it cut diagonal shortcuts across whole corners,
# turning a rectilinear building into a jagged wedge. Against the wall
# thickness it stays fixed relative to the building's own detail.
SIMPLIFY_RATIO = 4.0 / REFERENCE_GAUGE
# Smallest door or window blob worth trusting.
MIN_OPENING_AREA_RATIO = 40 / REFERENCE_GAUGE**2
# An opening further than this from any wall is dropped rather than bound to
# a distant one -- a misplaced opening cuts a hole through solid geometry.
MAX_OPENING_DISTANCE_RATIO = 40.0 / REFERENCE_GAUGE
# Footprint cleanup: bridge doorways, then erase spurs narrower than this.
CLOSE_SPAN_RATIO = 9 / REFERENCE_GAUGE
OPEN_SPAN_RATIO = 25 / REFERENCE_GAUGE
# Collinear runs closer than this along their line are one wall. Sized to
# span a doorway, which is what usually splits a wall in two.
MERGE_GAP_RATIO = 90.0 / REFERENCE_GAUGE
# Runs whose shared coordinate differs by less than this are on one line.
MERGE_OFFSET_RATIO = 12.0 / REFERENCE_GAUGE

# The values at the reference gauge, for callers that have no mask to
# measure and for the tests that pinned the old behaviour.
MIN_WALL_LENGTH = int(MIN_WALL_LENGTH_RATIO * REFERENCE_GAUGE)
MIN_ROOM_AREA = int(MIN_ROOM_AREA_RATIO * REFERENCE_GAUGE**2)
SIMPLIFY_PIXELS = SIMPLIFY_RATIO * REFERENCE_GAUGE
MIN_OPENING_AREA = int(MIN_OPENING_AREA_RATIO * REFERENCE_GAUGE**2)
MAX_OPENING_DISTANCE = MAX_OPENING_DISTANCE_RATIO * REFERENCE_GAUGE
CLOSE_SPAN = int(CLOSE_SPAN_RATIO * REFERENCE_GAUGE)
OPEN_SPAN = int(OPEN_SPAN_RATIO * REFERENCE_GAUGE)
MERGE_GAP = MERGE_GAP_RATIO * REFERENCE_GAUGE
MERGE_OFFSET = MERGE_OFFSET_RATIO * REFERENCE_GAUGE

# Closing the building envelope. These are in feet rather than pixels
# because this stage runs after calibration and knows the scale, which is
# the better reference wherever it is available -- a doorway is 2'6" on
# every drawing ever made.
ENVELOPE_SAMPLE_FT = 1.0
ENVELOPE_NEAR_FT = 2.0
ENVELOPE_PROBE_FT = 2.5
# Shorter gaps than this are doorways and reveals, not missing wall.
ENVELOPE_MIN_GAP_FT = 3.0
ENVELOPE_WALL_FT = 0.75

# A drawing whose walls measure outside this range of the reference is
# treated as unmeasurable and given the reference gauge instead. Both ends
# are far outside anything a real plan produces; the guard is against a
# mask so poor that the measurement means nothing.
GAUGE_LIMITS = (4.0, 240.0)


# Below this many walls the median is not worth trusting, so nothing is
# thrown away. A handful of runs on a sparse drawing could easily be
# mostly hatching, and taking their median would then discard the walls
# and keep the hatching.
MIN_WALLS_TO_JUDGE = 6

# The most of a drawing this may remove. Beyond it the reference was
# measuring the wrong thing, and the drawing is better served by its
# original runs than by a fraction of them. Set tight: on a plan where
# short noise fragments outnumbered the walls, allowing half to go took
# the real walls with them and left a median wall thickness of two
# pixels, which put the building at a thirtieth of its size.
MAX_DROPPED_FRACTION = 0.15


# A wall is longer than it is thick. Anything squarer than this is a blob
# -- a hatched panel, a stair core, the whole building caught by one
# orientation pass -- and however much of the drawing it covers it says
# nothing about how thick a wall is.
MIN_WALL_ASPECT = 1.5


def _typical_thickness(walls: list[Wall]) -> float:
    """The drawing's own wall thickness.

    Two things have to be kept out and they pull in opposite directions.

    Short specks of noise outnumber real walls on a poor mask, so a plain
    median reports a thickness below any real one. Weighting by length
    fixes that -- it says what most of the *drawn wall* measures rather
    than what most of the *runs* measure.

    But weighting by length alone hands the answer to a single enormous
    blob, which is long as well as thick: on one CubiCasa plan the whole
    building was caught as one run and reported a wall thickness of 2048
    pixels. So blobs are excluded first, on the one thing that separates a
    wall from a panel -- a wall is much longer than it is thick.
    """
    slender = [
        wall
        for wall in walls
        if wall.thickness > 0 and wall.length() >= wall.thickness * MIN_WALL_ASPECT
    ]
    # If nothing is slender the drawing is all blobs, and its own runs are
    # a better answer than none.
    ordered = sorted(slender or walls, key=lambda wall: wall.thickness)

    lengths = np.array([max(wall.length(), 1.0) for wall in ordered])
    running = np.cumsum(lengths)
    middle = int(np.searchsorted(running, running[-1] / 2.0))
    return float(ordered[min(middle, len(ordered) - 1)].thickness)


# Least wall a drawing must carry before its gauge is worth measuring, as
# a share of the page. Below it the mask is mostly background and the
# measurement says more about noise than about the building.
MIN_WALL_SHARE = 0.002


def wall_gauge(mask: np.ndarray, wall_class: int = WALL) -> float:
    """The drawing's own wall thickness in pixels.

    The one length a floor plan always contains, always draws to scale, and
    can be measured before anything else is known -- there is no room list
    yet, no calibration, and nothing saying what the sheet's resolution
    means. Every other size in this module is expressed against it, so the
    same building reconstructs the same way whatever size it is rendered
    at.

    Measured from the distance transform rather than from wall runs.
    Distance-to-background across a wall of thickness ``t`` is spread
    evenly over ``0`` to ``t/2``, so the median sits at ``t/4``. It is the
    right tool because it asks every wall pixel the same local question and
    weights nothing by length or area, which is what defeated the earlier
    attempts: measuring runs, a drawing with more specks than walls
    reported a thickness below any real one, and weighting those runs by
    length instead handed the answer to whichever blob was biggest -- one
    plan came back with a 2048 pixel wall.

    Falls back to the reference gauge where the mask holds too little wall
    to measure, or the answer is outside anything a real drawing produces.
    A wrong gauge is worse than the reference: it rescales every threshold
    at once.
    """
    binary = (mask == wall_class).astype(np.uint8)
    if binary.mean() < MIN_WALL_SHARE:
        logger.info("too little wall to measure a gauge; using the reference")
        return REFERENCE_GAUGE

    distances = cv2.distanceTransform(binary, cv2.DIST_L2, 5)[binary > 0]
    gauge = 4.0 * float(np.median(distances))

    low, high = GAUGE_LIMITS
    if not low <= gauge <= high:
        logger.info("wall gauge %.1f px is implausible; using the reference", gauge)
        return REFERENCE_GAUGE

    logger.info("wall gauge %.1f px (reference %.0f)", gauge, REFERENCE_GAUGE)
    return gauge


def _segments_along(
    binary: np.ndarray,
    horizontal: bool,
    min_length: int,
    sever: int | None = None,
) -> list[Wall]:
    """Extract wall segments running in one direction.

    The opening has one job: erase the walls running the other way, so what
    is left can be split into separate runs. For that its kernel has to be
    longer than those walls are thick -- a horizontal kernel shorter than a
    vertical wall's width slides along inside it and keeps it.

    That is what ``sever`` sets, and it is separate from ``min_length``.
    They were the same value, and at a wall thickness of 24 pixels against
    a kernel of 12 nothing was severed at all: the perimeter came back as
    one connected ring, reported as a single "wall" a thousand pixels
    thick. Sized off the drawing's gauge, the same plan decomposes into the
    walls it is drawn with.
    """
    sever = max(int(sever or min_length), 3)
    kernel_shape = (sever, 1) if horizontal else (1, sever)
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
        #
        # Grouped by how far apart they actually are rather than by rounding
        # each into a fixed bucket. Bucketing made the answer depend on
        # where the building happened to sit on the page: two runs six
        # pixels apart merged at one position and not at another, purely
        # because they straddled a boundary. A wall meeting a thicker wall
        # always offsets by half their difference, so this is the ordinary
        # case rather than an edge one.
        lanes: list[list[Wall]] = []
        for wall in sorted(candidates, key=lambda w: w.start[axis]):
            if lanes and wall.start[axis] - lanes[-1][-1].start[axis] <= offset:
                lanes[-1].append(wall)
            else:
                lanes.append([wall])

        for lane in lanes:
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


# How many times the drawing's own median wall thickness a run may measure
# before it is not a wall at all. Deliberately relative rather than a
# figure in feet: wall extraction runs before the scale is known, and the
# thicknesses on the drawing are the only reference available at that
# point. It also means the rule needs no adjusting between a plan drawn at
# 1:50 and one at 1:200.
#
# Real walls span a narrow range -- a 4 inch partition to an 18 inch
# external wall, occasionally a 2 foot column -- so four times the median
# clears every wall a building actually has. What it catches is boundary
# hatching, dimension bands and title-block rules, which the segmenter
# reports as enormously thick walls: on the reference sheet the median run
# is 10 inches and the fattest is nearly ten feet.
MAX_THICKNESS_RATIO = 4.0

def _drop_impossibly_thick(walls: list[Wall]) -> list[Wall]:
    """Remove runs far thicker than the drawing's own walls.

    A ten foot thick wall is not a wall. Left in, it becomes a solid slab
    across the plan -- and worse, it drags the wall-thickness scale
    estimate with it, so the whole building comes out the wrong size.
    """
    if len(walls) < MIN_WALLS_TO_JUDGE:
        return walls

    limit = _typical_thickness(walls) * MAX_THICKNESS_RATIO
    kept = [wall for wall in walls if wall.thickness <= limit]

    dropped = len(walls) - len(kept)
    if not dropped:
        return walls
    if dropped > len(walls) * MAX_DROPPED_FRACTION:
        logger.info(
            "not dropping %d of %d run(s): too many to be hatching", dropped, len(walls)
        )
        return walls

    logger.info("dropped %d run(s) thicker than %.0f px; not walls", dropped, limit)
    return kept


def extract_walls(
    mask: np.ndarray,
    wall_class: int = WALL,
    min_wall_length: int | None = None,
    merge: bool = True,
    gauge: float | None = None,
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

    gauge = wall_gauge(mask, wall_class) if gauge is None else gauge
    if min_wall_length is None:
        min_wall_length = max(
            int(MIN_WALL_LENGTH_RATIO * gauge), MIN_WALL_LENGTH_FLOOR
        )

    sever = int(SEVER_RATIO * gauge)
    walls = _segments_along(
        binary, horizontal=True, min_length=min_wall_length, sever=sever
    )
    walls += _segments_along(
        binary, horizontal=False, min_length=min_wall_length, sever=sever
    )

    # A wall is longer than it is thick. Each orientation pass sees the
    # other's walls end-on -- a band 40 across and 3 deep is reported by
    # the vertical pass as a wall 40 thick and 3 long -- and those stubs
    # are the same run counted twice, at right angles.
    walls = [
        wall
        for wall in walls
        if wall.thickness <= 0 or wall.length() >= wall.thickness * MIN_WALL_ASPECT
    ] or walls
    walls = _drop_impossibly_thick(walls)

    if merge and walls:
        before = len(walls)
        walls = _merge_collinear(
            walls,
            gap=MERGE_GAP_RATIO * gauge,
            offset=MERGE_OFFSET_RATIO * gauge,
        )
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
    min_area: int | None = None,
    max_distance: float | None = None,
    gauge: float | None = None,
) -> list[Opening]:
    """Find doors and windows and bind each to the wall it interrupts.

    An opening is only meaningful relative to its wall, so a component that
    sits too far from any wall is discarded rather than attached to a distant
    one -- a misplaced opening cuts a hole through solid geometry.

    Sizes come from the drawing's own wall thickness, so a plan rendered
    at twice the resolution is read the same way rather than losing every
    opening to a threshold set for a smaller sheet.
    """
    if not walls:
        return []

    gauge = wall_gauge(mask) if gauge is None else gauge
    if min_area is None:
        min_area = max(int(MIN_OPENING_AREA_RATIO * gauge**2), 4)
    if max_distance is None:
        max_distance = MAX_OPENING_DISTANCE_RATIO * gauge

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
                and (mask[row, column] in ROOM_CLASSES or mask[row, column] == WALL)
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
    simplify_pixels: float | None = None,
    gauge: float | None = None,
) -> list[tuple[float, float]]:
    """Outline of the built area, for generating floor slabs and a roof.

    Walls and rooms together make up the storey's extent, so the outer
    contour of both classes traces its footprint. Returns an empty list when
    the mask holds no building.
    """
    built = ((mask == WALL) | np.isin(mask, list(ROOM_CLASSES))).astype(np.uint8)
    if not built.any():
        return []

    # Sized against the drawing's own walls: a doorway is about the same
    # multiple of a wall thickness whatever the sheet's resolution, and a
    # spur worth erasing is thinner than a wall.
    gauge = wall_gauge(mask) if gauge is None else gauge
    if simplify_pixels is None:
        simplify_pixels = SIMPLIFY_RATIO * gauge
    close_span = max(int(CLOSE_SPAN_RATIO * gauge), 3)
    open_span = max(int(OPEN_SPAN_RATIO * gauge), 3)

    # Bridge doorways and joints so the storey reads as one region.
    closed = cv2.morphologyEx(built, cv2.MORPH_CLOSE, np.ones((close_span, close_span), np.uint8))

    # Erode away thin spurs, then restore the body. Segmentation bleeds into
    # landscaping and paving around the building, and those tendrils otherwise
    # end up in the outline -- producing slabs with jagged fingers reaching
    # out over open ground.
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((open_span, open_span), np.uint8))
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
    room_class: int | None = None,
    min_area: int | None = None,
    simplify_pixels: float | None = None,
    gauge: float | None = None,
) -> list[Room]:
    """Recover room polygons from a class mask, in the mask's pixel coordinates.

    Every room class is traced separately and each room carries the class it
    came from as its ``category``. Tracing them together would merge a
    kitchen into the living room it opens onto, since nothing but the class
    separates them -- there is no wall between an open kitchen and its
    dining area, which is the whole point of an open kitchen.

    Pass ``room_class`` to restrict the search to a single class.

    Regions that cannot form a valid polygon are logged and skipped rather
    than raised, so one malformed room does not abort a floor.

    Sizes come from the drawing's own wall thickness, so the same plan is
    read the same way whatever size it is rendered at.
    """
    gauge = wall_gauge(mask) if gauge is None else gauge
    if min_area is None:
        min_area = max(int(MIN_ROOM_AREA_RATIO * gauge**2), 16)
    if simplify_pixels is None:
        simplify_pixels = SIMPLIFY_RATIO * gauge

    wanted = ROOM_CLASSES if room_class is None else {room_class}

    rooms: list[Room] = []
    for class_index in sorted(wanted):
        binary = (mask == class_index).astype(np.uint8)
        if not binary.any():
            continue

        # The generic ROOM class carries no information, so it is left
        # uncategorised rather than labelled "room" -- downstream needs to
        # tell "no opinion" apart from a positive identification.
        category = "" if class_index == ROOM else CLASS_NAMES[class_index]

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            simplified = cv2.approxPolyDP(contour, simplify_pixels, closed=True)
            polygon = [(float(p[0][0]), float(p[0][1])) for p in simplified]

            try:
                rooms.append(Room(polygon=polygon, category=category))
            except ValueError as error:
                logger.warning("skipping region of area %.0f: %s", area, error)

    typed = sum(1 for room in rooms if room.category)
    logger.info("extracted %d room polygon(s), %d typed", len(rooms), typed)
    return rooms
