"""Turn wall geometry into a 3D mesh.

Plan coordinates are pixels on a page: X across, Y down. glTF is Y-up, so
the page's Y becomes the model's Z (depth) and Y becomes height. Everything
converts to metres on the way out, since glTF viewers assume metres.

Floors stack in the order given. They share a horizontal frame because the
sheets were cropped to a common box upstream, so no per-floor alignment is
applied here.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import unary_union
from trimesh.creation import extrude_polygon

from planto3d.design import Landscaping
from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.features import (
    GROUND_COVERS,
    feature_for,
    finish_for_room,
    group_by_feature,
    is_open_to_sky,
)
from planto3d.site import (
    BOUNDARY_HEIGHT_FT,
    BOUNDARY_THICKNESS_FT,
    COVER_THICKNESS_FT,
    ENTRANCE_RISER_FT,
    ENTRANCE_TREAD_FT,
    EXTERNAL_DOOR_MARGIN_FT,
    PLINTH_HEIGHT_FT,
    PLINTH_OVERHANG_FT,
    POOL_DEPTH_FT,
    POST_SPACING_FT,
    RAIL_DEPTH_FT,
    RAILING_HEIGHT_FT,
    RAILING_THICKNESS_FT,
    SITE_MARGIN_FT,
    SITE_THICKNESS_FT,
    boundary_walls,
    site_outline,
)

logger = logging.getLogger(__name__)

FEET_TO_METRES = 0.3048
# Storey height used when the drawing does not state one.
DEFAULT_WALL_HEIGHT_FT = 9.0
# Walls shorter than this in pixels are extraction noise, not geometry.
MIN_WALL_PIXELS = 1e-6
# Structural slab thickness, and the parapet standing above the roof.
SLAB_THICKNESS_FT = 0.5
PARAPET_HEIGHT_FT = 3.0
PARAPET_THICKNESS_FT = 0.6
# A coping caps the parapet, oversailing it to throw water clear.
COPING_THICKNESS_FT = 0.3
COPING_OVERHANG_FT = 0.35
# The enclosure over a stairwell that reaches the roof.
HEADROOM_HEIGHT_FT = 7.5

# --- Roof forms other than flat ----------------------------------------------
#
# A flat slab with a parapet is the common case on these houses and was the
# only thing built. Plenty of roofs are not that: a dome over a temple or a
# stairhall, a slanting glazed ceiling over a conservatory, a pitched roof
# over a wing. Each is raised over the room the drawing names, at roof level.
#
# The rises are proportions of the room's shorter span rather than fixed
# heights, so a dome over a small shrine and one over a hall both look
# right. A half rise is a true hemisphere, which is what "dome" means when
# a drawing does not say otherwise.
DOME_RISE_RATIO = 0.5
# Enough segments that the silhouette reads as a curve rather than a cone.
DOME_SEGMENTS = 4
# A low drum lifts the dome clear of the roof slab, the way a real one sits
# on a base rather than springing straight off the deck.
DOME_DRUM_FT = 0.8

# A pitched roof's ridge sits above the eaves by this share of the shorter
# span, which is about 35 degrees -- the ordinary domestic pitch.
PITCH_RISE_RATIO = 0.35
# Slanting glazing is laid shallower than a tiled roof, both because glass
# is usually a lean-to over a conservatory and because a steep glass plane
# reads as a wall.
GLAZED_RISE_RATIO = 0.22
GLAZED_THICKNESS_FT = 0.25

# Things that stand on the roof. Heights are absolute rather than
# proportional: a chimney is about a storey above the deck whatever the
# room beneath it measures, and scaling one to its flue would give a
# cottage a factory stack.
CHIMNEY_HEIGHT_FT = 5.0
TOWER_HEIGHT_FT = 12.0
# A tower is capped rather than left as a flat-topped box, which reads as
# an unfinished lift overrun rather than a turret.
TOWER_CAP_RISE_RATIO = 0.55

# An overhead tank stands clear of the deck on short legs, which is how
# every one of them is built -- resting directly on the roof it would read
# as a packing case.
TANK_HEIGHT_FT = 4.0
TANK_STAND_FT = 1.5

# A canopy is a thin projecting cover on brackets, set just below the
# ceiling of the storey it is drawn on rather than at roof level.
CANOPY_THICKNESS_FT = 0.6
CANOPY_DROP_FT = 1.0

# A ramp falls by this share of its length. About 1:12, the usual
# accessible gradient, which also looks right for a car ramp.
RAMP_FALL_RATIO = 0.085
RAMP_THICKNESS_FT = 0.5
HEADROOM_WALL_FT = 0.6
# Smallest piece of a divided slab worth building, against the largest.
# Cutting an open region out of a slab leaves shreds along the cut where
# the two outlines very nearly agree; those are artefacts of the boolean
# and not roof. Anything above this is a real part of the building.
MIN_SLAB_PIECE_SHARE = 0.02

# A footprint needs this many vertices to enclose an area.
MIN_FOOTPRINT_VERTICES = 3
# Door and window heads sit at a common height; windows also get a sill.
OPENING_HEAD_FT = 7.0
SILL_HEIGHT_FT = 3.0
# A comfortable riser. Steps are sized from this rather than fixed in number,
# so a flight climbing a taller storey simply gets more of them.
RISER_HEIGHT_FT = 0.58
# A floor finish is a skin on the slab, not a slab of its own.
FINISH_THICKNESS_FT = 0.08
# Slivers below these sizes are dropped rather than modelled -- a millimetre
# of wall between two openings is noise, not architecture.
MIN_OPENING_WIDTH_M = 0.15
MIN_PIER_M = 0.05
# Glazing is thinner than the wall it sits in, so it reads as a pane rather
# than a plug and does not z-fight with the reveal.
GLASS_THICKNESS_RATIO = 0.25
# Window frames: the section of a member, how far it stands proud of the
# reveal, and the widest pane before a mullion divides it.
FRAME_SECTION_FT = 0.28
FRAME_DEPTH_RATIO = 0.45
MAX_PANE_WIDTH_M = 1.4


def _place(
    extents: tuple[float, float, float],
    wall: Wall,
    along_m: float,
    centre_height_m: float,
    scale: float,
) -> trimesh.Trimesh:
    """Build a box and place it along a wall at a given distance and height."""
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    direction = end - start
    length_m = float(np.linalg.norm(direction))
    unit = direction / length_m

    box = trimesh.creation.box(extents=list(extents))

    # Rotate about the vertical axis to align with the wall's direction. The
    # page's downward Y maps to +Z, so the angle is measured in the XZ plane.
    angle = -np.arctan2(direction[1], direction[0])
    box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))

    centre = start + unit * along_m
    box.apply_transform(
        trimesh.transformations.translation_matrix([centre[0], centre_height_m, centre[1]])
    )
    return box


def _wall_parts(
    wall: Wall,
    openings: list[Opening],
    height_m: float,
    scale: float,
    base_m: float,
) -> list[trimesh.Trimesh]:
    """A wall as solid pieces, with gaps left where openings interrupt it.

    Built by splitting rather than boolean subtraction: the pier either side
    of an opening, plus a lintel over it and a sill under a window. CSG on
    this many boxes is slow and fragile, and splitting yields clean geometry
    with no dependency on a boolean backend.
    """
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    thickness_m = max(wall.thickness / scale * FEET_TO_METRES, 1e-4)

    length_m = float(np.linalg.norm(end - start))
    if length_m <= MIN_WALL_PIXELS:
        return []

    if not openings:
        return [
            _place(
                (length_m, height_m, thickness_m),
                wall,
                length_m / 2,
                base_m + height_m / 2,
                scale,
            )
        ]

    # Openings in order along the wall, clipped to it and non-overlapping.
    spans: list[tuple[float, float, str]] = []
    for opening in sorted(openings, key=lambda o: o.position):
        half = (opening.width / scale * FEET_TO_METRES) / 2
        centre = opening.position / scale * FEET_TO_METRES
        low, high = max(0.0, centre - half), min(length_m, centre + half)
        if high - low < MIN_OPENING_WIDTH_M:
            continue
        if spans and low < spans[-1][1]:
            low = spans[-1][1]
            if high - low < MIN_OPENING_WIDTH_M:
                continue
        spans.append((low, high, opening.type))

    if not spans:
        return [
            _place(
                (length_m, height_m, thickness_m),
                wall,
                length_m / 2,
                base_m + height_m / 2,
                scale,
            )
        ]

    parts: list[trimesh.Trimesh] = []
    cursor = 0.0

    for low, high, opening_type in spans:
        if low - cursor > MIN_PIER_M:
            pier = low - cursor
            parts.append(
                _place(
                    (pier, height_m, thickness_m),
                    wall,
                    cursor + pier / 2,
                    base_m + height_m / 2,
                    scale,
                )
            )

        span = high - low
        head_m = min(OPENING_HEAD_FT * FEET_TO_METRES, height_m)
        sill_m = SILL_HEIGHT_FT * FEET_TO_METRES if opening_type == "window" else 0.0

        if sill_m > 0:
            parts.append(
                _place(
                    (span, sill_m, thickness_m), wall, low + span / 2, base_m + sill_m / 2, scale
                )
            )

        lintel = height_m - head_m
        if lintel > MIN_PIER_M:
            parts.append(
                _place(
                    (span, lintel, thickness_m),
                    wall,
                    low + span / 2,
                    base_m + head_m + lintel / 2,
                    scale,
                )
            )

        cursor = high

    if length_m - cursor > MIN_PIER_M:
        remainder = length_m - cursor
        parts.append(
            _place(
                (remainder, height_m, thickness_m),
                wall,
                cursor + remainder / 2,
                base_m + height_m / 2,
                scale,
            )
        )

    return parts


def opening_frames(
    wall: Wall,
    openings: list[Opening],
    height_m: float,
    scale: float,
    base_m: float,
) -> list[trimesh.Trimesh]:
    """Frames and mullions around each window opening.

    Bare glass set into a hole reads as a void rather than a window. A frame
    around the edge and mullions dividing wide openings give the facade the
    articulation that makes glazing legible -- and wide windows really are
    divided, since glass is not manufactured in arbitrary widths.
    """
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    length_m = float(np.linalg.norm(end - start))
    if length_m <= MIN_WALL_PIXELS:
        return []

    depth_m = max(wall.thickness / scale * FEET_TO_METRES, 1e-4) * FRAME_DEPTH_RATIO
    section_m = FRAME_SECTION_FT * FEET_TO_METRES
    head_m = min(OPENING_HEAD_FT * FEET_TO_METRES, height_m)
    sill_m = SILL_HEIGHT_FT * FEET_TO_METRES

    members: list[trimesh.Trimesh] = []
    for opening in openings:
        if opening.type != "window":
            continue

        half = (opening.width / scale * FEET_TO_METRES) / 2
        centre = opening.position / scale * FEET_TO_METRES
        low, high = max(0.0, centre - half), min(length_m, centre + half)
        span = high - low
        if span < MIN_OPENING_WIDTH_M or head_m <= sill_m:
            continue

        pane_height = head_m - sill_m

        # Head and sill members, running the opening's width.
        for level in (sill_m + section_m / 2, head_m - section_m / 2):
            members.append(
                _place(
                    (span, section_m, depth_m), wall, low + span / 2, base_m + level, scale
                )
            )

        # Jambs at each side, and mullions dividing anything wider than a
        # single pane can sensibly span.
        divisions = max(int(span / MAX_PANE_WIDTH_M), 1)
        for step in range(divisions + 1):
            members.append(
                _place(
                    (section_m, pane_height, depth_m),
                    wall,
                    low + span * step / divisions,
                    base_m + sill_m + pane_height / 2,
                    scale,
                )
            )

    return members


def opening_panes(
    wall: Wall,
    openings: list[Opening],
    height_m: float,
    scale: float,
    base_m: float,
) -> list[trimesh.Trimesh]:
    """Thin panes filling each window opening, to be given a glass material.

    Doors are left empty: an open doorway reads correctly, whereas a glazed
    one does not.
    """
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    length_m = float(np.linalg.norm(end - start))
    if length_m <= MIN_WALL_PIXELS:
        return []

    thickness_m = max(wall.thickness / scale * FEET_TO_METRES, 1e-4)
    head_m = min(OPENING_HEAD_FT * FEET_TO_METRES, height_m)
    sill_m = SILL_HEIGHT_FT * FEET_TO_METRES

    panes = []
    for opening in openings:
        if opening.type != "window":
            continue

        half = (opening.width / scale * FEET_TO_METRES) / 2
        centre = opening.position / scale * FEET_TO_METRES
        low, high = max(0.0, centre - half), min(length_m, centre + half)
        span = high - low
        if span < MIN_OPENING_WIDTH_M or head_m <= sill_m:
            continue

        pane_height = head_m - sill_m
        panes.append(
            _place(
                (span, pane_height, thickness_m * GLASS_THICKNESS_RATIO),
                wall,
                low + span / 2,
                base_m + sill_m + pane_height / 2,
                scale,
            )
        )

    return panes


# What an open-to-sky area has to be before it is allowed to cut a hole in
# a roof. Somebody has to be able to stand in it: a balcony, a terrace, a
# courtyard or a lightwell is at least a couple of feet across and covers
# more than a doormat.
#
# Without this a segmentation artefact punches a slot through the roof and
# the parapet dutifully lines both sides of it, which reads as a fin
# standing across the roof plane. One measured case was 2.8 square feet and
# ten inches wide.
MIN_OPEN_AREA_SQFT = 15.0
MIN_OPEN_WIDTH_FT = 2.5


def open_to_sky(floor: FloorPlan, scale: float) -> list[list[tuple[float, float]]]:
    """Regions on this storey that must have nothing built over them.

    A balcony, a terrace, a courtyard, a light well, a terrace garden, a
    roof pool -- the drawing marks them differently from one another and
    they all mean the same thing to the geometry: no ceiling here.

    Collected from three sources because no one of them is reliable. The
    segmenter's ``outdoor`` class finds them on plans that draw them as
    rooms; a printed label finds them on plans that name them; colour
    finds the planted ones. The reference sheet's colour blobs cover 48%
    of the top storey while the room labelled TERRACE GARDEN covers 73%,
    so any single source leaves a quarter of the garden under a slab.

    Sized afterwards, so a sliver of misread paving cannot punch a hole
    through a floor.
    """
    regions = list(floor.planting)
    for category in GROUND_COVERS:
        regions += floor.labelled_regions.get(category, [])
    regions += [room.polygon for room in floor.rooms if is_open_to_sky(room)]
    regions = real_open_regions(regions, scale)
    return merge_regions(regions) if regions else []


def _beyond(
    regions: list[list[tuple[float, float]]], footprint: list[tuple[float, float]]
) -> list[list[tuple[float, float]]]:
    """The parts of ``regions`` lying outside ``footprint``.

    Used to keep a hole in a slab from reaching under the storey that slab
    belongs to: what is open below is only open where nothing stands above
    it. Returns the regions untouched if the difference cannot be taken,
    which errs towards the old behaviour rather than towards a hole in
    somebody's floor.
    """
    if not regions or len(footprint) < MIN_FOOTPRINT_VERTICES:
        return regions

    try:
        stands_on = Polygon(footprint).buffer(0)
    except Exception as error:
        logger.warning("could not read the storey's own outline: %s", error)
        return regions

    beyond = []
    for region in regions:
        if len(region) < MIN_FOOTPRINT_VERTICES:
            continue
        try:
            remainder = Polygon(region).buffer(0).difference(stands_on)
        except Exception as error:
            logger.warning("could not trim an open region: %s", error)
            continue
        if remainder.is_empty:
            continue
        pieces = (
            remainder.geoms if remainder.geom_type == "MultiPolygon" else [remainder]
        )
        for piece in pieces:
            if piece.area > 0:
                beyond.append([(float(x), float(y)) for x, y in piece.exterior.coords[:-1]])
    return beyond


def real_open_regions(
    regions: list[list[tuple[float, float]]], scale: float
) -> list[list[tuple[float, float]]]:
    """Keep only the open areas big enough to be real.

    Width is measured by shrinking the region until it disappears, which
    is the distance to its own medial axis -- so a long thin slot fails on
    width even when its area is respectable, and an L-shaped terrace
    passes on both.
    """
    if not regions or scale <= 0:
        return regions

    minimum_area = MIN_OPEN_AREA_SQFT * scale**2
    inset = MIN_OPEN_WIDTH_FT * scale / 2

    kept = []
    for region in regions:
        if len(region) < MIN_FOOTPRINT_VERTICES:
            continue
        try:
            polygon = Polygon(region).buffer(0)
            if polygon.is_empty or polygon.area < minimum_area:
                continue
            if polygon.buffer(-inset).is_empty:
                continue
        except Exception as error:
            logger.warning("could not size an open region: %s", error)
            continue
        kept.append(region)

    dropped = len(regions) - len(kept)
    if dropped:
        logger.info("ignored %d open region(s) too small to stand in", dropped)
    return kept


def merge_regions(
    polygons: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Combine overlapping regions into one outline each.

    The same area is often found twice -- once from the drawing's colour and
    once from its dimension label -- and laying both down draws a garden two
    or three times over, which makes it dominate the model and inflates its
    measured area well past what the sheet states.
    """
    if len(polygons) < 2:
        return polygons

    shapes = []
    for polygon in polygons:
        if len(polygon) < MIN_FOOTPRINT_VERTICES:
            continue
        shape = Polygon(polygon)
        if not shape.is_valid:
            shape = shape.buffer(0)
        if not shape.is_empty and shape.area > 0:
            shapes.append(shape)

    if not shapes:
        return []

    merged = unary_union(shapes)
    pieces = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]

    return [
        [(float(x), float(y)) for x, y in piece.exterior.coords[:-1]]
        for piece in pieces
        if piece.area > 0
    ]


# A storey occupies its wall height *plus* the slab it stands on. Stacking
# them at the wall height alone buried each slab in the top of the walls
# below, leaving the slab's upper face and the wall's upper face on exactly
# the same plane -- which renders as a field of speckle across every roof,
# two surfaces at identical depth with nothing to separate them.
def _storey_base_ft(index: int, wall_height_ft: float, plinth_ft: float = 0.0) -> float:
    """Height of a storey's slab underside, in feet above the ground."""
    return plinth_ft + index * (wall_height_ft + SLAB_THICKNESS_FT)


def slab_mesh(
    footprint: list[tuple[float, float]],
    thickness_ft: float,
    base_ft: float,
    scale: float,
    holes: list[list[tuple[float, float]]] | None = None,
) -> trimesh.Trimesh | None:
    """A horizontal slab covering a storey's footprint.

    ``holes`` are regions to cut out -- an open terrace garden, for instance,
    which must not be roofed over.

    Returns None when the outline cannot enclose an area, so a bad contour
    costs a slab rather than the whole model.
    """
    if len(footprint) < MIN_FOOTPRINT_VERTICES:
        return None

    def to_metres(points):
        return [(x / scale * FEET_TO_METRES, y / scale * FEET_TO_METRES) for x, y in points]

    try:
        polygon = Polygon(to_metres(footprint))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # repair self-intersections

        for hole in holes or []:
            if len(hole) < MIN_FOOTPRINT_VERTICES:
                continue
            cut = Polygon(to_metres(hole))
            if not cut.is_valid:
                cut = cut.buffer(0)
            if cut.is_valid and not cut.is_empty:
                polygon = polygon.difference(cut)

        if polygon.is_empty or polygon.area <= 0:
            return None

        # A difference can split the slab into several pieces -- a terrace
        # running the depth of a building leaves a roof either side of it,
        # which is a perfectly ordinary roof. Keeping only the largest of
        # them threw the rest away: an open region covering 5% of a storey
        # was enough to remove half that storey's roof, because the region
        # happened to span it rather than sit inside it. The building then
        # renders as half sealed and half open to the weather.
        #
        # So every piece is built. Only true slivers are dropped, which are
        # the artefacts of the boolean itself rather than parts of a roof.
        parts = (
            list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
        )
        largest = max(part.area for part in parts)
        parts = [
            part
            for part in parts
            if not part.is_empty and part.area > MIN_SLAB_PIECE_SHARE * largest
        ]
        if not parts:
            return None

        slab = trimesh.util.concatenate(
            [
                extrude_polygon(part, height=thickness_ft * FEET_TO_METRES)
                for part in parts
            ]
        )
    except Exception as error:
        logger.warning("could not build slab from footprint: %s", error)
        return None

    # extrude_polygon builds the outline in XY and extrudes along +Z. Rotating
    # +90 degrees about X sends the page's downward Y to +Z, matching how
    # walls are placed -- the opposite rotation mirrors the slab and doubles
    # the model's depth. That rotation leaves the slab hanging below the
    # origin, so it is lifted by its own thickness as well as the storey base.
    thickness_m = thickness_ft * FEET_TO_METRES
    slab.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    slab.apply_transform(
        trimesh.transformations.translation_matrix(
            [0, base_ft * FEET_TO_METRES + thickness_m, 0]
        )
    )
    return slab


# How far past an edge to look for floor before calling that edge a drop.
# A little over a wall's thickness, so a railing is not cancelled by the
# slab it stands on poking out beneath it.
DROP_PROBE_FT = 1.0


def _guarded_edges(
    polygon: list[tuple[float, float]], standing_on: "Polygon | None", scale: float
) -> list[bool]:
    """Which edges of ``polygon`` have a drop beyond them.

    A railing exists to stop someone falling. An edge with more floor on
    the far side of it is not a fall, it is a doorway or a change of
    surface, and a rail across it is a fence through the middle of a
    terrace.

    That was happening wherever the segmenter found a patch of open ground
    inside a larger open area -- a deck within a terrace garden, paving
    inside a courtyard. Each patch got a balustrade around its whole
    perimeter, so the reference building's roof garden came out with 300
    stray posts standing in the lawn tracing the shapes of its furniture.

    Returns one flag per edge, all True when the storey's extent is not
    known, which is the behaviour this replaces.
    """
    edges = [True] * len(polygon)
    if standing_on is None or standing_on.is_empty:
        return edges

    probe = DROP_PROBE_FT * scale
    for index in range(len(polygon)):
        start = np.asarray(polygon[index], dtype=float)
        end = np.asarray(polygon[(index + 1) % len(polygon)], dtype=float)
        along = end - start
        length = float(np.hypot(*along))
        if length <= 0:
            continue

        # Straight out from the edge's midpoint, on the side away from the
        # region's own interior.
        outward = np.array([along[1], -along[0]]) / length
        middle = (start + end) / 2
        centre = np.asarray(polygon, dtype=float).mean(axis=0)
        if np.dot(outward, middle - centre) < 0:
            outward = -outward

        beyond = middle + outward * probe
        edges[index] = not standing_on.contains(ShapelyPoint(float(beyond[0]), float(beyond[1])))

    return edges


def _railing_parts(
    polygon: list[tuple[float, float]],
    base_m: float,
    scale: float,
    standing_on: "Polygon | None" = None,
) -> list[trimesh.Trimesh]:
    """A balustrade around an open-edged room: posts under a top rail.

    Modelled as separate members rather than one solid panel. A panel the
    full height of a rail reads as a parapet wall and blocks the view through
    a balcony, which is most of what a balcony is for.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    thickness_px = RAILING_THICKNESS_FT * scale
    thickness_m = max(RAILING_THICKNESS_FT * FEET_TO_METRES, 1e-4)
    height_m = RAILING_HEIGHT_FT * FEET_TO_METRES
    rail_m = RAIL_DEPTH_FT * FEET_TO_METRES
    spacing_m = POST_SPACING_FT * FEET_TO_METRES

    guarded = _guarded_edges(polygon, standing_on, scale)

    parts: list[trimesh.Trimesh] = []
    for index in range(len(polygon)):
        if not guarded[index]:
            continue
        run = Wall(
            start=polygon[index],
            end=polygon[(index + 1) % len(polygon)],
            thickness=thickness_px,
        )

        start = np.array(run.start, dtype=float) / scale * FEET_TO_METRES
        end = np.array(run.end, dtype=float) / scale * FEET_TO_METRES
        length_m = float(np.linalg.norm(end - start))
        if length_m <= MIN_OPENING_WIDTH_M:
            continue

        # The top rail, running the whole length.
        parts.append(
            _place(
                (length_m, rail_m, thickness_m),
                run,
                length_m / 2,
                base_m + height_m - rail_m / 2,
                scale,
            )
        )

        # Posts at regular intervals, with one at each end.
        count = max(int(length_m / spacing_m), 1)
        for step in range(count + 1):
            along = length_m * step / count
            parts.append(
                _place(
                    (thickness_m, height_m, thickness_m),
                    run,
                    along,
                    base_m + height_m / 2,
                    scale,
                )
            )

    return parts


def _expand_outline(
    polygon: list[tuple[float, float]], margin_px: float
) -> list[tuple[float, float]]:
    """Grow an outline outwards, for a plinth that oversails its walls."""
    try:
        shape = Polygon(polygon)
        if not shape.is_valid:
            shape = shape.buffer(0)
        grown = shape.buffer(margin_px, join_style=2)  # mitred, to stay square
        if grown.geom_type == "MultiPolygon":
            grown = max(grown.geoms, key=lambda part: part.area)
        return [(float(x), float(y)) for x, y in grown.exterior.coords[:-1]]
    except Exception as error:
        logger.warning("could not expand outline: %s", error)
        return polygon


def _near_outline(
    wall: Wall, outline: list[tuple[float, float]], margin_px: float
) -> bool:
    """Whether a wall runs along the building's outer edge.

    Used to tell an external door from an internal one: only the former
    needs steps down to the ground.
    """
    if len(outline) < MIN_FOOTPRINT_VERTICES:
        return False

    try:
        edge = Polygon(outline).exterior
    except Exception:
        return False

    midpoint = (
        (wall.start[0] + wall.end[0]) / 2,
        (wall.start[1] + wall.end[1]) / 2,
    )
    return edge.distance(ShapelyPoint(midpoint)) <= margin_px


def _entrance_steps(
    wall: Wall,
    opening: Opening,
    plinth_ft: float,
    scale: float,
) -> list[trimesh.Trimesh]:
    """Steps from the ground up to a door standing on the plinth.

    Each tread projects further out than the one above it, so the flight
    steps down and away from the threshold rather than hanging off it.
    """
    if plinth_ft <= 0:
        return []

    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    direction = end - start
    length_m = float(np.linalg.norm(direction))
    if length_m <= MIN_WALL_PIXELS:
        return []

    # The outward normal, taken in the horizontal plane.
    unit = direction / length_m
    normal = np.array([-unit[1], unit[0]])

    width_m = max(opening.width / scale * FEET_TO_METRES, 0.6)
    along_m = opening.position / scale * FEET_TO_METRES
    tread_m = ENTRANCE_TREAD_FT * FEET_TO_METRES

    count = max(int(round(plinth_ft / ENTRANCE_RISER_FT)), 1)
    riser_m = plinth_ft * FEET_TO_METRES / count

    threshold = start + unit * along_m
    angle = -np.arctan2(direction[1], direction[0])

    steps = []
    for index in range(count):
        height_m = riser_m * (index + 1)
        # The lowest step reaches furthest from the wall.
        offset = tread_m * (count - index - 0.5)
        centre = threshold + normal * offset

        box = trimesh.creation.box(extents=[width_m, height_m, tread_m])
        box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
        box.apply_transform(
            trimesh.transformations.translation_matrix(
                [centre[0], height_m / 2, centre[1]]
            )
        )
        steps.append(box)

    return steps


def _stair_parts(
    polygon: list[tuple[float, float]],
    base_ft: float,
    rise_ft: float,
    scale: float,
) -> list[trimesh.Trimesh]:
    """A flight of steps climbing one storey within a stairwell's outline.

    Treads run across the flight's narrow dimension and climb along its long
    one, which is how a straight flight is drawn: the run needs the length,
    the width only needs to fit a person. Without this a storey has no
    visible way of reaching the one above, which reads as wrong even when
    everything else is right.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES or rise_ft <= 0:
        return []

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    width_px, depth_px = right - left, bottom - top
    if width_px <= 0 or depth_px <= 0:
        return []

    climbs_along_x = width_px >= depth_px
    run_px = width_px if climbs_along_x else depth_px
    across_px = depth_px if climbs_along_x else width_px

    steps = max(int(rise_ft / RISER_HEIGHT_FT), 1)
    riser_m = (rise_ft / steps) * FEET_TO_METRES
    tread_m = (run_px / steps) / scale * FEET_TO_METRES
    across_m = across_px / scale * FEET_TO_METRES
    base_m = base_ft * FEET_TO_METRES

    parts = []
    for index in range(steps):
        # Each tread is a solid block from the floor to its own height, so
        # the flight reads as a stair rather than a floating ribbon.
        height_m = riser_m * (index + 1)
        along_px = run_px * (index + 0.5) / steps

        if climbs_along_x:
            centre = (left + along_px, (top + bottom) / 2)
            extents = (tread_m, height_m, across_m)
        else:
            centre = ((left + right) / 2, top + along_px)
            extents = (across_m, height_m, tread_m)

        box = trimesh.creation.box(extents=list(extents))
        box.apply_transform(
            trimesh.transformations.translation_matrix(
                [
                    centre[0] / scale * FEET_TO_METRES,
                    base_m + height_m / 2,
                    centre[1] / scale * FEET_TO_METRES,
                ]
            )
        )
        parts.append(box)

    return parts


# Share of a region that must fall within the building's outline before it
# is treated as part of the building rather than the plot around it. Well
# under half, because a terrace that oversails its own storey is still a
# terrace, while a lawn merely brushing a corner of the footprint is not.
INSIDE_FRACTION = 0.4


# How far from an enclosed room a wall may stand and still be counted as
# enclosing it, in multiples of its own thickness. A wall bounding a room
# sits against it; one bounding open sky does not.
ENCLOSING_REACH = 2.5


def _open_air_walls(floor: FloorPlan, scale: float) -> set[int]:
    """Walls on this storey that enclose nothing, so should be low.

    A terrace garden or an open deck is drawn with its edge marked exactly
    as a wall is, and built at storey height it turns the terrace into the
    bottom of a well: three metres of blank masonry round a lawn, with the
    roof parapet on top of that. Seen from outside, the terrace reads a
    full floor lower than it is.

    Anything roofed is enclosed by definition, so this only asks which
    walls have a room behind them. Where a storey has no open feature at
    all -- the usual case -- the answer is "none of them" and nothing
    changes.
    """
    open_regions = [room.polygon for room in floor.rooms if is_open_to_sky(room)]
    open_regions += floor.planting
    for category in GROUND_COVERS:
        open_regions += floor.labelled_regions.get(category, [])
    open_regions = real_open_regions(open_regions, scale)
    if not open_regions:
        return set()

    enclosed = [
        room.polygon
        for room in floor.rooms
        if not is_open_to_sky(room) and feature_for(room) != "void"
    ]
    if not enclosed:
        return set()

    try:
        rooms = unary_union(
            [Polygon(p).buffer(0) for p in enclosed if len(p) >= MIN_FOOTPRINT_VERTICES]
        )
        sky = unary_union(
            [
                Polygon(p).buffer(0)
                for p in open_regions
                if len(p) >= MIN_FOOTPRINT_VERTICES
            ]
        )
    except Exception as error:
        logger.warning("could not work out which walls are parapets: %s", error)
        return set()

    if rooms.is_empty or sky.is_empty:
        return set()

    low = set()
    for wall_id, wall in enumerate(floor.walls):
        reach = max(wall.thickness * ENCLOSING_REACH, 1.0)
        line = LineString([wall.start, wall.end]).buffer(reach)
        # A wall with a room against it is holding something up. One with
        # only open ground either side is an edge.
        if not line.intersects(rooms) and line.intersects(sky):
            low.add(wall_id)

    if low:
        logger.info("%d wall(s) bound open sky and are built as parapets", len(low))
    return low


def _mostly_inside(
    polygon: list[tuple[float, float]], footprint: list[tuple[float, float]]
) -> bool:
    """Whether a region lies within the building rather than on the plot.

    A garden drawn inside the ground floor's outline is a courtyard and
    sits on that floor. One drawn beside the building is on the ground, and
    raising it to the floor level would leave it hanging at the front door.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES or len(footprint) < MIN_FOOTPRINT_VERTICES:
        return False

    try:
        region = Polygon(polygon)
        outline = Polygon(footprint)
        if not region.is_valid:
            region = region.buffer(0)
        if not outline.is_valid:
            outline = outline.buffer(0)
        if region.is_empty or region.area <= 0:
            return False
        return region.intersection(outline).area / region.area >= INSIDE_FRACTION
    except Exception as error:  # a malformed outline decides nothing
        logger.warning("could not place region against the footprint: %s", error)
        return False


def _bounds(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _to_metres(value: float, scale: float) -> float:
    return value / scale * FEET_TO_METRES


def _solid(vertices: list, faces: list) -> trimesh.Trimesh | None:
    """A closed mesh from explicit vertices and faces, wound outward.

    Written out rather than obtained from a convex hull or a plane slice,
    both of which need SciPy. A roof form that fails to build because an
    optional dependency is missing is worse than a few lines of index
    arithmetic, and these shapes are simple enough to state directly.

    Winding is checked rather than trusted: a mesh enclosing a negative
    volume is inside out, which is cheap to detect and cheap to correct.
    """
    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=float), faces=np.array(faces)
    )
    if mesh.is_empty:
        return None
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _dome(
    polygon: list[tuple[float, float]], base_ft: float, scale: float
) -> list[trimesh.Trimesh]:
    """A raised cap over a room, sitting on a low drum.

    Built from the room's bounding box rather than its outline. A dome is a
    surface of revolution and does not follow a ragged segmented polygon;
    forcing it to would give a lumpy shell rather than a dome.

    Assembled directly in model axes -- X across the page, Y up, Z down the
    page -- rather than built flat and rotated the way the slabs are, since
    a dome has no flat face to extrude from.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    left, top, right, bottom = _bounds(polygon)
    radius_x = _to_metres((right - left) / 2, scale)
    radius_z = _to_metres((bottom - top) / 2, scale)
    if radius_x <= 0 or radius_z <= 0:
        return []

    rise = min(radius_x, radius_z) * 2 * DOME_RISE_RATIO
    drum_m = DOME_DRUM_FT * FEET_TO_METRES
    base_m = base_ft * FEET_TO_METRES
    centre_x = _to_metres((left + right) / 2, scale)
    centre_z = _to_metres((top + bottom) / 2, scale)

    segments = DOME_SEGMENTS * 8
    rings = DOME_SEGMENTS * 2
    springing = base_m + drum_m

    # The drum: a closed elliptical cylinder from the roof up to the
    # springing line, so the dome rests on a base rather than growing
    # straight out of the deck.
    vertices = []
    for level in (base_m, springing):
        for step in range(segments):
            angle = 2 * np.pi * step / segments
            vertices.append(
                [
                    centre_x + radius_x * np.cos(angle),
                    level,
                    centre_z + radius_z * np.sin(angle),
                ]
            )
    vertices += [[centre_x, base_m, centre_z], [centre_x, springing, centre_z]]
    bottom_centre, top_centre = 2 * segments, 2 * segments + 1

    faces = []
    for step in range(segments):
        nxt = (step + 1) % segments
        faces += [
            [step, nxt, segments + nxt],
            [step, segments + nxt, segments + step],
            [bottom_centre, nxt, step],
            [top_centre, segments + step, segments + nxt],
        ]
    drum = _solid(vertices, faces)

    # The cap: a half ellipsoid in latitude rings, closed with a fan at the
    # top and a disc underneath so it is a solid rather than a shell.
    vertices = []
    for ring in range(rings):
        polar = (np.pi / 2) * ring / rings
        for step in range(segments):
            angle = 2 * np.pi * step / segments
            vertices.append(
                [
                    centre_x + radius_x * np.cos(polar) * np.cos(angle),
                    springing + rise * np.sin(polar),
                    centre_z + radius_z * np.cos(polar) * np.sin(angle),
                ]
            )
    vertices += [
        [centre_x, springing + rise, centre_z],
        [centre_x, springing, centre_z],
    ]
    apex, base_centre = rings * segments, rings * segments + 1

    faces = []
    for ring in range(rings - 1):
        here, above = ring * segments, (ring + 1) * segments
        for step in range(segments):
            nxt = (step + 1) % segments
            faces += [
                [here + step, here + nxt, above + nxt],
                [here + step, above + nxt, above + step],
            ]
    last = (rings - 1) * segments
    for step in range(segments):
        nxt = (step + 1) % segments
        faces.append([last + step, last + nxt, apex])
        faces.append([base_centre, nxt, step])
    cap = _solid(vertices, faces)

    return [part for part in (drum, cap) if part is not None]


def _sloped_roof(
    polygon: list[tuple[float, float]],
    base_ft: float,
    scale: float,
    rise_ratio: float,
    ridged: bool,
    thickness_ft: float = 0.0,
) -> list[trimesh.Trimesh]:
    """A roof whose top is not level: a ridged gable, or a single slope.

    A gable is a prism with its ridge along the room's longer side, since a
    gable spanning the long way would need impossibly long rafters. A single
    slope is a slab with its back edge lifted, given real thickness so a
    glazed plane has an edge rather than coming to a knife point.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    left, top, right, bottom = _bounds(polygon)
    width = _to_metres(right - left, scale)
    depth = _to_metres(bottom - top, scale)
    if width <= 0 or depth <= 0:
        return []

    rise = min(width, depth) * rise_ratio
    base_m = base_ft * FEET_TO_METRES
    x0, x1 = _to_metres(left, scale), _to_metres(right, scale)
    z0, z1 = _to_metres(top, scale), _to_metres(bottom, scale)

    if ridged:
        if width >= depth:
            middle = (z0 + z1) / 2
            ridge = ([x0, base_m + rise, middle], [x1, base_m + rise, middle])
        else:
            middle = (x0 + x1) / 2
            ridge = ([middle, base_m + rise, z0], [middle, base_m + rise, z1])

        vertices = [
            [x0, base_m, z0],
            [x1, base_m, z0],
            [x1, base_m, z1],
            [x0, base_m, z1],
            list(ridge[0]),
            list(ridge[1]),
        ]
        faces = [
            [0, 2, 1], [0, 3, 2],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 4], [2, 4, 5],
            [0, 4, 3], [1, 2, 5],
        ]
        solid = _solid(vertices, faces)
        return [solid] if solid is not None else []

    # A single slope falls forwards, so the back edge is the raised one.
    drop = (thickness_ft or GLAZED_THICKNESS_FT) * FEET_TO_METRES
    back, front = base_m + rise, base_m
    vertices = [
        [x0, back - drop, z0], [x1, back - drop, z0],
        [x1, front - drop, z1], [x0, front - drop, z1],
        [x0, back, z0], [x1, back, z0],
        [x1, front, z1], [x0, front, z1],
    ]
    faces = [
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [3, 0, 4], [3, 4, 7],
    ]
    solid = _solid(vertices, faces)
    return [solid] if solid is not None else []


def _roof_structure(
    polygon: list[tuple[float, float]],
    base_ft: float,
    scale: float,
    height_ft: float,
    capped: bool = False,
) -> list[trimesh.Trimesh]:
    """A mass standing on the roof: a chimney, a turret, a stair tower.

    A prism over the room's own outline rather than its bounding box, since
    these follow whatever shape is drawn -- a chimney breast is often not
    rectangular. ``capped`` finishes it with a shallow pyramid, which is
    what separates a turret from an unfinished lift overrun.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    parts = []
    shaft = slab_mesh(polygon, height_ft, base_ft, scale)
    if shaft is not None:
        parts.append(shaft)

    if capped:
        parts += _sloped_roof(
            polygon,
            base_ft + height_ft,
            scale,
            TOWER_CAP_RISE_RATIO,
            ridged=True,
        )
    return parts


def _water_tank(
    polygon: list[tuple[float, float]], base_ft: float, scale: float
) -> list[trimesh.Trimesh]:
    """An overhead tank on its stand.

    Standing clear of the deck rather than resting on it, which is how they
    are actually built and what stops it reading as a packing case left on
    the roof.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    parts = []
    # The legs are represented by a narrower block rather than four posts:
    # at the size a tank occupies on an elevation the difference is not
    # visible, and a shrunken outline cannot fail the way posts placed on a
    # ragged polygon's corners can.
    stand = slab_mesh(
        _expand_outline(polygon, -TANK_STAND_FT * scale / 2), TANK_STAND_FT, base_ft, scale
    )
    if stand is not None:
        parts.append(stand)

    tank = slab_mesh(polygon, TANK_HEIGHT_FT, base_ft + TANK_STAND_FT, scale)
    if tank is not None:
        parts.append(tank)
    return parts


def _canopy(
    polygon: list[tuple[float, float]], ceiling_ft: float, scale: float
) -> list[trimesh.Trimesh]:
    """A thin projecting cover, hung below the ceiling of its own storey.

    Belongs to the storey it is drawn on, not to the roof: a porch over the
    front door is at first floor soffit level, and putting it on the roof
    would leave the door uncovered and a slab floating three storeys up.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    slab = slab_mesh(polygon, CANOPY_THICKNESS_FT, ceiling_ft - CANOPY_DROP_FT, scale)
    return [slab] if slab is not None else []


def _ramp(
    polygon: list[tuple[float, float]], base_ft: float, scale: float
) -> list[trimesh.Trimesh]:
    """A sloped slab, falling along its longer direction.

    The same solid as a lean-to roof, laid at floor level instead of above
    it. Falling the long way because a ramp gains its gradient over length,
    and sloping it across the short side would be a step rather than a ramp.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    return _sloped_roof(
        polygon,
        base_ft,
        scale,
        RAMP_FALL_RATIO,
        ridged=False,
        thickness_ft=RAMP_THICKNESS_FT,
    )


def _headroom_box(
    polygon: list[tuple[float, float]], base_ft: float, scale: float
) -> list[trimesh.Trimesh]:
    """A small enclosure over a stairwell that reaches the roof.

    A flight arriving at roof level has to come up through something. Left
    off, the stair simply stops at the slab and the roofline loses the box
    that appears on every one of these houses.
    """
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    height_m = HEADROOM_HEIGHT_FT * FEET_TO_METRES
    thickness_px = HEADROOM_WALL_FT * scale

    parts = []
    for index in range(len(polygon)):
        wall = Wall(
            start=polygon[index],
            end=polygon[(index + 1) % len(polygon)],
            thickness=thickness_px,
        )
        parts.extend(_wall_parts(wall, [], height_m, scale, base_ft * FEET_TO_METRES))

    cap = slab_mesh(polygon, SLAB_THICKNESS_FT, base_ft + HEADROOM_HEIGHT_FT, scale)
    if cap is not None:
        parts.append(cap)
    return parts


def _roof_outline(
    footprint: list[tuple[float, float]], open_regions: list
) -> list[tuple[float, float]]:
    """The part of the top storey that is actually roofed.

    A parapet belongs around the roof rather than around the storey. Run
    around the whole footprint it stands over the open terrace too, three
    metres above the garden and attached to nothing.

    Falls back to the footprint whenever the difference cannot be taken or
    leaves nothing usable -- a parapet in the wrong place is a smaller
    fault than a roof with no edge at all.
    """
    if not open_regions:
        return footprint

    try:
        roofed = Polygon(footprint).buffer(0)
        for region in open_regions:
            if len(region) >= MIN_FOOTPRINT_VERTICES:
                roofed = roofed.difference(Polygon(region).buffer(0))

        if roofed.geom_type == "MultiPolygon":
            roofed = max(roofed.geoms, key=lambda part: part.area)
        if roofed.is_empty or roofed.area <= 0:
            return footprint
        return [(float(x), float(y)) for x, y in roofed.exterior.coords[:-1]]
    except Exception as error:
        logger.warning("could not trace the roofed area: %s", error)
        return footprint


def _parapet_walls(footprint: list[tuple[float, float]], base_ft: float, scale: float) -> list[Wall]:
    """The low wall running around a flat roof's edge."""
    if len(footprint) < MIN_FOOTPRINT_VERTICES:
        return []

    thickness_px = PARAPET_THICKNESS_FT * scale
    return [
        Wall(start=footprint[i], end=footprint[(i + 1) % len(footprint)], thickness=thickness_px)
        for i in range(len(footprint))
    ]


def walls_to_mesh(
    walls: list[Wall],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
    base_ft: float = 0.0,
    openings: list[Opening] | None = None,
) -> trimesh.Trimesh:
    """Extrude one floor's walls into a mesh, leaving gaps for openings.

    ``scale`` is pixels per foot, as measured by the calibration stage.
    ``base_ft`` lifts the floor, for stacking storeys.
    """
    if not walls:
        raise ValueError("no walls to extrude")

    height_m = wall_height_ft * FEET_TO_METRES
    base_m = base_ft * FEET_TO_METRES

    by_wall: dict[int, list[Opening]] = {}
    for opening in openings or []:
        if 0 <= opening.wall_id < len(walls):
            by_wall.setdefault(opening.wall_id, []).append(opening)

    parts: list[trimesh.Trimesh] = []
    for wall_id, wall in enumerate(walls):
        parts.extend(
            _wall_parts(wall, by_wall.get(wall_id, []), height_m, scale, base_m)
        )

    if not parts:
        raise ValueError("every wall was degenerate; nothing to extrude")

    mesh = trimesh.util.concatenate(parts)
    logger.info(
        "extruded %d wall(s) with %d opening(s) into %d faces",
        len(walls),
        sum(len(v) for v in by_wall.values()),
        len(mesh.faces),
    )
    return mesh


def floors_to_mesh(
    floors: list[FloorPlan],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
) -> trimesh.Trimesh:
    """Extrude several floors and stack them, lowest first.

    Each storey gets a floor slab under its walls, and the topmost gains a
    roof slab with a parapet around the edge. Walls alone leave a building
    open top and bottom, which reads as floating fragments rather than a
    house.
    """
    if not floors:
        raise ValueError("no floors to extrude")

    meshes: list[trimesh.Trimesh] = []

    for index, floor in enumerate(floors):
        base_ft = _storey_base_ft(index, wall_height_ft)

        slab = slab_mesh(floor.footprint, SLAB_THICKNESS_FT, base_ft, scale)
        if slab is not None:
            meshes.append(slab)
        elif floor.footprint:
            logger.warning("floor %d footprint gave no slab", index)

        if floor.walls:
            meshes.append(
                walls_to_mesh(
                    floor.walls,
                    wall_height_ft=wall_height_ft,
                    scale=scale,
                    base_ft=base_ft + SLAB_THICKNESS_FT,
                    openings=floor.openings,
                )
            )
        else:
            logger.warning("floor %d has no walls", index)

    # Cap the building: a roof slab over the top storey, with a parapet.
    roof_base_ft = _storey_base_ft(len(floors), wall_height_ft)
    top = floors[-1]
    roof = slab_mesh(top.footprint, SLAB_THICKNESS_FT, roof_base_ft, scale)
    if roof is not None:
        meshes.append(roof)
        parapet = _parapet_walls(top.footprint, roof_base_ft, scale)
        if parapet:
            meshes.append(
                walls_to_mesh(
                    parapet,
                    wall_height_ft=PARAPET_HEIGHT_FT,
                    scale=scale,
                    base_ft=roof_base_ft + SLAB_THICKNESS_FT,
                )
            )

    if not meshes:
        raise ValueError("no floor produced any geometry")

    logger.info("stacked %d floor(s) into %d part(s)", len(floors), len(meshes))
    return trimesh.util.concatenate(meshes)


def floors_to_parts(
    floors: list[FloorPlan],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
    page_size: tuple[int, int] | None = None,
    site: Landscaping | None = None,
) -> dict[str, list[trimesh.Trimesh]]:
    """Build the building's geometry grouped by what it is made of.

    Keys name a material rather than a floor, so the scene builder can give
    walls, slabs, the roof and glazing distinct appearances.

    ``site`` decides how much of a setting the building gets. Turned off
    entirely it stands alone against the sky, which is what a massing study
    wants and a presentation render does not.
    """
    site = site or Landscaping()
    if not floors:
        raise ValueError("no floors to extrude")

    height_m = wall_height_ft * FEET_TO_METRES
    parts: dict[str, list[trimesh.Trimesh]] = {
        "wall": [],
        "floor": [],
        "roof": [],
        "glass": [],
    }

    # The building stands on a plinth rather than flush with the ground.
    # Standing it directly on the site makes it look sunk into the earth.
    for index, floor in enumerate(floors):
        base_ft = _storey_base_ft(index, wall_height_ft, PLINTH_HEIGHT_FT)
        wall_base_m = (base_ft + SLAB_THICKNESS_FT) * FEET_TO_METRES

        features = group_by_feature(floor.rooms)

        # A double-height space, atrium or shaft is a hole through this
        # storey's floor. Slabbed over, the drawing's most dramatic space
        # becomes an ordinary ceiling.
        voids = [room.polygon for room in features.get("void", [])]

        # A slab is also the ceiling of the storey below, so it must reach at
        # least as far as that storey did. Built from its own footprint alone
        # it leaves rooms underneath open to the sky wherever the plan steps
        # in -- 75 sq ft of the reference building, at every level.
        outline = floor.footprint
        if index > 0 and floors[index - 1].footprint:
            outline = merge_regions([outline, floors[index - 1].footprint])
            outline = max(outline, key=len) if outline else floor.footprint

            # And because it is that ceiling, it has to stop where the storey
            # below is open to the sky. Only the roof did this, so an open
            # area on any storey but the top was sealed under the floor of
            # the storey above -- and the reach-back just above made it
            # worse, deliberately stretching the slab out over the very
            # projection a balcony sits on.
            #
            # But only where this storey does not itself stand. That
            # distinction is the whole of it, and it is the difference
            # between the two kinds of balcony a plan can draw:
            #
            #   projecting  the storey above stops short of it, so there is
            #               sky above and the slab must not reach out
            #   recessed    the storey above covers it, so its ceiling is
            #               that floor -- and cutting the hole anyway would
            #               take away the floor of the room upstairs
            #
            # Nothing here knows what a balcony is. It asks which regions
            # are open and which of them this storey stands on, and a
            # terrace, a courtyard, a light well and a roof pool all answer
            # the same way.
            voids = voids + _beyond(open_to_sky(floors[index - 1], scale), floor.footprint)

        slab = slab_mesh(outline, SLAB_THICKNESS_FT, base_ft, scale, holes=voids)
        if slab is not None:
            parts["floor"].append(slab)

        by_wall: dict[int, list[Opening]] = {}
        for opening in floor.openings:
            if 0 <= opening.wall_id < len(floor.walls):
                by_wall.setdefault(opening.wall_id, []).append(opening)

        # Walls around an open terrace are its edge, not its enclosure.
        parapets = _open_air_walls(floor, scale)
        parapet_m = PARAPET_HEIGHT_FT * FEET_TO_METRES

        for wall_id, wall in enumerate(floor.walls):
            openings = by_wall.get(wall_id, [])
            if wall_id in parapets:
                # No glazing in a parapet, and no opening cut through it:
                # a door onto a terrace is in the wall behind, not in the
                # rail around it.
                parts["wall"].extend(
                    _wall_parts(wall, [], parapet_m, scale, wall_base_m)
                )
                continue

            parts["wall"].extend(
                _wall_parts(wall, openings, height_m, scale, wall_base_m)
            )
            parts["glass"].extend(
                opening_panes(wall, openings, height_m, scale, wall_base_m)
            )
            parts.setdefault("frame", []).extend(
                opening_frames(wall, openings, height_m, scale, wall_base_m)
            )

        # Balconies and terraces are open to the air. Without a railing an
        # upper floor reads as a hole punched in the facade -- but only
        # along the edges that are actually a drop. See ``_guarded_edges``.
        standing_on = None
        if len(floor.footprint) >= MIN_FOOTPRINT_VERTICES:
            try:
                standing_on = Polygon(floor.footprint).buffer(0)
            except Exception as error:
                logger.warning("could not read the storey's extent: %s", error)

        for room in features.get("open", []):
            parts.setdefault("railing", []).extend(
                _railing_parts(room.polygon, wall_base_m, scale, standing_on)
            )

        # A void's edge is a drop on every side by definition: the floor
        # stops and there is a storey's fall beneath it. So it is guarded
        # all the way round regardless of what surrounds it.
        for room in features.get("void", []):
            parts.setdefault("railing", []).extend(
                _railing_parts(room.polygon, wall_base_m, scale)
            )

        # Each room's floor is finished according to what it is for, so a
        # bedroom reads differently from a kitchen. Laid just above the slab
        # rather than replacing it, so a missing label costs a finish rather
        # than the floor itself.
        for room in floor.rooms:
            # Resolved once: the printed name where there is one, the
            # segmenter's predicted type otherwise. On a plan with no text
            # the prediction is the only thing distinguishing a tiled
            # bathroom from a boarded bedroom.
            category = feature_for(room)
            if category in GROUND_COVERS or category == "void":
                continue
            finish = "wet" if category == "wet" else finish_for_room(room)
            patch = slab_mesh(
                room.polygon,
                FINISH_THICKNESS_FT,
                base_ft + SLAB_THICKNESS_FT,
                scale,
            )
            if patch is not None:
                parts.setdefault(finish, []).append(patch)

        # Stairs climb from this storey's floor to the next one's.
        for room in features.get("stairs", []):
            parts.setdefault("stairs", []).extend(
                _stair_parts(
                    room.polygon,
                    base_ft + SLAB_THICKNESS_FT,
                    wall_height_ft,
                    scale,
                )
            )

        # Ground cover for this storey: lawn, paving, and water recessed into
        # the surface. A pool laid flat on the ground reads as a blue carpet.
        #
        # Polygons come from segmented rooms where they exist, and from
        # dimension labels where they do not -- which is most outdoor areas,
        # since a lawn or a driveway is not a room the model can find.
        # A canopy and a ramp belong to the storey they are drawn on rather
        # than to the roof. A porch over the front door sits at first floor
        # soffit level; moved to the roof it would leave the door uncovered
        # and hang a slab three storeys up.
        floor_ft = base_ft + SLAB_THICKNESS_FT
        for room in features.get("canopy", []):
            parts.setdefault("canopy", []).extend(
                _canopy(room.polygon, floor_ft + wall_height_ft, scale)
            )
        for room in features.get("ramp", []):
            parts.setdefault("stairs", []).extend(_ramp(room.polygon, floor_ft, scale))

        # A cover belongs to the storey it was drawn on, at that storey's
        # own floor level. The one exception is the plot around the
        # building: a lawn beside the house lies on the ground, not on the
        # plinth the house stands on, and lifting it would leave the garden
        # hanging in the air at the front door.
        #
        # So the level is decided per region by where it actually is rather
        # than per storey. Deciding it per storey -- ground floor at site
        # level, everything else at its own -- buried any garden drawn
        # inside the ground floor's own outline underneath the plinth.
        storey_ft = base_ft + SLAB_THICKNESS_FT
        for category in GROUND_COVERS if site.planting else ():
            polygons = [room.polygon for room in features.get(category, [])]
            polygons += floor.labelled_regions.get(category, [])
            if category == "lawn":
                polygons += floor.planting

            # Colour and labels frequently find the same area, so overlapping
            # outlines are merged before anything is built.
            for polygon in merge_regions(polygons):
                surface_ft = storey_ft
                if index == 0 and not _mostly_inside(polygon, floor.footprint):
                    surface_ft = 0.0

                if category == "water":
                    # A pool on the ground storey is excavated: there is
                    # earth under it, whether it sits on the plot or in a
                    # courtyard. One on any storey above is built up on the
                    # slab, because sinking it would drop it through the
                    # ceiling of the room beneath.
                    top_ft = surface_ft if index == 0 else surface_ft + POOL_DEPTH_FT
                    patch = slab_mesh(
                        polygon, POOL_DEPTH_FT, top_ft - POOL_DEPTH_FT, scale
                    )
                else:
                    patch = slab_mesh(polygon, COVER_THICKNESS_FT, surface_ft, scale)
                if patch is not None:
                    parts.setdefault(category, []).append(patch)

    # The plinth itself: the ground floor's footprint, oversailing the walls
    # so the elevation has a visible base, with steps at every external door.
    ground = floors[0]
    if ground.footprint:
        plinth = _expand_outline(ground.footprint, PLINTH_OVERHANG_FT * scale)
        block = slab_mesh(plinth, PLINTH_HEIGHT_FT, 0.0, scale)
        if block is not None:
            parts.setdefault("plinth", []).append(block)

        for opening in ground.openings:
            if opening.type != "door":
                continue
            if not 0 <= opening.wall_id < len(ground.walls):
                continue
            wall = ground.walls[opening.wall_id]
            if _near_outline(wall, ground.footprint, EXTERNAL_DOOR_MARGIN_FT * scale):
                parts.setdefault("plinth", []).extend(
                    _entrance_steps(wall, opening, PLINTH_HEIGHT_FT, scale)
                )

    roof_base_ft = _storey_base_ft(len(floors), wall_height_ft, PLINTH_HEIGHT_FT)
    top = floors[-1]

    # A planted terrace is open to the sky. Roofing over it hides the garden
    # entirely and makes the top storey read as a sealed box.
    #
    # Every open area counts, not only the ones found by colour. The
    # reference sheet's colour blobs cover 48% of the top storey while the
    # room it labels TERRACE GARDEN covers 73%, so roofing to the colour
    # alone left a quarter of the garden under a slab.
    sky = open_to_sky(top, scale)

    roof = slab_mesh(
        top.footprint, SLAB_THICKNESS_FT, roof_base_ft, scale, holes=sky
    )
    if roof is not None:
        parts["roof"].append(roof)
        parapet_base_ft = roof_base_ft + SLAB_THICKNESS_FT
        # Around the roof that exists, not around the whole storey: a
        # parapet standing over an open terrace is a wall in mid-air.
        parapet = _parapet_walls(_roof_outline(top.footprint, sky), roof_base_ft, scale)
        for wall in parapet:
            parts["roof"].extend(
                _wall_parts(
                    wall,
                    [],
                    PARAPET_HEIGHT_FT * FEET_TO_METRES,
                    scale,
                    parapet_base_ft * FEET_TO_METRES,
                )
            )

        # A coping caps the parapet and oversails it slightly, throwing water
        # clear of the wall below. It also gives the roofline a defined edge
        # instead of a raw extrusion, which is what makes a parapet read.
        # Following the roof as the parapet does. Traced round the whole
        # storey it left a thin rail hanging in the air over the terrace,
        # capping a parapet that is not there.
        roofed = _roof_outline(top.footprint, sky)
        coping_outline = _expand_outline(roofed, COPING_OVERHANG_FT * scale)
        inner = _expand_outline(
            roofed, -(PARAPET_THICKNESS_FT + COPING_OVERHANG_FT) * scale
        )
        coping = slab_mesh(
            coping_outline,
            COPING_THICKNESS_FT,
            parapet_base_ft + PARAPET_HEIGHT_FT,
            scale,
            holes=[inner] if len(inner) >= MIN_FOOTPRINT_VERTICES else None,
        )
        if coping is not None:
            parts.setdefault("coping", []).append(coping)

    top_features = group_by_feature(top.rooms)

    # A flight reaching the top storey needs headroom above it, which on a
    # flat roof is a small enclosure -- the box that appears on every roofline
    # in the reference elevations.
    for room in top_features.get("stairs", []):
        parts.setdefault("roof", []).extend(
            _headroom_box(room.polygon, roof_base_ft + SLAB_THICKNESS_FT, scale)
        )

    # Roofs that are not flat. A flat slab with a parapet is the common case
    # on these houses and was the only thing built, which left a drawing
    # marked DOME or GLASS ROOF or SLOPING ROOF looking like every other.
    #
    # All three are raised over rooms named on the *top* storey, because
    # that is the plan a roof feature is drawn on. A dome marked on a ground
    # floor temple in a three storey house is not directly under the sky,
    # and guessing which part of the roof it belongs beneath would be worse
    # than leaving it off.
    deck_ft = roof_base_ft + SLAB_THICKNESS_FT
    for room in top_features.get("dome", []):
        parts.setdefault("dome", []).extend(_dome(room.polygon, deck_ft, scale))

    # Masses standing on the roof. A tower is capped, which is what
    # separates a turret from an unfinished lift overrun.
    for room in top_features.get("chimney", []):
        parts.setdefault("chimney", []).extend(
            _roof_structure(room.polygon, deck_ft, scale, CHIMNEY_HEIGHT_FT)
        )

    # Kept out of "roof": a turret is masonry carried up from the building
    # below, and in the deck's grey it read as plant housing.
    for room in top_features.get("tower", []):
        parts.setdefault("tower", []).extend(
            _roof_structure(room.polygon, deck_ft, scale, TOWER_HEIGHT_FT, capped=True)
        )

    for room in top_features.get("tank", []):
        parts.setdefault("tank", []).extend(_water_tank(room.polygon, deck_ft, scale))

    # Kept apart from the flat deck so it takes the tiled finish. Merged in
    # with "roof" it was invisible -- a low slope, the same grey as the deck
    # it stands on, half hidden behind the parapet.
    for room in top_features.get("pitched", []):
        parts.setdefault("pitched", []).extend(
            _sloped_roof(room.polygon, deck_ft, scale, PITCH_RISE_RATIO, ridged=True)
        )

    # Glazing goes in with the windows so it picks up the same transparent
    # material, rather than reading as a solid panel over the room.
    for room in top_features.get("glazed", []):
        parts.setdefault("glass", []).extend(
            _sloped_roof(
                room.polygon,
                deck_ft,
                scale,
                GLAZED_RISE_RATIO,
                ridged=False,
                thickness_ft=GLAZED_THICKNESS_FT,
            )
        )

    # Merged rather than assigned: the site block also produces lawn and
    # paving, and replacing the keys would discard whatever the storeys
    # contributed under the same names.
    if site.ground:
        for name, meshes in _site_parts(
            floors, scale, wall_height_ft, page_size, boundary=site.boundary
        ).items():
            parts.setdefault(name, []).extend(meshes)

    return {name: meshes for name, meshes in parts.items() if meshes}


def _site_parts(
    floors: list[FloorPlan],
    scale: float,
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    page_size: tuple[int, int] | None = None,
    boundary: bool = True,
) -> dict[str, list[trimesh.Trimesh]]:
    """The ground the building stands on, with lawn, paving and a boundary.

    Sits just below zero so the building's own ground-floor slab reads as
    resting on it rather than z-fighting with it.
    """
    outline = site_outline(
        [floor.footprint for floor in floors if floor.footprint],
        margin_px=SITE_MARGIN_FT * scale,
        page_size=page_size,
    )
    if not outline:
        return {}

    parts: dict[str, list[trimesh.Trimesh]] = {
        "ground": [],
        "lawn": [],
        "paving": [],
        "boundary": [],
    }

    ground = slab_mesh(outline, SITE_THICKNESS_FT, -SITE_THICKNESS_FT, scale)
    if ground is not None:
        parts["ground"].append(ground)

    # A compound wall around the plot, as the reference elevations show.
    for wall in boundary_walls(outline, BOUNDARY_THICKNESS_FT * scale) if boundary else ():
        parts["boundary"].extend(
            _wall_parts(
                wall, [], BOUNDARY_HEIGHT_FT * FEET_TO_METRES, scale, 0.0
            )
        )

    return parts


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> Path:
    """Write a mesh as binary glTF, the format web viewers expect."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    logger.info("wrote %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
