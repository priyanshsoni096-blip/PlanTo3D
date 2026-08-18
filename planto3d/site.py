"""Ground, lawn and paving around the building.

A building rendered without a site floats in space, which reads as a massing
study rather than a house. The plan already says where the ground cover goes:
rooms labelled LANDSCAPE, TERRACE GARDEN, PARKING and the like are outdoor
areas the pipeline has already found and named, so planting and paving are
placed from measured polygons rather than invented.

Classification is by keyword on the OCR label. That is deliberately loose --
labels arrive misread ("PLAV-AREA", "UFT") and vary between drafting offices,
so an unrecognised label falls back to plain ground rather than guessing.
"""

import logging

import numpy as np

from planto3d.geometry_types import Room

logger = logging.getLogger(__name__)

# Keywords are matched against the room label, longest first, so "TERRACE
# GARDEN" is not caught by a bare "TERRACE" rule.
GROUND_COVER = {
    "lawn": ("LANDSCAPE", "GARDEN", "LAWN", "GREEN"),
    "paving": ("PARKING", "DRIVEWAY", "PORCH", "VERANDAH", "DECK", "PASSAGE"),
}

# Rooms open to the air at an upper storey. These need a railing along their
# outer edge; without one an upper floor reads as a hole in the facade.
#
# "BAL" is here because OCR routinely truncates the word: on the reference
# first floor the label comes back as exactly "BAL", which matched none of
# the longer spellings and left every balcony unrailed.
OPEN_EDGE_KEYWORDS = ("BAL", "TERRACE", "DECK", "VERANDA", "PORCH")
# Railing dimensions, in feet. Slim and waist-high, as a rail should be.
RAILING_HEIGHT_FT = 3.4
# Slim, as a balustrade is. Thicker than this and a rail reads as a parapet
# wall, which changes how the whole facade looks.
RAILING_THICKNESS_FT = 0.15
# The top rail's depth, and how far apart the posts beneath it stand.
RAIL_DEPTH_FT = 0.3
POST_SPACING_FT = 4.0
# How far a pool is sunk below the surface around it. A pool laid flat on the
# ground reads as a blue carpet rather than water.
POOL_DEPTH_FT = 4.0

# Fallback margin around the building when the plot extent is unknown.
SITE_MARGIN_FT = 14.0
# Thicknesses in feet: the site slab, and the cover laid on top of it.
SITE_THICKNESS_FT = 0.6
COVER_THICKNESS_FT = 0.12
# Compound wall around the plot: a boundary at roughly chest height.
BOUNDARY_HEIGHT_FT = 6.0
BOUNDARY_THICKNESS_FT = 0.8
# The plot is inset from the sheet edge so the wall sits inside the drawing
# rather than straddling its border.
PLOT_INSET_PX = 6.0


def classify_cover(label: str) -> str | None:
    """Which ground cover a room label implies, or None if it is indoors."""
    if not label:
        return None

    upper = label.upper()
    for cover, keywords in GROUND_COVER.items():
        if any(keyword in upper for keyword in keywords):
            return cover
    return None


def outdoor_rooms(rooms: list[Room]) -> dict[str, list[Room]]:
    """Group the rooms that describe ground cover, by the cover they imply."""
    grouped: dict[str, list[Room]] = {}
    for room in rooms:
        cover = classify_cover(room.label)
        if cover:
            grouped.setdefault(cover, []).append(room)

    if grouped:
        logger.info(
            "ground cover from labels: %s",
            {cover: len(rooms) for cover, rooms in grouped.items()},
        )
    return grouped


def site_outline(
    footprints: list[list[tuple[float, float]]],
    margin_px: float,
    page_size: tuple[int, int] | None = None,
) -> list[tuple[float, float]]:
    """The plot the building sits on.

    Prefers the drawing's own extent: the sheet is cropped to the drawing
    frame, and that frame encloses the whole site -- setbacks, driveway and
    garden included -- so it is the plot, measured rather than assumed. Falls
    back to a margin around the building when no page size is given.

    Rectangular either way: a plot is a rectangle, and a ground slab that
    hugs the walls looks like a plinth.
    """
    if page_size is not None:
        width, height = page_size
        inset = PLOT_INSET_PX
        return [
            (inset, inset),
            (width - inset, inset),
            (width - inset, height - inset),
            (inset, height - inset),
        ]

    points = [point for footprint in footprints for point in footprint]
    if not points:
        return []

    array = np.array(points, dtype=float)
    left, top = array.min(axis=0) - margin_px
    right, bottom = array.max(axis=0) + margin_px
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def has_open_edge(label: str) -> bool:
    """Whether a room is open to the air and so needs a railing."""
    return bool(label) and any(
        keyword in label.upper() for keyword in OPEN_EDGE_KEYWORDS
    )


def railed_rooms(rooms: list[Room]) -> list[Room]:
    """Rooms that need a railing around them -- balconies and terraces."""
    railed = [room for room in rooms if has_open_edge(room.label)]
    if railed:
        logger.info("railing %d open-edged room(s)", len(railed))
    return railed


def boundary_walls(outline: list[tuple[float, float]], thickness_px: float) -> list:
    """A compound wall running around the plot's edge."""
    from planto3d.geometry_types import Wall

    if len(outline) < 3:
        return []

    return [
        Wall(
            start=outline[i],
            end=outline[(i + 1) % len(outline)],
            thickness=thickness_px,
        )
        for i in range(len(outline))
    ]
