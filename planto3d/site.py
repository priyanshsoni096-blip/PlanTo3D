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

# The site slab extends this far beyond the building on every side, in feet.
SITE_MARGIN_FT = 14.0
# Thicknesses in feet: the site slab, and the cover laid on top of it.
SITE_THICKNESS_FT = 0.6
COVER_THICKNESS_FT = 0.12


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
) -> list[tuple[float, float]]:
    """A rectangle enclosing every floor's footprint, plus a margin.

    Rectangular rather than following the building's outline: a site is a
    plot, and a ground slab that hugs the walls looks like a plinth.
    """
    points = [point for footprint in footprints for point in footprint]
    if not points:
        return []

    array = np.array(points, dtype=float)
    left, top = array.min(axis=0) - margin_px
    right, bottom = array.max(axis=0) + margin_px
    return [(left, top), (right, top), (right, bottom), (left, bottom)]
