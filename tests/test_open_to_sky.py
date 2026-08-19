"""Anything open to the sky, on any storey, treated the same way.

The terrace garden was found sitting at the bottom of a three-metre well
of masonry with a parapet on top, so it read a full floor below where the
plan put it. That was never about gardens. A balcony, a roof deck, a
courtyard, a rooftop pool and a parking bay are all the same case: sky
above, an edge rather than an enclosure around, and a floor belonging to
the storey they were drawn on.

Parameterised over the whole vocabulary rather than written against one
feature, because the last version of this bug was found by eye on one
building and the next one will not be.
"""

import pytest
from shapely.geometry import Point, Polygon

from planto3d.extrude import (
    FEET_TO_METRES,
    PARAPET_HEIGHT_FT,
    SLAB_THICKNESS_FT,
    _storey_base_ft,
    floors_to_parts,
)
from planto3d.features import OPEN_TO_SKY, classify, is_open_to_sky
from planto3d.geometry_types import FloorPlan, Room, Wall
from planto3d.site import PLINTH_HEIGHT_FT

SCALE = 20.0
HEIGHT = 9.0
OUTLINE = [(0.0, 0.0), (800.0, 0.0), (800.0, 500.0), (0.0, 500.0)]

# The right-hand half is open; the left is a room, so the storey has both.
OPEN_HALF = [(420.0, 40.0), (760.0, 40.0), (760.0, 460.0), (420.0, 460.0)]
ROOM_HALF = [(40.0, 40.0), (380.0, 40.0), (380.0, 460.0), (40.0, 460.0)]

# One label per open category, drawn from what a plan actually prints.
OPEN_LABELS = [
    "TERRACE GARDEN",
    "BALCONY",
    "OPEN TERRACE",
    "SWIMMING POOL",
    "COURTYARD",
    "PARKING",
    "SIT OUT",
]


def _walls(outline):
    return [
        Wall(start=outline[i], end=outline[(i + 1) % len(outline)], thickness=12.0)
        for i in range(len(outline))
    ]


def _floor(open_label=None):
    rooms = [Room(polygon=list(ROOM_HALF), label="BEDROOM")]
    walls = _walls(OUTLINE)
    if open_label:
        rooms.append(Room(polygon=list(OPEN_HALF), label=open_label))
        # The terrace's own edge, drawn exactly as a wall is.
        walls += _walls(OPEN_HALF)
    return FloorPlan(walls=walls, footprint=list(OUTLINE), rooms=rooms)


def _storey_floor_m(index):
    return (_storey_base_ft(index, HEIGHT, PLINTH_HEIGHT_FT) + SLAB_THICKNESS_FT) * FEET_TO_METRES


class TestTheVocabularyAgrees:
    @pytest.mark.parametrize("label", OPEN_LABELS)
    def test_every_open_label_is_recognised_as_open(self, label):
        assert classify(label) in OPEN_TO_SKY, f"{label} -> {classify(label)}"

    def test_a_room_is_not_open_just_because_it_is_large(self):
        assert not is_open_to_sky(Room(polygon=list(ROOM_HALF), label="BEDROOM"))

    def test_a_double_height_space_is_still_roofed(self):
        # A void is a hole through a floor, which is a different thing from
        # a space with sky above it. A double-height living room is roofed.
        assert "void" not in OPEN_TO_SKY

    def test_a_predicted_outdoor_room_counts_without_any_label(self):
        # Most plans print no names at all, so the rule has to work off the
        # segmenter's type or it does not work on most plans.
        assert is_open_to_sky(Room(polygon=list(OPEN_HALF), category="outdoor"))


@pytest.mark.parametrize("label", OPEN_LABELS)
class TestOnTheTopStorey:
    def _parts(self, label):
        return floors_to_parts(
            [_floor(), _floor(open_label=label)], wall_height_ft=HEIGHT, scale=SCALE
        )

    def test_nothing_is_roofed_over_it(self, label):
        # Tested on where the roof's geometry actually is, not on its
        # bounding box: an L-shaped roof wrapping an open corner has a box
        # covering the whole storey while roofing none of the corner.
        parts = self._parts(label)
        # Shrunk, so the hole's own boundary vertices -- which sit exactly
        # on the edge -- are not counted as being over it.
        inside = Polygon(OPEN_HALF).buffer(0).buffer(-24.0)

        for mesh in parts.get("roof", []):
            for x, _, z in mesh.vertices:
                page = Point(x / FEET_TO_METRES * SCALE, z / FEET_TO_METRES * SCALE)
                assert not inside.contains(page), (
                    f"{label} has roof at ({page.x:.0f}, {page.y:.0f}), "
                    "which is inside the open area"
                )

    def test_its_edge_is_a_parapet_not_a_storey_high_wall(self, label):
        # The failure this guards: three metres of blank masonry round an
        # open space, so it reads a full floor below where the plan put it.
        parts = self._parts(label)
        top_floor = _storey_floor_m(1)

        heights = [
            mesh.bounds[1][1] - top_floor
            for mesh in parts["wall"]
            if mesh.bounds[0][1] >= top_floor - 0.1
        ]
        assert heights, f"no top-storey walls built for {label}"
        assert min(heights) <= PARAPET_HEIGHT_FT * FEET_TO_METRES + 0.1


def test_a_storey_with_nothing_open_is_left_alone():
    # The rule must cost nothing on the ordinary case, which is every
    # storey that is simply rooms.
    plain = floors_to_parts([_floor(), _floor()], wall_height_ft=HEIGHT, scale=SCALE)
    top_floor = _storey_floor_m(1)

    heights = [
        mesh.bounds[1][1] - top_floor
        for mesh in plain["wall"]
        if mesh.bounds[0][1] >= top_floor - 0.1
    ]
    assert min(heights) > PARAPET_HEIGHT_FT * FEET_TO_METRES
