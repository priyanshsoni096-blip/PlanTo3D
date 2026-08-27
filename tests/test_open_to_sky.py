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


# A storey that stops short of the open half, so whatever is below it
# projects out from under the building rather than being set into it.
SET_BACK = [(0.0, 0.0), (400.0, 0.0), (400.0, 500.0), (0.0, 500.0)]


def _storey(open_label=None, outline=None):
    """One storey, optionally standing on a smaller footprint than the rest."""
    outline = outline or OUTLINE
    rooms = [Room(polygon=list(ROOM_HALF), label="BEDROOM")]
    walls = _walls(outline)
    if open_label:
        rooms.append(Room(polygon=list(OPEN_HALF), label=open_label))
        walls += _walls(OPEN_HALF)
    return FloorPlan(walls=walls, footprint=list(outline), rooms=rooms)


def _covers(mesh, page_point):
    """Does this mesh have surface over ``page_point``?

    Asked of the mesh's triangles rather than its vertices. A slab that
    covers an area whole has no vertex anywhere near it -- its corners are
    out at the building's edge -- so testing vertices passes a solid slab
    and proves nothing. This projects each triangle back to page
    coordinates and asks whether any of them lies over the point.
    """
    flat = [
        (x / FEET_TO_METRES * SCALE, z / FEET_TO_METRES * SCALE)
        for x, _, z in mesh.vertices
    ]
    for a, b, c in mesh.faces:
        triangle = Polygon([flat[a], flat[b], flat[c]]).buffer(0)
        if triangle.contains(page_point):
            return True
    return False


@pytest.mark.parametrize("label", OPEN_LABELS)
class TestOnAStoreyThatIsNotTheTop:
    """The same rule one storey down, where a balcony can be either kind.

    Only the roof ever cut holes for open areas, so an open space on any
    storey but the top was sealed under the floor of the storey above --
    and the slab is deliberately stretched out to cover whatever the
    storey below covered, which reached it out over the very projection a
    balcony stands on.

    Which of the two cases applies is decided by the storey above, not by
    the label:

        projecting  it stops short, so there is sky above
        recessed    it covers the area, so that floor is the ceiling

    Both are drawn "BALCONY". Getting the second one wrong would punch a
    hole through the floor of the room upstairs, so both are tested.
    """

    def _centre(self):
        return Polygon(OPEN_HALF).centroid

    def test_a_projecting_balcony_has_sky_above_it(self, label):
        parts = floors_to_parts(
            [
                _storey(),
                _storey(open_label=label),
                _storey(outline=SET_BACK),
            ],
            wall_height_ft=HEIGHT,
            scale=SCALE,
        )
        overhead = _storey_floor_m(1) + 0.5
        for mesh in parts.get("floor", []) + parts.get("roof", []):
            if mesh.bounds[1][1] < overhead:
                continue
            assert not _covers(mesh, self._centre()), (
                f"a projecting {label} is covered by a slab at "
                f"{mesh.bounds[0][1]:.1f}m"
            )

    def test_a_projecting_balcony_still_has_a_floor(self, label):
        # The other way to pass the test above is to cut the hole through
        # every slab, which leaves the balcony with nothing to stand on.
        parts = floors_to_parts(
            [_storey(), _storey(open_label=label), _storey(outline=SET_BACK)],
            wall_height_ft=HEIGHT,
            scale=SCALE,
        )
        own_floor = _storey_floor_m(1)
        assert any(
            _covers(mesh, self._centre())
            for mesh in parts.get("floor", [])
            if mesh.bounds[1][1] <= own_floor + 0.1
        ), f"a projecting {label} has no floor under it"

    def test_a_recessed_balcony_keeps_the_floor_above_it(self, label):
        # Set into the building rather than projecting from it, so the
        # storey above stands over it and that slab is its ceiling.
        # Cutting the hole anyway would take the floor out from under the
        # room upstairs.
        parts = floors_to_parts(
            [_storey(), _storey(open_label=label), _storey()],
            wall_height_ft=HEIGHT,
            scale=SCALE,
        )
        overhead = _storey_floor_m(1) + 0.5
        assert any(
            _covers(mesh, self._centre())
            for mesh in parts.get("floor", [])
            if mesh.bounds[1][1] >= overhead
        ), f"a recessed {label} left the room above it with no floor"


@pytest.mark.parametrize("label", OPEN_LABELS)
def test_the_roof_really_is_open_over_a_top_storey_terrace(label):
    """The roof, asked about its surface rather than its vertices."""
    parts = floors_to_parts(
        [_storey(), _storey(open_label=label)], wall_height_ft=HEIGHT, scale=SCALE
    )
    centre = Polygon(OPEN_HALF).centroid
    for mesh in parts.get("roof", []):
        assert not _covers(mesh, centre), f"{label} has roof over its centre"
