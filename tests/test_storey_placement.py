"""Anything drawn on a storey must be built on that storey.

This is one invariant covering a whole family of bugs. A terrace garden
that lands a floor low, a rooftop pool sunk through the ceiling below it, a
porch hoisted to roof level -- each was found separately, and each is the
same mistake: a feature placed at a height decided by its kind rather than
by the plan it was read from.

The tests are written per feature and per storey rather than against one
building, because the failure only ever shows on the storey nobody checked.
"""

import pytest

from planto3d.extrude import (
    FEET_TO_METRES,
    SLAB_THICKNESS_FT,
    _storey_base_ft,
    floors_to_parts,
)
from planto3d.geometry_types import FloorPlan, Room, Wall
from planto3d.site import PLINTH_HEIGHT_FT

SCALE = 20.0
HEIGHT = 9.0
OUTLINE = [(0.0, 0.0), (600.0, 0.0), (600.0, 500.0), (0.0, 500.0)]
# Well inside the outline, so it reads as part of the building rather than
# as the plot around it.
INSIDE = [(120.0, 100.0), (480.0, 100.0), (480.0, 400.0), (120.0, 400.0)]


def _floor(rooms=None):
    return FloorPlan(
        walls=[
            Wall(start=OUTLINE[i], end=OUTLINE[(i + 1) % 4], thickness=12.0)
            for i in range(4)
        ],
        footprint=list(OUTLINE),
        rooms=rooms or [],
    )


def _storey_floor_m(index: int) -> float:
    """Where storey ``index``'s walking surface sits, in metres."""
    base = _storey_base_ft(index, HEIGHT, PLINTH_HEIGHT_FT)
    return (base + SLAB_THICKNESS_FT) * FEET_TO_METRES


def _build(storeys: int, on: int, label: str):
    floors = [_floor() for _ in range(storeys)]
    floors[on] = _floor([Room(polygon=list(INSIDE), label=label)])
    return floors_to_parts(floors, wall_height_ft=HEIGHT, scale=SCALE)


@pytest.mark.parametrize("storey", [0, 1, 2])
@pytest.mark.parametrize(
    "label, part",
    [
        ("TERRACE GARDEN", "lawn"),
        ("SWIMMING POOL", "water"),
        ("COURTYARD", "paving"),
    ],
)
def test_a_cover_is_built_on_the_storey_it_was_drawn_on(label, part, storey):
    parts = _build(3, storey, label)

    assert part in parts, f"{label} on storey {storey + 1} built nothing"
    lowest = min(mesh.bounds[0][1] for mesh in parts[part])
    highest = max(mesh.bounds[1][1] for mesh in parts[part])
    floor = _storey_floor_m(storey)

    # The cover has to *meet* its storey's floor rather than sit above it:
    # a lawn lies on the slab, a terrace pool stands on it, and a pool on
    # the ground is dug into it. All three touch that level, and nothing
    # belonging to another storey would.
    assert lowest - 0.05 <= floor <= highest + 0.05, (
        f"{label} on storey {storey + 1} spans {lowest:.2f}..{highest:.2f}m, "
        f"but that storey's floor is at {floor:.2f}m"
    )


def test_a_terrace_pool_is_not_sunk_through_the_ceiling_below():
    # A pool at ground level is excavated. One on a terrace is built up on
    # the slab, because there is a room under it.
    parts = _build(3, 2, "SWIMMING POOL")
    lowest = min(mesh.bounds[0][1] for mesh in parts["water"])

    assert lowest >= _storey_floor_m(2) - 0.05


def test_a_pool_on_the_ground_is_excavated_rather_than_stood_on_the_lawn():
    # Earth under it either way, whether it sits on the plot or in a
    # courtyard, so it is dug rather than built up.
    parts = _build(1, 0, "SWIMMING POOL")
    lowest = min(mesh.bounds[0][1] for mesh in parts["water"])

    assert lowest < _storey_floor_m(0) - 0.1


def test_planting_beside_the_building_stays_on_the_ground():
    # The exception the rule needs: a lawn on the plot lies on the ground,
    # not on the plinth the house stands on. Raised, the garden hangs in
    # the air at the front door.
    beside = [(700.0, 100.0), (900.0, 100.0), (900.0, 400.0), (700.0, 400.0)]
    floors = [_floor([Room(polygon=beside, label="LANDSCAPE")])]

    parts = floors_to_parts(floors, wall_height_ft=HEIGHT, scale=SCALE)
    lowest = min(mesh.bounds[0][1] for mesh in parts["lawn"])

    assert lowest == pytest.approx(0.0, abs=0.05)


def test_a_courtyard_inside_the_ground_floor_sits_on_its_floor():
    # And the other half of that exception: drawn inside the outline it is
    # part of the building, and burying it under the plinth loses it.
    parts = _build(1, 0, "COURTYARD")
    lowest = min(mesh.bounds[0][1] for mesh in parts["paving"])

    assert lowest > 0.0


@pytest.mark.parametrize("storey", [0, 1])
def test_a_canopy_belongs_to_its_own_storey_not_the_roof(storey):
    parts = _build(2, storey, "PORTICO")

    assert "canopy" in parts
    highest = max(mesh.bounds[1][1] for mesh in parts["canopy"])

    # Below the ceiling of its own storey, and above its floor.
    assert _storey_floor_m(storey) < highest < _storey_floor_m(storey + 1) + 0.1


def test_features_on_one_storey_do_not_appear_on_another():
    # The check that would have caught the terrace garden landing a floor
    # low: build the same feature on each storey in turn and confirm the
    # height moves with it.
    heights = []
    for storey in range(3):
        parts = _build(3, storey, "TERRACE GARDEN")
        heights.append(min(mesh.bounds[0][1] for mesh in parts["lawn"]))

    assert heights[0] < heights[1] < heights[2]
