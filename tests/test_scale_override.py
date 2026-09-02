"""A scale the user states outright, rather than one inferred.

Every inferred route rests on an assumption about a standard element --
a 2'6" door, a 9" wall -- and those assumptions are what the residual
error is made of: measured over 30 ground-truthed sheets, the real wall
thickness runs from 0.478 to 1.176 ft around a median of 0.648, so no
constant fits every building. A stated measurement has no such floor.
"""

import pytest

from planto3d.calibrate import scale_from_known_room
from planto3d.geometry_types import Room


def _room(width_px: float, height_px: float) -> Room:
    return Room(
        polygon=[(0.0, 0.0), (width_px, 0.0), (width_px, height_px), (0.0, height_px)]
    )


def test_a_square_room_gives_its_own_scale():
    # 200 px across a room the user says is 10 ft: 20 px per foot.
    assert scale_from_known_room(_room(200.0, 200.0), 10.0, 10.0) == pytest.approx(20.0)


def test_scale_comes_from_area_not_one_edge():
    # 200x100 px stated as 10x5 ft is 20 px/ft either way; taking one edge
    # would also give 20 here, so use a case where the room is drawn out of
    # proportion to what was stated and the area still settles it.
    # 400x100 px (40000) stated as 10x10 ft (100) -> sqrt(400) = 20.
    assert scale_from_known_room(_room(400.0, 100.0), 10.0, 10.0) == pytest.approx(20.0)


def test_a_degenerate_room_yields_nothing():
    # A zero-area polygon cannot state a scale, and returning a huge or
    # zero number here would resize the whole building.
    assert scale_from_known_room(_room(0.0, 0.0), 10.0, 10.0) is None


def test_a_zero_size_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        scale_from_known_room(_room(200.0, 200.0), 0.0, 10.0)


def test_a_negative_size_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        scale_from_known_room(_room(200.0, 200.0), 10.0, -4.0)
