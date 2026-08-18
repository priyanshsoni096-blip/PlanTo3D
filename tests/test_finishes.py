"""Interior floors finished according to what each room is for."""

import pytest

from planto3d.extrude import floors_to_parts
from planto3d.features import DEFAULT_FINISH, finish_for
from planto3d.geometry_types import FloorPlan, Room, Wall

SCALE = 20.0
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


@pytest.mark.parametrize(
    "label, expected",
    [
        # Lived in: warm.
        ("BEDROOM 15'0\"X18'0\"", "timber"),
        ("MASTER BEDROOM", "timber"),
        ("STUDY", "timber"),
        ("DRESS", "timber"),
        ("TEMPLE", "timber"),
        # Worked in: harder and cooler.
        ("KITCHEN 13'10\"X16'6\"", "tile"),
        ("CHEF'S KITCHEN/WASH AREA", "tile"),
        ("PANTRY", "tile"),
        ("STORE", "tile"),
        ("BOX ROOM", "tile"),
        ("MULTI-PURPOSE HALL", "tile"),
        # Circulation and reception.
        ("LIVING", "stone"),
        ("DINING", "stone"),
        ("FOYER 7'6\"X27'0\"", "stone"),
        ("ASILE", "stone"),
        ("6'4\" WIDE PASSAGE", "stone"),
    ],
)
def test_a_room_is_finished_for_its_purpose(label, expected):
    assert finish_for(label) == expected


@pytest.mark.parametrize("label", ["", "UFT", "5 GE", "SOMETHING ODD"])
def test_an_unknown_room_takes_the_default_finish(label):
    assert finish_for(label) == DEFAULT_FINISH


def test_a_specific_name_beats_a_generic_one():
    # "CHEF'S KITCHEN" contains neither LIVING nor HALL, but a plainer rule
    # ordering could still let a generic match win.
    assert finish_for("CHEF'S KITCHEN") == "tile"


class TestFinishesInTheModel:
    def _floor(self, rooms):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        return FloorPlan(walls=walls, footprint=list(FOOTPRINT), rooms=rooms)

    def _room(self, label, offset=0.0, size=100.0):
        return Room(
            polygon=[
                (offset, offset),
                (offset + size, offset),
                (offset + size, offset + size),
                (offset, offset + size),
            ],
            label=label,
        )

    def test_different_rooms_get_different_finishes(self):
        parts = floors_to_parts(
            [self._floor([self._room("BEDROOM"), self._room("KITCHEN", 150)])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert "timber" in parts
        assert "tile" in parts

    def test_a_wet_room_is_tiled_as_a_wet_room(self):
        parts = floors_to_parts(
            [self._floor([self._room("DRESS/TOILET")])], wall_height_ft=9.0, scale=SCALE
        )

        assert "wet" in parts

    def test_outdoor_areas_get_no_interior_finish(self):
        # A lawn must not also be floored in stone.
        parts = floors_to_parts(
            [self._floor([self._room("LANDSCAPE")])], wall_height_ft=9.0, scale=SCALE
        )

        assert "lawn" in parts
        assert "stone" not in parts

    def test_a_void_gets_no_floor_finish(self):
        # There is no floor there to finish.
        parts = floors_to_parts(
            [self._floor([self._room("DOUBLE HEIGHT")])], wall_height_ft=9.0, scale=SCALE
        )

        assert "stone" not in parts
        assert "timber" not in parts

    def test_the_structural_slab_survives_a_missing_label(self):
        # A finish is a skin on the slab, so an unlabelled room costs a
        # finish rather than its floor.
        parts = floors_to_parts(
            [self._floor([self._room("")])], wall_height_ft=9.0, scale=SCALE
        )

        assert "floor" in parts
