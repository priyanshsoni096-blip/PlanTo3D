import numpy as np
import pytest

from planto3d.classes import BACKGROUND, DOOR, WALL, WINDOW
from planto3d.extract import extract_openings
from planto3d.extrude import (
    FEET_TO_METRES,
    OPENING_HEAD_FT,
    SILL_HEIGHT_FT,
    walls_to_mesh,
)
from planto3d.geometry_types import Opening, Wall

SCALE = 20.0
WALL_20FT = Wall(start=(0.0, 100.0), end=(400.0, 100.0), thickness=10.0)


class TestExtractOpenings:
    def _mask(self, size=200):
        mask = np.full((size, size), BACKGROUND, dtype=np.int64)
        mask[98:103, 0:200] = WALL  # a horizontal wall
        return mask

    def _wall(self):
        return Wall(start=(0.0, 100.0), end=(200.0, 100.0), thickness=5.0)

    def test_binds_a_door_to_its_wall_at_the_right_distance_along(self):
        mask = self._mask()
        mask[98:103, 80:120] = DOOR

        openings = extract_openings(mask, [self._wall()])

        assert len(openings) == 1
        assert openings[0].type == "door"
        assert openings[0].wall_id == 0
        assert openings[0].position == pytest.approx(100, abs=5)
        assert openings[0].width == pytest.approx(40, abs=5)

    def test_distinguishes_windows_from_doors(self):
        mask = self._mask()
        mask[98:103, 20:50] = DOOR
        mask[98:103, 140:170] = WINDOW

        openings = extract_openings(mask, [self._wall()])

        assert sorted(o.type for o in openings) == ["door", "window"]

    def test_binds_each_opening_to_the_nearest_wall(self):
        mask = np.full((300, 300), BACKGROUND, dtype=np.int64)
        mask[48:53, 0:300] = WALL
        mask[248:253, 0:300] = WALL
        mask[248:253, 100:140] = DOOR  # sits on the lower wall

        walls = [
            Wall(start=(0.0, 50.0), end=(300.0, 50.0), thickness=5.0),
            Wall(start=(0.0, 250.0), end=(300.0, 250.0), thickness=5.0),
        ]
        openings = extract_openings(mask, walls)

        assert len(openings) == 1
        assert openings[0].wall_id == 1

    def test_an_opening_far_from_every_wall_is_dropped(self):
        # Binding a stray blob to a distant wall would cut a hole through
        # solid geometry somewhere unrelated.
        mask = self._mask()
        mask[10:20, 10:20] = DOOR

        assert extract_openings(mask, [self._wall()]) == []

    def test_speckles_below_the_area_floor_are_ignored(self):
        mask = self._mask()
        mask[100, 50] = DOOR

        assert extract_openings(mask, [self._wall()]) == []

    def test_no_walls_means_no_openings(self):
        mask = self._mask()
        mask[98:103, 80:120] = DOOR

        assert extract_openings(mask, []) == []


class TestWallsWithOpenings:
    def _height(self, mesh):
        return float(mesh.bounds[1][1] - mesh.bounds[0][1])

    def test_an_opening_removes_material_from_the_wall(self):
        solid = walls_to_mesh([WALL_20FT], wall_height_ft=9.0, scale=SCALE)
        pierced = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=0, position=200.0, width=60.0, type="door")],
        )

        assert pierced.volume < solid.volume

    def test_a_door_reaches_the_floor_but_a_window_does_not(self):
        # A window keeps a sill under it; a door does not.
        door = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=0, position=200.0, width=60.0, type="door")],
        )
        window = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=0, position=200.0, width=60.0, type="window")],
        )

        assert window.volume > door.volume

    def test_the_wall_keeps_its_full_height_above_an_opening(self):
        # The lintel over the opening must still reach the ceiling.
        pierced = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=0, position=200.0, width=60.0, type="door")],
        )

        assert self._height(pierced) == pytest.approx(9.0 * FEET_TO_METRES, abs=0.01)

    def test_several_openings_in_one_wall_all_register(self):
        one = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=0, position=100.0, width=40.0, type="door")],
        )
        three = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[
                Opening(wall_id=0, position=100.0, width=40.0, type="door"),
                Opening(wall_id=0, position=200.0, width=40.0, type="door"),
                Opening(wall_id=0, position=300.0, width=40.0, type="door"),
            ],
        )

        assert three.volume < one.volume

    def test_overlapping_openings_do_not_produce_inverted_geometry(self):
        mesh = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[
                Opening(wall_id=0, position=200.0, width=80.0, type="door"),
                Opening(wall_id=0, position=210.0, width=80.0, type="door"),
            ],
        )

        assert mesh.volume > 0
        assert len(mesh.faces) > 0

    def test_an_opening_wider_than_its_wall_leaves_a_lintel(self):
        mesh = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=0, position=200.0, width=10_000.0, type="door")],
        )

        # Only the lintel survives, but it still spans from the opening head
        # to the ceiling -- the wall must not vanish and leave a floating slab.
        assert mesh.volume > 0
        assert mesh.bounds[1][1] == pytest.approx(9.0 * FEET_TO_METRES, abs=0.01)
        assert self._height(mesh) == pytest.approx(
            (9.0 - OPENING_HEAD_FT) * FEET_TO_METRES, abs=0.01
        )

    def test_an_opening_naming_a_wall_that_does_not_exist_is_ignored(self):
        solid = walls_to_mesh([WALL_20FT], wall_height_ft=9.0, scale=SCALE)
        stray = walls_to_mesh(
            [WALL_20FT],
            wall_height_ft=9.0,
            scale=SCALE,
            openings=[Opening(wall_id=99, position=100.0, width=40.0, type="door")],
        )

        assert stray.volume == pytest.approx(solid.volume)

    def test_openings_are_optional(self):
        mesh = walls_to_mesh([WALL_20FT], wall_height_ft=9.0, scale=SCALE, openings=None)

        assert mesh.volume > 0
