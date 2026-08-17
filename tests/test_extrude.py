import numpy as np
import pytest

from planto3d.extrude import (
    FEET_TO_METRES,
    PARAPET_HEIGHT_FT,
    SLAB_THICKNESS_FT,
    export_glb,
    floors_to_mesh,
    walls_to_mesh,
)
from planto3d.geometry_types import FloorPlan, Wall

# A 10ft wall drawn at 20 px/ft.
SCALE = 20.0
WALL_10FT = Wall(start=(0.0, 0.0), end=(200.0, 0.0), thickness=10.0)


def _height(mesh) -> float:
    return float(mesh.bounds[1][1] - mesh.bounds[0][1])


def test_wall_is_extruded_to_the_requested_height_in_metres():
    mesh = walls_to_mesh([WALL_10FT], wall_height_ft=9.0, scale=SCALE)

    assert _height(mesh) == pytest.approx(9.0 * FEET_TO_METRES, abs=0.01)


def test_wall_length_and_thickness_convert_from_pixels_to_metres():
    mesh = walls_to_mesh([WALL_10FT], wall_height_ft=9.0, scale=SCALE)

    extents = mesh.bounds[1] - mesh.bounds[0]
    horizontal = sorted([extents[0], extents[2]])
    assert horizontal[1] == pytest.approx(10.0 * FEET_TO_METRES, abs=0.05)  # length
    assert horizontal[0] == pytest.approx(0.5 * FEET_TO_METRES, abs=0.05)  # thickness


def test_mesh_is_watertight_and_non_empty():
    mesh = walls_to_mesh([WALL_10FT], wall_height_ft=9.0, scale=SCALE)

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_watertight


def test_perpendicular_walls_keep_their_orientation():
    vertical = Wall(start=(0.0, 0.0), end=(0.0, 200.0), thickness=10.0)

    horizontal_mesh = walls_to_mesh([WALL_10FT], wall_height_ft=9.0, scale=SCALE)
    vertical_mesh = walls_to_mesh([vertical], wall_height_ft=9.0, scale=SCALE)

    h_extents = horizontal_mesh.bounds[1] - horizontal_mesh.bounds[0]
    v_extents = vertical_mesh.bounds[1] - vertical_mesh.bounds[0]
    # The long axis swaps between X and Z; height stays on Y.
    assert h_extents[0] > h_extents[2]
    assert v_extents[2] > v_extents[0]
    assert h_extents[1] == pytest.approx(v_extents[1])


def test_zero_length_walls_are_skipped_rather_than_breaking_the_mesh():
    degenerate = Wall(start=(50.0, 50.0), end=(50.0, 50.0), thickness=10.0)

    mesh = walls_to_mesh([WALL_10FT, degenerate], wall_height_ft=9.0, scale=SCALE)

    assert len(mesh.vertices) > 0


def test_no_walls_raises_rather_than_returning_an_empty_mesh():
    with pytest.raises(ValueError):
        walls_to_mesh([], wall_height_ft=9.0, scale=SCALE)


SQUARE_FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def _floor(footprint=None) -> FloorPlan:
    return FloorPlan(walls=[WALL_10FT], footprint=footprint or list(SQUARE_FOOTPRINT))


class TestFloorStacking:
    def test_each_floor_sits_above_the_one_below(self):
        mesh = floors_to_mesh([_floor(), _floor(), _floor()], wall_height_ft=9.0, scale=SCALE)

        # Three storeys plus the roof slab and its parapet on top.
        expected = (3 * 9.0 + SLAB_THICKNESS_FT + PARAPET_HEIGHT_FT) * FEET_TO_METRES
        assert _height(mesh) == pytest.approx(expected, abs=0.1)

    def test_floors_share_one_horizontal_frame(self):
        # Floors are cropped to a common box upstream, so identical geometry
        # on two floors must land at the same X/Z, not drift apart.
        two = floors_to_mesh([_floor(), _floor()], wall_height_ft=9.0, scale=SCALE)
        one = floors_to_mesh([_floor()], wall_height_ft=9.0, scale=SCALE)

        assert two.bounds[0][0] == pytest.approx(one.bounds[0][0])
        assert two.bounds[1][0] == pytest.approx(one.bounds[1][0])

    def test_a_floor_slab_is_generated_under_each_storey(self):
        with_slab = floors_to_mesh([_floor()], wall_height_ft=9.0, scale=SCALE)
        without = floors_to_mesh([FloorPlan(walls=[WALL_10FT])], wall_height_ft=9.0, scale=SCALE)

        assert len(with_slab.faces) > len(without.faces)

    def test_the_building_is_capped_by_a_roof(self):
        # Without a roof the model tops out at the last storey's walls.
        mesh = floors_to_mesh([_floor()], wall_height_ft=9.0, scale=SCALE)

        assert _height(mesh) > 9.0 * FEET_TO_METRES

    def test_a_floor_without_a_footprint_still_extrudes_its_walls(self):
        mesh = floors_to_mesh([FloorPlan(walls=[WALL_10FT])], wall_height_ft=9.0, scale=SCALE)

        assert len(mesh.faces) > 0

    def test_the_slab_lands_on_the_same_ground_as_its_walls(self):
        # extrude_polygon works in a different frame to the wall boxes. Rotate
        # it the wrong way and the slab mirrors in Z: the model's depth
        # doubles and the roof sits beside the building instead of over it.
        perimeter = [
            Wall(start=SQUARE_FOOTPRINT[i], end=SQUARE_FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        walls_only = walls_to_mesh(perimeter, wall_height_ft=9.0, scale=SCALE)
        with_slab = floors_to_mesh(
            [FloorPlan(walls=perimeter, footprint=list(SQUARE_FOOTPRINT))],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        # Adding a slab must not extend the building's depth.
        wall_depth = walls_only.bounds[1][2] - walls_only.bounds[0][2]
        slab_depth = with_slab.bounds[1][2] - with_slab.bounds[0][2]
        assert slab_depth == pytest.approx(wall_depth, abs=0.2)

    def test_a_slab_sits_at_or_above_its_storey_base(self):
        mesh = floors_to_mesh([_floor()], wall_height_ft=9.0, scale=SCALE)

        assert mesh.bounds[0][1] == pytest.approx(0.0, abs=0.01)

    def test_a_degenerate_footprint_costs_only_the_slab(self):
        # Two points cannot enclose an area; the walls must still survive.
        floor = FloorPlan(walls=[WALL_10FT], footprint=[(0.0, 0.0), (10.0, 0.0)])

        mesh = floors_to_mesh([floor], wall_height_ft=9.0, scale=SCALE)

        assert len(mesh.faces) > 0


def test_export_glb_writes_a_readable_file(tmp_path):
    import trimesh

    mesh = walls_to_mesh([WALL_10FT], wall_height_ft=9.0, scale=SCALE)
    path = tmp_path / "house.glb"

    export_glb(mesh, path)

    assert path.exists() and path.stat().st_size > 0
    assert trimesh.load(str(path)) is not None
