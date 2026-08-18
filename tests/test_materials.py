import pytest
import trimesh

from planto3d.extrude import floors_to_parts
from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.materials import SURFACES, build_scene, export_scene

SCALE = 20.0
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def _floor(openings=None) -> FloorPlan:
    walls = [
        Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0) for i in range(4)
    ]
    return FloorPlan(walls=walls, footprint=list(FOOTPRINT), openings=openings or [])


class TestFloorsToParts:
    def test_groups_geometry_by_material(self):
        parts = floors_to_parts([_floor()], wall_height_ft=9.0, scale=SCALE)

        assert "wall" in parts
        assert "floor" in parts
        assert "roof" in parts

    def test_windows_produce_glass_but_doors_do_not(self):
        # An open doorway reads correctly; a glazed one does not.
        window = floors_to_parts(
            [_floor([Opening(wall_id=0, position=200.0, width=60.0, type="window")])],
            wall_height_ft=9.0,
            scale=SCALE,
        )
        door = floors_to_parts(
            [_floor([Opening(wall_id=0, position=200.0, width=60.0, type="door")])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert "glass" in window
        assert "glass" not in door

    def test_empty_material_groups_are_omitted(self):
        parts = floors_to_parts([_floor()], wall_height_ft=9.0, scale=SCALE)

        assert all(meshes for meshes in parts.values())

    def test_no_floors_is_an_error(self):
        with pytest.raises(ValueError):
            floors_to_parts([], scale=SCALE)


class TestBuildScene:
    def test_each_surface_type_becomes_its_own_geometry(self):
        scene = build_scene(
            [_floor([Opening(wall_id=0, position=200.0, width=60.0, type="window")])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert {"wall", "floor", "roof", "glass"} <= set(scene.geometry)

    def test_materials_are_attached_to_the_geometry(self):
        scene = build_scene([_floor()], wall_height_ft=9.0, scale=SCALE)

        for name, geometry in scene.geometry.items():
            assert geometry.visual.material is not None
            assert geometry.visual.material.name == name

    def test_glass_is_translucent_and_walls_are_not(self):
        # The alpha mode decides whether an opening looks glazed or boarded up.
        assert SURFACES["glass"].opacity < 1.0
        assert SURFACES["glass"].to_material().alphaMode == "BLEND"
        assert SURFACES["wall"].opacity == 1.0
        assert SURFACES["wall"].to_material().alphaMode == "OPAQUE"

    def test_glass_is_far_smoother_than_masonry(self):
        assert SURFACES["glass"].roughness < 0.2
        assert SURFACES["wall"].roughness > 0.8


class TestExport:
    def test_written_file_reloads_with_its_materials(self, tmp_path):
        scene = build_scene(
            [_floor([Opening(wall_id=0, position=200.0, width=60.0, type="window")])],
            wall_height_ft=9.0,
            scale=SCALE,
        )
        path = tmp_path / "house.glb"

        export_scene(scene, path)

        assert path.stat().st_size > 0
        reloaded = trimesh.load(str(path))
        assert len(reloaded.geometry) == len(scene.geometry)
        materials = {
            geometry.visual.material.name for geometry in reloaded.geometry.values()
        }
        assert "glass" in materials

    def test_the_model_keeps_its_real_world_size(self, tmp_path):
        scene = build_scene([_floor()], wall_height_ft=9.0, scale=SCALE)
        path = tmp_path / "house.glb"
        export_scene(scene, path)

        reloaded = trimesh.load(str(path), force="mesh")
        bounds = reloaded.bounds[1] - reloaded.bounds[0]

        # 400px at 20px/ft is 20ft, which is 6.096m.
        assert bounds[0] == pytest.approx(6.096, abs=0.3)
