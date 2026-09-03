"""The Blender path, tested where it can be tested without Blender.

bpy is a 659 MB optional dependency. The suite must pass on a machine
that does not have it, so everything here either avoids importing it or
skips explicitly -- never silently.
"""

import pytest

from planto3d import blender_render


def test_availability_is_reported_not_guessed():
    # Callers need to know whether the Blender path can run at all, and
    # a bare ImportError deep inside a render is a poor way to find out.
    assert isinstance(blender_render.available(), bool)


def test_the_module_imports_without_bpy_installed():
    # Importing planto3d.blender_render must never require bpy -- the
    # pipeline imports planto3d modules freely and most machines running
    # it will not have Blender.
    assert hasattr(blender_render, "render_view")
    assert hasattr(blender_render, "DEFAULT_SAMPLES")


def test_the_sample_default_is_the_measured_one():
    # 32 samples renders 640x480 in 7.7s; 64 costs 9.5s. Quality is nearly
    # free here, so the default sits above the minimum rather than at it.
    assert blender_render.DEFAULT_SAMPLES >= 32


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_a_model_renders_to_a_real_image(tmp_path):
    from planto3d.geometry_types import FloorPlan, Room, Wall
    from planto3d.materials import build_scene, export_scene

    outline = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    walls = [
        Wall(start=outline[i], end=outline[(i + 1) % 4], thickness=6.0)
        for i in range(4)
    ]
    plan = FloorPlan(
        walls=walls,
        rooms=[Room(polygon=outline, label="BEDROOM")],
        openings=[],
        footprint=outline,
    )
    model = export_scene(build_scene([plan], scale=20.0), tmp_path / "house.glb")

    out = blender_render.render_view(
        model, tmp_path / "aerial.png", azimuth=38.0, elevation=45.0,
        resolution=(320, 240), samples=16,
    )
    assert out.is_file()
    # A blank frame is also a file. Anything real is larger than this.
    assert out.stat().st_size > 5_000
