"""The Blender path, tested where it can be tested without Blender.

bpy is a 659 MB optional dependency. The suite must pass on a machine
that does not have it, so everything here either avoids importing it or
skips explicitly -- never silently.
"""

import logging

import numpy as np
import pytest
from PIL import Image

from planto3d import blender_render, materials


def test_the_view_set_matches_the_rasterizer():
    # The two renderers must answer to the same view names, or comparing
    # them frame for frame means renaming files by hand.
    from planto3d.preview import VIEWS

    assert set(blender_render.STANDARD_VIEWS) == set(VIEWS)
    for name, angles in VIEWS.items():
        assert blender_render.STANDARD_VIEWS[name] == angles


def test_standard_views_cover_the_named_angles_and_aerial_is_oblique():
    # STANDARD_VIEWS == preview.VIEWS by construction (it is that dict,
    # imported under a second name) -- asserting so would just be
    # asserting X == X. What is actually worth guarding is that
    # preview.py itself still defines the six named views this renderer's
    # framing assumes, and that "aerial" is genuinely oblique rather than
    # top-down or level: _add_camera's dominant-axis test only covers the
    # five axis-aligned views, so aerial's obliqueness is what makes it a
    # meaningful third viewpoint rather than a duplicate of "top".
    from planto3d.preview import VIEWS

    assert set(VIEWS) == {"top", "front", "back", "left", "right", "aerial"}
    _, aerial_elevation = VIEWS["aerial"]
    assert 0.0 < aerial_elevation < 90.0


def test_render_views_produces_one_path_per_standard_view(tmp_path, monkeypatch):
    # render_views loops render_view once per name; this checks the loop
    # and the naming/prefix contract without paying for six real renders.
    calls = []

    def fake_render_view(model_path, output_path, azimuth, elevation, **kwargs):
        calls.append((output_path, azimuth, elevation))
        output_path.write_bytes(b"fake")
        return output_path

    monkeypatch.setattr(blender_render, "render_view", fake_render_view)

    rendered = blender_render.render_views(tmp_path / "house.glb", tmp_path, prefix="blender")

    assert set(rendered) == set(blender_render.STANDARD_VIEWS)
    for name, path in rendered.items():
        assert path == tmp_path / f"blender-{name}.png"
        assert path.is_file()
    assert len(calls) == len(blender_render.STANDARD_VIEWS)


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


def _tiny_model(out_dir):
    """A minimal one-room glb, for tests that just need something to render."""
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
    return export_scene(build_scene([plan], scale=20.0), out_dir / "house.glb")


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_a_model_renders_to_a_real_image(tmp_path):
    model = _tiny_model(tmp_path)

    out = blender_render.render_view(
        model, tmp_path / "aerial.png", azimuth=38.0, elevation=45.0,
        resolution=(320, 240), samples=16,
    )
    assert out.is_file()
    # A blank frame is also a file. Anything real is larger than this.
    assert out.stat().st_size > 5_000


def test_lighting_is_optional_and_defaults_to_the_shared_preset():
    # Every existing caller passes no lighting and must keep working.
    import inspect

    from planto3d.style import Lighting

    signature = inspect.signature(blender_render.render_view)
    parameter = signature.parameters["lighting"]
    assert parameter.default is None
    # Keyword-optional and last, so positional callers are unaffected.
    assert list(signature.parameters)[-1] == "lighting"


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_the_time_of_day_changes_the_image():
    # style.py already maps the user's choice onto a Lighting preset and
    # preview.py honours it. If this renderer ignores it, the two outputs
    # disagree about what hour it is -- the same defect the photoreal
    # prompt had before it was given the design.
    import tempfile
    from pathlib import Path

    from planto3d.style import LIGHTING_PRESETS

    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir)
        model = _tiny_model(out)
        rendered = {}
        for name in ("midday", "dusk"):
            path = blender_render.render_view(
                model, out / f"{name}.png", azimuth=38.0, elevation=45.0,
                resolution=(160, 120), samples=16,
                lighting=LIGHTING_PRESETS[name],
            )
            rendered[name] = path.read_bytes()
        assert rendered["midday"] != rendered["dusk"]


def _silhouette(path, size: int = 64) -> np.ndarray:
    """A cropped, size-normalised black/white mask of the rendered content.

    Thresholding against the most common pixel value used to be enough
    when the world was a flat grey void. Task 5 gave the renderer a sky
    and a ground plane, which are now the two dominant flat regions in
    any low-elevation frame (measured on this fixture: together over 60%
    of pixels) -- masking out only one of them leaves the other counted
    as "content" and swamps the building's own silhouette. Masking out
    the two most common values instead excludes sky and ground and keeps
    the building plus its contact shadow, which is what this test means
    by content.
    """
    img = np.array(Image.open(path).convert("L"))
    values, counts = np.unique(img, return_counts=True)
    backgrounds = values[np.argsort(-counts)[:2]]
    mask = ~np.isin(img, backgrounds)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=bool)
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    resized = Image.fromarray(cropped.astype(np.uint8) * 255).resize((size, size))
    return np.array(resized) > 127


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_left_and_right_views_produce_real_differing_content(tmp_path):
    # NOT a mirroring regression guard -- see
    # test_camera_positions_match_the_named_views for that. A bug that
    # swaps which camera position is called "left" and which is called
    # "right" exchanges the two views wholesale (buggy_left renders
    # exactly what correct_right would have), and any difference(left,
    # right) metric is symmetric under swapping its own arguments, so it
    # cannot see a pure swap: it was tried, and proven blind to it, in an
    # earlier round -- see the fix report. What this test does check is
    # that the two views are live renders with real, non-trivial content
    # rather than e.g. both being blank or both showing the same frame
    # twice by accident.
    #
    # The two blocks sit far apart along plan X, which is the axis the
    # left/right cameras look straight down. Orthographically the two
    # opposite-facing silhouettes would be identical regardless of which
    # camera is "correct" -- occlusion along the view axis is invisible
    # to a silhouette test by construction. What actually discriminates
    # here is perspective: with the blocks compact in Y/Z and spread out
    # in X, the block nearest each camera projects larger than the block
    # farther away, and which block is near flips between the two
    # cameras. A wing offset sideways (across the frame) was tried first
    # and reads identically from both sides -- see the fix report.
    from planto3d.geometry_types import FloorPlan, Room, Wall
    from planto3d.materials import build_scene, export_scene
    from planto3d.preview import VIEWS

    def walls_for(poly):
        return [
            Wall(start=poly[i], end=poly[(i + 1) % len(poly)], thickness=6.0)
            for i in range(len(poly))
        ]

    near_block = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    far_block = [(500.0, 0.0), (600.0, 0.0), (600.0, 100.0), (500.0, 100.0)]
    plan = FloorPlan(
        walls=walls_for(near_block) + walls_for(far_block),
        rooms=[
            Room(polygon=near_block, label="BEDROOM"),
            Room(polygon=far_block, label="BEDROOM"),
        ],
        openings=[],
        footprint=near_block,
    )
    model = export_scene(build_scene([plan], scale=20.0), tmp_path / "dumbbell.glb")

    silhouettes = {}
    for name in ("left", "right"):
        azimuth, elevation = VIEWS[name]
        out = blender_render.render_view(
            model, tmp_path / f"{name}.png", azimuth=azimuth, elevation=elevation,
            resolution=(320, 240), samples=16,
        )
        silhouettes[name] = _silhouette(out)

    left, right = silhouettes["left"], silhouettes["right"]
    mirrored_right = right[:, ::-1]

    # Measured on this fixture: left-vs-right and left-vs-mirrored(right)
    # both differ by about 8% of pixels. 2% is comfortably below that
    # while well above the noise floor of a genuinely identical or
    # genuinely mirrored pair, which is what a still-swapped camera or a
    # symmetric fixture would produce.
    assert (left != right).mean() > 0.02, "left and right rendered identically"
    assert (left != mirrored_right).mean() > 0.02, (
        "left is a mirror image of right -- the camera-side bug is back"
    )


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_camera_positions_match_the_named_views():
    # This is the actual regression guard for the left/right camera-sign
    # bug. A left-versus-right content comparison is symmetric under
    # swapping which camera is called "left" and which is called "right"
    # -- that swap is exactly the bug that shipped, so a relative
    # comparison is mathematically blind to it (proven directly: a
    # reviewer monkeypatched the sign back and the difference metrics in
    # the test above came back bit-identical). Asserting each named
    # view's camera position against what the name is supposed to mean
    # does not have that blind spot, needs no render at all -- so it is
    # fast and free of path-tracer noise -- and fails on the very first
    # assertion the moment the sign is wrong.
    #
    # Expected dominant axis and sign in glTF space, using the remap
    # documented in _add_camera (X_gltf = X_blender, Y_gltf = Z_blender,
    # Z_gltf = -Y_blender). Measured directly against the fixed code and
    # independently confirmed to match what preview.py means by the same
    # names (its own yaw convention, inverted).
    #
    #   top   -> +Y      front -> +Z      back -> -Z
    #   right -> -X      left  -> +X
    expected_dominant = {
        "top": ("Y", 1),
        "front": ("Z", 1),
        "back": ("Z", -1),
        "left": ("X", 1),
        "right": ("X", -1),
    }

    # aerial (azimuth=38, elevation=45) is oblique, so its single dominant
    # axis is simply "up" -- checking that only proves the camera is above
    # the model, and would not catch a flipped sin(az) term the way the
    # five axis-aligned views above do. Its full three-axis sign pattern
    # is checked instead, which does catch that: measured directly on the
    # fixed code as gltf = (X -1.05, Y +1.84, Z +1.51) -> signs (-, +, +).
    aerial_signs = {"X": -1, "Y": 1, "Z": 1}

    import bpy

    from planto3d.preview import VIEWS

    def gltf_position(name):
        azimuth, elevation = VIEWS[name]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        blender_render._add_camera(
            bpy.context.scene, (0.0, 0.0, 0.0), radius=1.0,
            azimuth=azimuth, elevation=elevation,
        )
        loc = bpy.context.scene.camera.location
        return {"X": loc.x, "Y": loc.z, "Z": -loc.y}

    for name, (axis, sign) in expected_dominant.items():
        gltf = gltf_position(name)
        value = gltf[axis]
        assert value * sign > 0.5, (
            f"{name}: expected the dominant axis to be "
            f"{'+' if sign > 0 else '-'}{axis} in glTF space, got "
            f"gltf=({gltf['X']:.2f}, {gltf['Y']:.2f}, {gltf['Z']:.2f})"
        )

    gltf = gltf_position("aerial")
    for axis, sign in aerial_signs.items():
        assert gltf[axis] * sign > 0.1, (
            f"aerial: expected {'+' if sign > 0 else '-'}{axis} in glTF "
            f"space, got gltf=({gltf['X']:.2f}, {gltf['Y']:.2f}, "
            f"{gltf['Z']:.2f})"
        )


def test_every_surface_materials_py_can_produce_has_a_shader():
    # planto3d/materials.py -- not one demo plan's glb -- is the source of
    # truth for what surface names can occur. Fix round 1: the previous
    # version of this test hardcoded 13 names taken from importing
    # data/bridge/11001.gif's glb and reading its materials back, which
    # only proves the table covers what that one plan happens to contain.
    # materials.SURFACES has 24 entries; a plan with a dome, a water
    # feature or a pitched roof exercises the other eleven, and any of
    # them missing here falls back to DEFAULT_SHADER -- plaster -- with
    # no error anywhere.
    assert set(materials.SURFACES) <= set(blender_render.SURFACE_SHADERS)


def test_no_surface_falls_back_to_default():
    # Same guard as above, phrased the way the drift actually bites:
    # each name individually, so a future addition to materials.py that
    # is not yet reflected here fails on that one name rather than as
    # a set-difference someone has to decode.
    for name in materials.SURFACES:
        assert name in blender_render.SURFACE_SHADERS, (
            f"{name!r} has no shader entry and would render as plaster"
        )


def test_shader_settings_are_derived_from_materials_surfaces():
    # SURFACE_SHADERS must be a derivation of materials.SURFACES, not a
    # second, hand-copied table -- a hand-written copy already drifted
    # once (round 1: 13 entries vs materials.py's 24). Checking every
    # field against the source, for every surface, is what makes a
    # second drift structurally impossible rather than just unlikely.
    for name, surface in materials.SURFACES.items():
        shader = blender_render.SURFACE_SHADERS[name]
        assert shader["metallic"] == surface.metallic
        assert shader["roughness"] == surface.roughness
        # materials.py expresses translucency as glTF opacity; Cycles
        # expresses it as transmission. Opaque (opacity 1.0) must derive
        # to zero transmission, and the two near-transparent surfaces
        # (glass at 0.35 opacity, water at 0.72) must derive to visibly
        # transmissive settings rather than a second hand-tuned number.
        assert shader["transmission"] == pytest.approx(1.0 - surface.opacity)


def test_glass_is_actually_transmissive():
    # The one surface where a wrong material is unmistakable.
    glass = blender_render.SURFACE_SHADERS["glass"]
    assert glass["transmission"] > 0.5
    assert glass["roughness"] < 0.2


def test_water_is_actually_transmissive():
    # Round 1's worst case: water was missing from the hand-written table
    # entirely, so it would have rendered as opaque plaster with metallic
    # 0.0 and roughness 0.8 -- flat and dull, nothing like a pool. It was
    # not in data/bridge/11001.gif, which is exactly how that slipped
    # past a render-based check. materials.py gives it roughness 0.04
    # (near-mirror) and opacity 0.72 (partly see-through, less so than
    # glass's 0.35), deriving to transmission 0.28: some, and glossy --
    # a materially different, correct surface, not the fallback.
    water = blender_render.SURFACE_SHADERS["water"]
    assert water["transmission"] > 0.2
    assert water["roughness"] < blender_render.DEFAULT_SHADER["roughness"]
    assert water["metallic"] == blender_render.DEFAULT_SHADER["metallic"]


def test_railings_and_frames_are_more_metallic_than_masonry():
    # Not an absolute ">0.5" threshold: materials.py gives frame a
    # metallic of 0.4, which a hand-picked ">0.5" bar (round 1's table
    # used a hand-tuned 0.8) would fail even though frame is correctly
    # more metallic than any masonry surface. The comparison that
    # actually matters, and that survives the source data changing, is
    # relative: railings and frames read as metal, masonry does not.
    masonry_metallic = max(
        blender_render.SURFACE_SHADERS[name]["metallic"]
        for name in ("wall", "stone", "plinth", "coping")
    )
    for surface in ("railing", "frame"):
        assert blender_render.SURFACE_SHADERS[surface]["metallic"] > masonry_metallic


def test_masonry_is_rough_and_not_metal():
    for surface in ("wall", "stone", "plinth", "coping"):
        shader = blender_render.SURFACE_SHADERS[surface]
        assert shader["roughness"] > 0.5
        assert shader["metallic"] == 0.0


class _FakeSocket:
    """Stands in for a bpy NodeSocket: nothing but a settable value."""

    def __init__(self):
        self.default_value = None


class _FakePrincipled:
    """Stands in for a bpy Principled BSDF node's ``.inputs``.

    Keyed by the same socket names _apply_materials indexes by, so this
    exercises the real SOCKET_NAMES lookup -- including the
    ``KeyError``-on-miss behaviour -- without needing bpy installed.
    """

    def __init__(self):
        self.inputs = {name: _FakeSocket() for name in blender_render.SOCKET_NAMES.values()}


class _FakeNodeTree:
    def __init__(self, principled):
        self.nodes = {"Principled BSDF": principled}


class _FakeMaterial:
    def __init__(self, name):
        self.name = name
        self.use_nodes = False
        self.principled = _FakePrincipled()
        self.node_tree = _FakeNodeTree(self.principled)


class _FakeSlot:
    def __init__(self, material):
        self.material = material


class _FakeObject:
    def __init__(self, *material_names):
        self.material_slots = [_FakeSlot(_FakeMaterial(name)) for name in material_names]


def test_apply_materials_recognises_every_materials_py_surface_without_bpy():
    # _apply_materials only ever touches duck-typed attributes
    # (material_slots / material / name / use_nodes / node_tree / inputs),
    # so its recognition logic is testable without bpy at all -- this is
    # a stronger guarantee than the dict-membership tests above because
    # it runs the actual lookup-and-assign code path, including the
    # SOCKET_NAMES indexing.
    objects = [_FakeObject(name) for name in materials.SURFACES]
    recognised = blender_render._apply_materials(objects)
    assert recognised == len(materials.SURFACES)
    for obj in objects:
        material = obj.material_slots[0].material
        surface = materials.SURFACES[material.name]
        principled = material.principled
        assert principled.inputs["Metallic"].default_value == surface.metallic
        assert principled.inputs["Roughness"].default_value == surface.roughness
        assert principled.inputs["Transmission Weight"].default_value == pytest.approx(
            1.0 - surface.opacity
        )


def test_apply_materials_warns_on_an_unrecognised_surface(caplog):
    # DEFAULT_SHADER is meant to be a genuine anomaly now, not the
    # routine path an unmapped name quietly takes -- so a miss must be
    # loud. This is the regression guard for exactly the failure mode
    # round 1 hit: a name with no entry, discovered only by luck if
    # anyone happens to look at a render.
    obj = _FakeObject("wall", "some_future_surface_nobody_added_yet")
    with caplog.at_level(logging.WARNING, logger="planto3d.blender_render"):
        recognised = blender_render._apply_materials([obj])
    assert recognised == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "some_future_surface_nobody_added_yet" in warnings[0].message
