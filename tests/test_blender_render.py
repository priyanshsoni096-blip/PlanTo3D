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
    #
    # Fix round 3: this used to assert STANDARD_VIEWS == VIEWS, which is
    # X == X -- STANDARD_VIEWS *is* VIEWS, imported under a second name --
    # so it could not fail. What is worth guarding is the aliasing
    # itself: the moment someone replaces the import with a copied dict
    # to "decouple" the two renderers, the two view sets can drift and
    # nothing else in the suite would notice. Identity is the property
    # that makes drift impossible rather than merely unlikely.
    from planto3d.preview import VIEWS

    assert blender_render.STANDARD_VIEWS is VIEWS


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
    # The angles come from VIEWS rather than being typed here: they used
    # to read azimuth=38, while VIEWS["aerial"] is 35, so this was not
    # rendering the view its filename claimed.
    from planto3d.preview import VIEWS

    model = _tiny_model(tmp_path)
    azimuth, elevation = VIEWS["aerial"]

    out = blender_render.render_view(
        model, tmp_path / "aerial.png", azimuth=azimuth, elevation=elevation,
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
    #
    # Fix round 3: this used to compare path.read_bytes(). That assertion
    # could not fail. Blender stamps the wall-clock render time into the
    # PNG's tEXt chunks -- docs/AUDIT.md's own determinism section
    # measures exactly this -- so *any* two renders differ as files,
    # including two renders of the same scene with the same preset. A
    # reviewer passed LIGHTING_PRESETS["midday"] for both and got
    # "bytes differ: True" alongside "pixels max abs diff: 0".
    #
    # So: decoded pixels, and a direction that is specific to the hour
    # rather than merely non-zero. Measured on this fixture at the aerial
    # view, 320x240, 16 samples -- mean abs pixel difference 31.9, and
    # the warm cast (mean R minus mean B) is +20.2 at midday against
    # +76.3 at dusk, a gap of 56. Both thresholds below sit far under
    # what was measured and far above the zero that any hour-ignoring
    # renderer produces.
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from planto3d.preview import VIEWS
    from planto3d.style import LIGHTING_PRESETS

    azimuth, elevation = VIEWS["aerial"]
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir)
        model = _tiny_model(out)
        rendered = {}
        for name in ("midday", "dusk"):
            path = blender_render.render_view(
                model, out / f"{name}.png", azimuth=azimuth, elevation=elevation,
                resolution=(320, 240), samples=16,
                lighting=LIGHTING_PRESETS[name],
            )
            rendered[name] = np.array(Image.open(path).convert("RGB"), dtype=float)

    midday, dusk = rendered["midday"], rendered["dusk"]
    assert np.abs(midday - dusk).mean() > 5.0, (
        "midday and dusk decoded to near-identical pixels -- the hour is "
        "being ignored somewhere between the argument and the render"
    )

    def warmth(image):
        red, _, blue = image.mean(axis=(0, 1))
        return red - blue

    # Direction, not just magnitude: dusk's sun is (255,176,122) against
    # midday's neutral, so dusk must come out the warmer of the two. A
    # difference metric alone would accept two frames that differ for any
    # reason at all, including noise; this only passes if the hour that
    # is supposed to be warmer actually is.
    assert warmth(dusk) - warmth(midday) > 20.0, (
        f"dusk is not warmer than midday: R-B was {warmth(midday):+.1f} at "
        f"midday and {warmth(dusk):+.1f} at dusk"
    )


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
    # Fix round 3: this used to call _add_camera, which needs bpy for
    # camera_add, so the sole guard for a bug that actually shipped was
    # skipped on every machine without the 659 MB optional extra -- which
    # is most machines and every CI runner. The positioning is pure
    # trigonometry, so it now lives in _camera_position and this test
    # points at that instead. Same arithmetic, same expected values, no
    # skip.
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

    # aerial (azimuth=35, elevation=45) is oblique, so its single dominant
    # axis is simply "up" -- checking that only proves the camera is above
    # the model, and would not catch a flipped sin(az) term the way the
    # five axis-aligned views above do. Its full three-axis sign pattern
    # is checked instead, which does catch that: measured directly on the
    # fixed code as gltf = (X -1.05, Y +1.84, Z +1.51) -> signs (-, +, +).
    aerial_signs = {"X": -1, "Y": 1, "Z": 1}

    from planto3d.preview import VIEWS

    def gltf_position(name):
        azimuth, elevation = VIEWS[name]
        x, y, z = blender_render._camera_position(
            (0.0, 0.0, 0.0), radius=1.0, azimuth=azimuth, elevation=elevation,
        )
        return {"X": x, "Y": z, "Z": -y}

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


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_add_camera_places_the_camera_where_camera_position_says():
    # _camera_position was split out of _add_camera so the regression
    # guard above can run without bpy. That split introduces a seam: the
    # pure function could stay right while _add_camera grew its own,
    # different arithmetic, and the guard would not notice. This is the
    # stitch across that seam -- the one thing that needs bpy, and the
    # only thing this test checks.
    import bpy

    centre, radius, azimuth, elevation = (1.0, -2.0, 0.5), 3.0, 35.0, 45.0
    bpy.ops.wm.read_factory_settings(use_empty=True)
    blender_render._add_camera(
        bpy.context.scene, centre, radius, azimuth=azimuth, elevation=elevation,
    )
    placed = tuple(bpy.context.scene.camera.location)
    expected = blender_render._camera_position(centre, radius, azimuth, elevation)
    assert placed == pytest.approx(expected, abs=1e-5)


def _bare_sky(preset: str, elevation: float, resolution=(160, 120), samples=8):
    """Render the world alone -- no model, no ground -- and return the pixels.

    The sky is the only thing under test here, so nothing else is put in
    front of it. Uses the same camera _add_camera builds, because the
    ramp positions are chosen against the field of view that camera has.
    """
    import math
    import tempfile
    from pathlib import Path

    import bpy

    from planto3d.style import LIGHTING_PRESETS

    lighting = LIGHTING_PRESETS[preset]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = blender_render.ENGINE
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = resolution
    blender_render._add_camera(scene, (0.0, 0.0, 0.0), 1.0, 0.0, elevation)
    blender_render._add_world(scene, lighting)
    scene.view_settings.exposure = math.log2(max(lighting.exposure, 1e-3))
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "sky.png"
        scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        return np.array(Image.open(out).convert("RGB"), dtype=float)


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_the_sky_is_graduated_and_the_right_way_up():
    # Nothing on this branch tested the sky at all: a reviewer reverted
    # _add_world to the flat lighting.sky colour it had before the last
    # two commits -- the exact bug those commits exist to fix -- and the
    # whole suite still passed. This is the test that would have caught
    # it, and it catches the other two ways the sky has actually been
    # wrong here: the Incoming.Z sign inverted (blue below, warm above),
    # and SKY_TOP_POSITION set above the ~0.26 these cameras can reach,
    # which leaves the top of frame warm because the top of the ramp is
    # never sampled.
    #
    # Measured on the dusk preset at elevation 0, bare world, 160x120,
    # 8 samples: the top row is (9,32,61) -- B-R = +52 -- and a quarter
    # of the way down it is (129,98,80) -- B-R = -49, warm, near the
    # horizon glow. A flat sky gives the same sign at both heights; an
    # inverted ramp swaps them; SKY_TOP_POSITION=0.35 gives (99,78,73)
    # at the top, B-R = -26, and fails the first assertion.
    image = _bare_sky("dusk", elevation=0.0)
    rows = image.mean(axis=1)

    def blueness(row):
        return row[2] - row[0]

    top, quarter_down = rows[0], rows[len(rows) // 4]
    assert blueness(top) > 15.0, (
        f"the top of the dusk sky is not blue: {tuple(top.round(0))}"
    )
    assert blueness(quarter_down) < -15.0, (
        "the sky is not graduated -- near the horizon it should be warm, "
        f"got {tuple(quarter_down.round(0))}"
    )


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_the_sky_follows_the_hour():
    # _add_world reads sky_top/sky_bottom/sky_glow off the preset. If it
    # stopped -- the mutation the reviewer ran -- every hour would render
    # the same sky. Midday's zenith is (96,140,190) and dusk's is a much
    # darker blue, so the same row of the same frame must differ
    # substantially between the two. Measured at elevation 0, 160x120,
    # 8 samples: midday's top row is (47,83,112) against dusk's (9,32,61).
    midday = _bare_sky("midday", elevation=0.0).mean(axis=1)[0]
    dusk = _bare_sky("dusk", elevation=0.0).mean(axis=1)[0]
    assert np.abs(midday - dusk).mean() > 15.0, (
        f"midday and dusk skies are the same: {tuple(midday.round(0))} vs "
        f"{tuple(dusk.round(0))}"
    )


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_the_model_stands_on_a_ground_plane(tmp_path):
    # Deleting _add_ground entirely left the suite green. Without it the
    # model floats in a void with nothing to catch its contact shadow,
    # which is most of what makes a render read as photographed. Checked
    # through render_view rather than by calling _add_ground directly, so
    # that dropping the *call* is caught as well as dropping the function.
    import bpy

    model = _tiny_model(tmp_path)
    blender_render.render_view(
        model, tmp_path / "front.png", azimuth=0.0, elevation=0.0,
        resolution=(64, 48), samples=1,
    )

    planes = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH"
        and any(
            slot.material is not None and slot.material.name.startswith("ground_plane")
            for slot in obj.material_slots
        )
    ]
    assert len(planes) == 1, "no ground plane in the rendered scene"
    plane = planes[0]

    # It must reach past the building, or the horizon shows inside the
    # frame and the model reads as standing on a table.
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o is not plane]
    _, radius = blender_render._bounds(meshes)
    assert max(plane.dimensions) == pytest.approx(
        radius * blender_render.GROUND_EXTENT, rel=0.01
    )
    # And it must meet the model rather than cutting through it or
    # hovering below it.
    assert plane.location.z == pytest.approx(blender_render._lowest_z(), abs=1e-4)


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_the_light_rig_follows_the_hour():
    # Making _add_lighting ignore the chosen hour left the suite green
    # too. The sun's colour and strength are the whole of what "what time
    # is it" means to Cycles here, so they are what this reads back --
    # from the scene, after the rig is built, with no render needed.
    import bpy

    from planto3d.style import LIGHTING_PRESETS

    def sun_for(preset):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        lighting = LIGHTING_PRESETS[preset]
        blender_render._add_lighting(
            bpy.context.scene, (0.0, 0.0, 0.0), 1.0, lighting
        )
        suns = [
            obj.data for obj in bpy.data.objects
            if obj.type == "LIGHT" and obj.data.type == "SUN"
        ]
        assert len(suns) == 1, "expected exactly one key sun"
        return suns[0], lighting

    for preset in ("midday", "golden hour", "dusk"):
        sun, lighting = sun_for(preset)
        # Straight from style.py, not a second table of hours.
        assert tuple(sun.color) == pytest.approx(
            blender_render._to_linear(lighting.sun)[:3], abs=1e-6
        )
        assert sun.energy == pytest.approx(
            lighting.key_strength * blender_render.KEY_ENERGY
        )

    # And the presets must actually reach different rigs, not merely be
    # read: dusk's sun is (255,176,122) against midday's (255,237,209).
    midday_sun, _ = sun_for("midday")
    midday_colour = tuple(midday_sun.color)
    dusk_sun, _ = sun_for("dusk")
    assert tuple(dusk_sun.color) != pytest.approx(midday_colour, abs=1e-6)


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
