"""The Blender path, tested where it can be tested without Blender.

bpy is a 659 MB optional dependency. The suite must pass on a machine
that does not have it, so everything here either avoids importing it or
skips explicitly -- never silently.
"""

import numpy as np
import pytest
from PIL import Image

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


def _silhouette(path, size: int = 64) -> np.ndarray:
    """A cropped, size-normalised black/white mask of the rendered content.

    Thresholding against the most common pixel value (the background,
    whatever it happens to be lit as) and cropping to the content's own
    bounding box makes the comparison indifferent to exact framing and
    shading, so it isolates shape rather than incidental pixel noise.
    """
    img = np.array(Image.open(path).convert("L"))
    values, counts = np.unique(img, return_counts=True)
    background = values[np.argmax(counts)]
    mask = img != background
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=bool)
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    resized = Image.fromarray(cropped.astype(np.uint8) * 255).resize((size, size))
    return np.array(resized) > 127


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_left_and_right_views_are_not_mirrored(tmp_path):
    # Regression for a sign error in _add_camera's X offset: it put the
    # "right" camera where "left" belonged and vice versa. A file-size
    # check can't catch this -- a mirrored render is exactly as large as
    # a correct one -- so this checks the rendered content itself.
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
