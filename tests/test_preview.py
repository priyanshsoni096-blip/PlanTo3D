import numpy as np
import trimesh
from PIL import Image

from planto3d.materials import build_scene, export_scene
from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.preview import _material_colour, render, render_glb

FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def _pixels(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def test_a_near_surface_hides_what_is_behind_it(tmp_path):
    # The bug this replaced: faces sorted by average depth, so a large slab
    # and the small boxes on it interleaved wrongly and punched holes in each
    # other. A big near panel must completely cover a small far one.
    far = trimesh.creation.box(extents=[1, 1, 1])
    far.apply_translation([0, 0, -6])
    near = trimesh.creation.box(extents=[8, 8, 1])

    colours = np.vstack(
        [
            np.tile([255, 0, 0], (len(far.faces), 1)),
            np.tile([0, 0, 255], (len(near.faces), 1)),
        ]
    )
    path = render(
        trimesh.util.concatenate([far, near]),
        tmp_path / "occlusion.png",
        resolution=(200, 200),
        azimuth=0,
        elevation=0,
        face_colours=colours,
    )

    pixels = _pixels(path).reshape(-1, 3).astype(int)
    # Nothing of the far box should survive: no pixel may be more red than blue.
    assert not ((pixels[:, 0] > pixels[:, 2] + 30).any())


def test_render_writes_an_image_of_the_requested_size(tmp_path):
    path = render(
        trimesh.creation.box(extents=[1, 1, 1]),
        tmp_path / "box.png",
        resolution=(320, 240),
    )

    assert _pixels(path).shape == (240, 320, 3)


def test_the_model_is_actually_drawn_not_just_background(tmp_path):
    path = render(
        trimesh.creation.box(extents=[1, 1, 1]),
        tmp_path / "box.png",
        resolution=(160, 160),
    )

    # More than one distinct colour means geometry landed on the canvas.
    assert len(np.unique(_pixels(path).reshape(-1, 3), axis=0)) > 3


def test_material_colour_accepts_both_glTF_conventions():
    # 0-1 floats as written, 0-255 bytes as trimesh returns after a reload.
    class Visual:
        def __init__(self, factor):
            self.material = type("M", (), {"baseColorFactor": factor})()

    float_form = type("G", (), {"visual": Visual([0.5, 0.25, 1.0, 1.0])})()
    byte_form = type("G", (), {"visual": Visual([128, 64, 255, 255])})()

    assert np.allclose(_material_colour(float_form), [127.5, 63.75, 255.0])
    assert np.allclose(_material_colour(byte_form), [128.0, 64.0, 255.0])


def test_materials_change_what_is_drawn(tmp_path):
    # render_glb must carry each part's material colour through, rather than
    # painting the whole building one tone.
    walls = [Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0) for i in range(4)]
    plan = FloorPlan(
        walls=walls,
        footprint=list(FOOTPRINT),
        openings=[Opening(wall_id=0, position=200.0, width=120.0, type="window")],
    )
    model = tmp_path / "house.glb"
    export_scene(build_scene([plan], wall_height_ft=9.0, scale=20.0), model)

    with_materials = _pixels(render_glb(model, tmp_path / "a.png", resolution=(400, 300)))
    uniform = _pixels(
        render(
            trimesh.load(str(model), force="mesh"),
            tmp_path / "b.png",
            resolution=(400, 300),
        )
    )

    assert not np.array_equal(with_materials, uniform)
    # Several distinct surface tones, not one flat colour over the model.
    assert len(np.unique(with_materials.reshape(-1, 3), axis=0)) > 10
