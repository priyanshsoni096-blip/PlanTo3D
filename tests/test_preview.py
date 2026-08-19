import numpy as np
import pytest
import trimesh
from PIL import Image

from planto3d.materials import build_scene, export_scene
from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.preview import (
    KEY_LIGHT,
    _material_colour,
    _occlusion,
    _shading,
    _specular,
    _to_linear,
    _to_srgb,
    _tonemap,
    render,
    render_glb,
    render_views,
)

FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def _pixels(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _drawn(pixels: np.ndarray) -> np.ndarray:
    """Coordinates of pixels showing geometry rather than sky.

    The sky is a vertical gradient, so it is constant across any row. Anything
    differing from its own row's most common value is the model. Testing
    brightness instead would fail the moment the sky stopped being dark.
    """
    rows = np.median(pixels.reshape(pixels.shape[0], -1, 3), axis=1)
    difference = np.abs(pixels.astype(int) - rows[:, None, :]).sum(axis=2)
    return np.argwhere(difference > 30)


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


class TestStandardViews:
    def _model(self, tmp_path):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        plan = FloorPlan(
            walls=walls,
            footprint=list(FOOTPRINT),
            openings=[Opening(wall_id=0, position=200.0, width=120.0, type="window")],
        )
        path = tmp_path / "house.glb"
        export_scene(build_scene([plan], wall_height_ft=9.0, scale=20.0), path)
        return path

    def test_every_standard_view_is_produced(self, tmp_path):
        views = render_views(self._model(tmp_path), tmp_path, resolution=(200, 160))

        assert set(views) == {"top", "front", "back", "left", "right", "aerial"}
        assert all(path.exists() for path in views.values())

    def test_each_view_shows_something_different(self, tmp_path):
        views = render_views(self._model(tmp_path), tmp_path, resolution=(200, 160))

        rendered = {name: _pixels(path).tobytes() for name, path in views.items()}
        assert len(set(rendered.values())) == len(rendered)

    def test_the_plan_view_is_wider_than_it_is_tall(self, tmp_path):
        # Seen from above, a 400x300 footprint must read landscape, not
        # square-on like an elevation.
        views = render_views(self._model(tmp_path), tmp_path, resolution=(400, 400))

        pixels = _pixels(views["top"])
        drawn = _drawn(pixels)

        assert np.ptp(drawn[:, 1]) > np.ptp(drawn[:, 0])

    def test_elevations_are_no_taller_than_the_building(self, tmp_path):
        # A square-on elevation shows storey height, not the plan's depth.
        views = render_views(self._model(tmp_path), tmp_path, resolution=(400, 400))

        pixels = _pixels(views["front"])
        drawn = _drawn(pixels)

        assert np.ptp(drawn[:, 1]) > np.ptp(drawn[:, 0])

    def test_views_can_be_named_with_a_prefix(self, tmp_path):
        views = render_views(
            self._model(tmp_path), tmp_path, resolution=(160, 120), prefix="soni"
        )

        assert all(path.name.startswith("soni-") for path in views.values())


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


class TestTheLight:
    """Colour, roll-off and contact darkening.

    Lit by one scalar brightness every surface is the same hue at a
    different level, which is what grey card looks like. What reads as
    sunlight is the shift towards warm on the lit face and towards blue in
    the shade, so the shading has to return colour, not a number.
    """

    def _normals(self):
        return np.array(
            [
                [0.0, 1.0, 0.0],   # up, sees sky
                [0.0, -1.0, 0.0],  # down, sees the ground bounce
                [1.0, 0.0, 0.0],
            ]
        )

    def test_shading_returns_a_colour_per_face(self):
        light = _shading(self._normals())

        assert light.shape == (3, 3)

    def test_a_face_looking_up_is_cooler_than_one_looking_down(self):
        # The hemisphere: sky above is blue, what bounces off the ground is
        # warm. That single gradient does most of the work of making a
        # massing read as solid, before any direct light lands on it.
        up, down, _ = _shading(self._normals())

        assert up[2] / up[0] > down[2] / down[0]

    def test_light_is_not_grey(self):
        # The failure this replaced: every channel scaled by one number.
        light = _shading(self._normals())

        assert not np.allclose(light[:, 0], light[:, 2])

    def test_a_rough_surface_barely_glints(self):
        # Measured square into the highlight, since a narrow one is only
        # visible to a face pointing straight down the halfway vector --
        # which is the point of it being narrow.
        key = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
        halfway = key + np.array([0.0, 0.0, 1.0])
        normals = np.array([halfway / np.linalg.norm(halfway)])

        matt = _specular(normals, np.array([0.95]))
        polished = _specular(normals, np.array([0.08]))

        assert polished[0, 0] > matt[0, 0]

    def test_a_narrow_highlight_falls_away_off_axis(self):
        key = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
        halfway = key + np.array([0.0, 0.0, 1.0])
        halfway /= np.linalg.norm(halfway)

        on_axis = _specular(np.array([halfway]), np.array([0.1]))
        off_axis = _specular(np.array([[0.0, -1.0, 0.0]]), np.array([0.1]))

        assert on_axis[0, 0] > off_axis[0, 0]

    def test_no_highlight_without_a_material_to_take_it_from(self):
        assert _specular(self._normals(), None).sum() == 0

    def test_tonemapping_rolls_off_instead_of_clipping(self):
        # Two values well past white must stay distinguishable. Clipped,
        # both land on 1.0 and a sunlit parapet becomes a hard blank shape.
        bright, brighter = _tonemap(np.array([1.6])), _tonemap(np.array([2.6]))

        assert brighter[0] > bright[0]
        assert brighter[0] <= 1.0

    def test_black_stays_black_and_nothing_exceeds_white(self):
        values = _tonemap(np.array([0.0, 0.5, 1.0, 12.0]))

        assert values[0] == pytest.approx(0.0, abs=1e-6)
        assert values.max() <= 1.0

    def test_srgb_round_trips(self):
        original = np.array([0.0, 18.0, 128.0, 255.0])

        assert _to_srgb(_to_linear(original)) == pytest.approx(original, abs=0.5)

    def test_linear_is_not_the_stored_value(self):
        # Mid grey is about 21% of the light of white, not 50%. Doing the
        # arithmetic on stored bytes gets every shaded surface wrong.
        assert _to_linear(np.array([128.0]))[0] == pytest.approx(0.216, abs=0.01)

    def test_a_pixel_behind_its_neighbours_is_darkened(self):
        # A crevice: geometry all round it nearer than it is.
        depth = np.full((21, 21), 1.0)
        depth[10, 10] = 0.0

        assert _occlusion(depth)[10, 10] < _occlusion(depth)[0, 0]

    def test_an_empty_depth_buffer_darkens_nothing(self):
        assert np.all(_occlusion(np.full((8, 8), -np.inf)) == 1.0)

    def test_supersampling_softens_the_edges(self, tmp_path):
        # An aliased edge is a hard switch between two colours. An
        # antialiased one blends across it, so it introduces tones that
        # exist nowhere else in the image -- counting distinct colours is
        # the simplest way to see that having happened.
        box = trimesh.creation.box(extents=[1, 1, 1])
        box.apply_transform(trimesh.transformations.rotation_matrix(0.4, [0, 0, 1]))

        def tones(path):
            return len(np.unique(_pixels(path).reshape(-1, 3), axis=0))

        rough = render(box, tmp_path / "a.png", resolution=(200, 200), supersample=1)
        smooth = render(box, tmp_path / "b.png", resolution=(200, 200), supersample=3)

        assert tones(smooth) > tones(rough)
