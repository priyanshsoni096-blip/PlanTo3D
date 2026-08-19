"""User control over what the house is made of and what light falls on it.

A drawing does not say what a building is clad in -- that is the
architect's decision rather than the plan's -- so these were the wrong
things to have decided in code.
"""

import numpy as np
import pytest

from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.materials import SURFACES, build_scene
from planto3d.preview import _background, _shading
from planto3d.style import (
    LIGHTING_PRESETS,
    PALETTE_PRESETS,
    Lighting,
    Palette,
    parse_colour,
    to_hex,
)

OUTLINE = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def _plan():
    return FloorPlan(
        walls=[
            Wall(start=OUTLINE[i], end=OUTLINE[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ],
        footprint=list(OUTLINE),
        openings=[Opening(wall_id=0, position=200.0, width=120.0, type="window")],
    )


def _wall_colour(scene):
    return [int(c) for c in scene.geometry["wall"].visual.material.baseColorFactor[:3]]


class TestReadingAColour:
    @pytest.mark.parametrize(
        "written, expected",
        [
            ("#ff8000", (255, 128, 0)),
            ("#FF8000", (255, 128, 0)),
            ("#f80", (255, 136, 0)),
            ("rgb(255, 128, 0)", (255, 128, 0)),
            ("rgb(255,128,0)", (255, 128, 0)),
            ((255, 128, 0), (255, 128, 0)),
            ([255, 128, 0], (255, 128, 0)),
        ],
    )
    def test_every_form_a_picker_hands_back(self, written, expected):
        # Pickers return whichever of these they feel like, and a picker's
        # output arriving unusable is a poor reason to lose a choice.
        assert parse_colour(written) == expected

    @pytest.mark.parametrize("written", [None, "", "not a colour", "#12345"])
    def test_nonsense_reads_as_nothing_rather_than_raising(self, written):
        assert parse_colour(written) is None

    def test_hex_round_trips(self):
        assert parse_colour(to_hex((17, 204, 85))) == (17, 204, 85)


class TestThePalette:
    def test_a_chosen_colour_reaches_the_model(self):
        scene = build_scene(
            [_plan()], wall_height_ft=9.0, scale=20.0, palette=Palette({"wall": (158, 96, 74)})
        )

        assert _wall_colour(scene) == [158, 96, 74]

    def test_asking_for_nothing_changes_nothing(self):
        plain = build_scene([_plan()], wall_height_ft=9.0, scale=20.0)
        empty = build_scene([_plan()], wall_height_ft=9.0, scale=20.0, palette=Palette())

        assert _wall_colour(plain) == _wall_colour(empty)

    def test_surfaces_not_named_keep_their_own_finish(self):
        scene = build_scene(
            [_plan()], wall_height_ft=9.0, scale=20.0, palette=Palette({"wall": (10, 20, 30)})
        )
        roof = [int(c) for c in scene.geometry["roof"].visual.material.baseColorFactor[:3]]

        assert roof == list(SURFACES["roof"].colour)

    def test_roughness_survives_a_recolour(self):
        # Roughness is what separates glass from masonry. Recolouring a
        # surface must not quietly make it matt.
        scene = build_scene(
            [_plan()], wall_height_ft=9.0, scale=20.0, palette=Palette({"glass": (200, 40, 40)})
        )

        assert scene.geometry["glass"].visual.material.roughnessFactor == pytest.approx(
            SURFACES["glass"].roughness
        )

    def test_a_scheme_names_only_what_gives_it_character(self):
        # Otherwise every scheme is a full palette and none of them can be
        # combined with a choice of anything else.
        for name, colours in PALETTE_PRESETS.items():
            assert set(colours) <= set(SURFACES), name

    def test_with_colour_leaves_the_original_alone(self):
        original = Palette({"wall": (1, 2, 3)})
        changed = original.with_colour("roof", (4, 5, 6))

        assert "roof" not in original.colours
        assert changed.colours["wall"] == (1, 2, 3)


class TestTheLighting:
    def test_every_preset_is_a_complete_lighting(self):
        assert all(isinstance(value, Lighting) for value in LIGHTING_PRESETS.values())

    def test_the_default_matches_what_was_hard_coded(self):
        assert LIGHTING_PRESETS["midday"] == Lighting()

    def test_a_warmer_sun_shades_warmer(self):
        normals = np.array([[0.45, 0.8, 0.4]])
        normals /= np.linalg.norm(normals)

        midday = _shading(normals, lighting=LIGHTING_PRESETS["midday"])
        golden = _shading(normals, lighting=LIGHTING_PRESETS["golden hour"])

        assert golden[0, 0] / golden[0, 2] > midday[0, 0] / midday[0, 2]

    def test_overcast_flattens_the_key_light(self):
        # Light arrives from the whole sky at once, so the difference
        # between a face turned to the sun and one turned away collapses.
        towards = np.array([[0.45, 0.8, 0.4]])
        towards = towards / np.linalg.norm(towards)
        away = -towards

        def contrast(lighting):
            lit = _shading(towards, lighting=lighting).mean()
            shade = _shading(away, lighting=lighting).mean()
            return lit / shade

        assert contrast(LIGHTING_PRESETS["overcast"]) < contrast(
            LIGHTING_PRESETS["midday"]
        )

    def test_the_sky_changes_with_the_hour(self):
        midday = _background((40, 30), LIGHTING_PRESETS["midday"])
        dusk = _background((40, 30), LIGHTING_PRESETS["dusk"])

        assert not np.allclose(midday, dusk)

    def test_dusk_is_darker_overhead_than_midday(self):
        midday = _background((40, 30), LIGHTING_PRESETS["midday"])
        dusk = _background((40, 30), LIGHTING_PRESETS["dusk"])

        assert dusk[0].mean() < midday[0].mean()

    def test_asking_for_no_lighting_uses_the_default(self):
        normals = np.array([[0.0, 1.0, 0.0]])

        assert _shading(normals) == pytest.approx(
            _shading(normals, lighting=Lighting())
        )
