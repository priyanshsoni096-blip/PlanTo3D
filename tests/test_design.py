"""The five choices a drawing cannot make for you.

A plan fixes the geometry and says nothing about the building: not what it
is clad in, not what hour it is seen at, not whether there is a garden.
"""

import pytest

from planto3d.design import (
    CREATIVITY,
    LANDSCAPING,
    STYLES,
    TIMES,
    TONES,
    UNTONED,
    Design,
    Tone,
    apply_tone,
)
from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.materials import SURFACES, build_scene
from planto3d.style import LIGHTING_PRESETS

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


def _lightness(colour) -> float:
    return sum(colour) / 3


class TestTone:
    def test_light_lifts_and_dark_drops(self):
        base = (150, 140, 130)

        assert _lightness(apply_tone(base, TONES["light"])) > _lightness(base)
        assert _lightness(apply_tone(base, TONES["dark"])) < _lightness(base)

    def test_nothing_is_pushed_past_black_or_white(self):
        # Shifting towards the limit rather than by a flat amount, so a
        # dark surface cannot wrap around.
        for colour in ((4, 4, 4), (252, 252, 252)):
            for tone in TONES.values():
                shifted = apply_tone(colour, tone)
                assert all(0 <= channel <= 255 for channel in shifted)

    def test_warm_moves_towards_orange(self):
        warmed = apply_tone((150, 150, 150), TONES["warm"])

        assert warmed[0] > warmed[2]

    def test_a_brick_stays_brick_when_darkened(self):
        # Scaling the channels directly turns a saturated colour grey. In
        # HLS the hue survives, which is the point of working there.
        brick = (176, 96, 72)
        darkened = apply_tone(brick, TONES["dark"])

        assert darkened[0] > darkened[1] > darkened[2]

    def test_no_shift_changes_nothing_much(self):
        assert apply_tone((128, 120, 110), Tone()) == pytest.approx(
            (128, 120, 110), abs=2
        )


class TestDesign:
    def test_the_defaults_are_a_complete_design(self):
        design = Design()

        assert design.palette().colours
        assert design.lighting() in LIGHTING_PRESETS.values()
        assert 0 < design.conditioning() <= 1

    @pytest.mark.parametrize("style", list(STYLES))
    @pytest.mark.parametrize("tone", list(TONES))
    def test_every_combination_resolves(self, style, tone):
        colours = Design(style=style, colour=tone).palette().colours

        assert colours
        assert all(
            all(0 <= channel <= 255 for channel in value) for value in colours.values()
        )

    def test_styles_name_only_real_surfaces(self):
        for name, colours in STYLES.items():
            assert set(colours) <= set(SURFACES), name

    def test_two_styles_do_not_look_the_same(self):
        modern = Design(style="modern").palette().colours
        traditional = Design(style="traditional").palette().colours

        assert modern["wall"] != traditional["wall"]

    def test_glazing_is_never_retinted(self):
        # A scheme that darkens the windows is how you get a building with
        # no windows.
        for surface in UNTONED:
            for style in STYLES:
                assert surface not in Design(style=style, colour="dark").palette().colours

    def test_the_hour_changes_the_light(self):
        assert Design(time="day").lighting() != Design(time="night").lighting()

    def test_creativity_orders_from_strict_to_loose(self):
        assert (
            Design(creativity="strict").conditioning()
            > Design(creativity="balanced").conditioning()
            > Design(creativity="creative").conditioning()
        )

    def test_an_unknown_choice_falls_back_rather_than_raising(self):
        # These arrive from a dropdown, and a stale value should cost a
        # look rather than the whole build.
        design = Design(style="art deco", colour="puce", time="dawn")

        assert design.palette().colours
        assert design.lighting() == LIGHTING_PRESETS["midday"]


class TestLandscaping:
    def _built(self, level):
        scene = build_scene(
            [_plan()],
            wall_height_ft=9.0,
            scale=20.0,
            site=LANDSCAPING[level],
        )
        return set(scene.geometry)

    def test_none_leaves_the_building_alone(self):
        assert "ground" not in self._built("none")
        assert "boundary" not in self._built("none")

    def test_basic_gives_it_a_plot_but_no_wall(self):
        built = self._built("basic")

        assert "ground" in built
        assert "boundary" not in built

    def test_premium_walls_the_plot(self):
        assert "boundary" in self._built("premium")

    def test_the_building_itself_is_never_affected(self):
        for level in LANDSCAPING:
            assert {"wall", "roof", "glass"} <= self._built(level)
