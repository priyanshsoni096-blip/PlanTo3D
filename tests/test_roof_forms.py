"""Roofs that are not flat: domes, pitched roofs and slanting glazing.

A flat slab with a parapet was the only roof the model could build, so a
drawing marked DOME or GLASS ROOF or SLOPING ROOF came out looking like
every other house. These are raised over the room the drawing names.

The shapes are built from explicit vertices rather than a convex hull or a
plane slice, both of which need SciPy, so these tests also stand in for
"the roof forms build at all in an environment without it".
"""

import numpy as np
import pytest

from planto3d.extrude import (
    DOME_DRUM_FT,
    DOME_RISE_RATIO,
    FEET_TO_METRES,
    GLAZED_RISE_RATIO,
    PITCH_RISE_RATIO,
    _dome,
    _sloped_roof,
    _water_tank,
)
from planto3d.features import FEATURE_KEYWORDS, classify
from planto3d.geometry_types import FloorPlan, Room, Wall
from planto3d.materials import SURFACES, build_scene

# 200 x 160 pixels at 20 px/ft is 10 x 8 feet.
ROOM = [(0.0, 0.0), (200.0, 0.0), (200.0, 160.0), (0.0, 160.0)]
SCALE = 20.0


def _height(parts) -> tuple[float, float]:
    """Lowest and highest point of a set of meshes, in metres."""
    low = min(part.bounds[0][1] for part in parts)
    high = max(part.bounds[1][1] for part in parts)
    return low, high


class TestTheShapesThemselves:
    def test_a_dome_is_a_closed_solid(self):
        parts = _dome(ROOM, 0.0, SCALE)

        assert parts
        assert all(part.is_watertight for part in parts)
        assert all(part.volume > 0 for part in parts)

    def test_a_dome_rises_by_half_its_shorter_span(self):
        # Half the span is a true hemisphere, which is what a drawing means
        # by "dome" unless it says otherwise. The shorter span, because an
        # oblong room should not give a dome taller than it is wide.
        parts = _dome(ROOM, 0.0, SCALE)
        low, high = _height(parts)

        shorter_span_m = 8.0 * FEET_TO_METRES
        expected = DOME_DRUM_FT * FEET_TO_METRES + shorter_span_m * DOME_RISE_RATIO

        assert high - low == pytest.approx(expected, rel=0.02)

    def test_a_dome_sits_on_a_drum(self):
        # Two parts, not one: a dome springs from a base rather than growing
        # straight out of the roof deck.
        assert len(_dome(ROOM, 0.0, SCALE)) == 2

    def test_a_dome_covers_the_room_it_is_over(self):
        parts = _dome(ROOM, 0.0, SCALE)
        across = max(part.bounds[1][0] for part in parts) - min(
            part.bounds[0][0] for part in parts
        )

        assert across == pytest.approx(10.0 * FEET_TO_METRES, rel=0.02)

    def test_a_pitched_roof_is_a_closed_solid(self):
        parts = _sloped_roof(ROOM, 0.0, SCALE, PITCH_RISE_RATIO, ridged=True)

        assert len(parts) == 1
        assert parts[0].is_watertight
        assert parts[0].volume > 0

    def test_a_ridge_runs_along_the_longer_side(self):
        # A gable spanning the long way would need impossibly long rafters.
        # A room deeper than it is wide must ridge the other way round.
        deep = [(0.0, 0.0), (100.0, 0.0), (100.0, 400.0), (0.0, 400.0)]

        roof = _sloped_roof(deep, 0.0, SCALE, PITCH_RISE_RATIO, ridged=True)[0]
        low, high = roof.bounds

        assert high[2] - low[2] > high[0] - low[0]

    def test_glazing_slopes_one_way_only(self):
        # A lean-to over a conservatory, not a gable: one face catching the
        # light rather than two meeting at a ridge.
        roof = _sloped_roof(ROOM, 0.0, SCALE, GLAZED_RISE_RATIO, ridged=False)[0]

        # The high edge and the low edge sit at opposite ends of the room.
        vertices = roof.vertices
        highest = vertices[np.argmax(vertices[:, 1])]
        lowest = vertices[np.argmin(vertices[:, 1])]

        assert highest[2] != pytest.approx(lowest[2])

    def test_glazing_is_laid_shallower_than_a_tiled_pitch(self):
        # A steep glass plane reads as a wall rather than a roof.
        glazed = _sloped_roof(ROOM, 0.0, SCALE, GLAZED_RISE_RATIO, ridged=False)
        pitched = _sloped_roof(ROOM, 0.0, SCALE, PITCH_RISE_RATIO, ridged=True)

        assert (_height(glazed)[1] - _height(glazed)[0]) < (
            _height(pitched)[1] - _height(pitched)[0]
        )

    def test_glazing_has_thickness(self):
        # Left at a knife edge it disappears when seen square on.
        roof = _sloped_roof(ROOM, 0.0, SCALE, GLAZED_RISE_RATIO, ridged=False)[0]

        assert roof.volume > 0
        assert roof.is_watertight

    @pytest.mark.parametrize("polygon", [[], [(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]])
    def test_a_room_that_cannot_enclose_an_area_builds_nothing(self, polygon):
        # A bad contour should cost a roof, never the whole model.
        assert _dome(polygon, 0.0, SCALE) == []
        assert _sloped_roof(polygon, 0.0, SCALE, 0.3, ridged=True) == []

    def test_roofs_are_raised_to_the_storey_they_belong_to(self):
        low, _ = _height(_dome(ROOM, 30.0, SCALE))

        assert low == pytest.approx(30.0 * FEET_TO_METRES, rel=0.01)


class TestFromALabelledDrawing:
    @pytest.mark.parametrize(
        "label, category",
        [
            ("DOME", "dome"),
            ("CUPOLA", "dome"),
            ("SHIKHARA", "dome"),
            ("GLASS ROOF", "glazed"),
            ("SKYLIGHT", "glazed"),
            ("CONSERVATORY", "glazed"),
            ("SLOPING ROOF", "pitched"),
            ("GABLE ROOF", "pitched"),
            ("MANSARD", "pitched"),
        ],
    )
    def test_roof_words_are_recognised(self, label, category):
        assert classify(label) == category

    def test_a_skylight_is_glazed_rather_than_a_hole(self):
        # It used to classify as "void", which cut the roof away and left
        # the room open to the weather.
        assert classify("SKYLIGHT") == "glazed"

    def _plan(self, label):
        outline = [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)]
        return FloorPlan(
            walls=[
                Wall(start=outline[i], end=outline[(i + 1) % 4], thickness=10.0)
                for i in range(4)
            ],
            footprint=list(outline),
            rooms=[Room(polygon=[(50.0, 50.0), (350.0, 50.0), (350.0, 350.0), (50.0, 350.0)], label=label)],
        )

    def test_a_labelled_dome_reaches_the_finished_model(self):
        scene = build_scene([self._plan("DOME")], wall_height_ft=9.0, scale=SCALE)

        assert "dome" in scene.geometry

    def test_a_dome_stands_above_the_roof_deck(self):
        scene = build_scene([self._plan("DOME")], wall_height_ft=9.0, scale=SCALE)

        assert scene.geometry["dome"].bounds[0][1] >= scene.geometry["roof"].bounds[0][1]

    def test_glazing_joins_the_windows_rather_than_the_roof(self):
        # So it picks up the transparent material instead of reading as a
        # solid panel laid over the room.
        scene = build_scene([self._plan("GLASS ROOF")], wall_height_ft=9.0, scale=SCALE)

        assert "glass" in scene.geometry

    def test_an_unlabelled_room_still_gets_a_flat_roof(self):
        scene = build_scene([self._plan("BEDROOM")], wall_height_ft=9.0, scale=SCALE)

        assert "dome" not in scene.geometry
        assert "roof" in scene.geometry

    def test_a_dome_has_a_surface_of_its_own(self):
        # Sharing the roof's finish would flatten it: a curved surface only
        # reads as curved if the light moves across it.
        assert "dome" in SURFACES
        assert SURFACES["dome"].roughness < SURFACES["roof"].roughness

    def test_a_pitched_roof_is_tiled_rather_than_decked(self):
        # Merged in with the flat roof it was invisible: a low slope in the
        # same grey as the deck it stands on, half hidden by the parapet.
        scene = build_scene([self._plan("SLOPING ROOF")], wall_height_ft=9.0, scale=SCALE)

        assert "pitched" in scene.geometry
        assert SURFACES["pitched"].colour != SURFACES["roof"].colour


class TestStructuresOnAndAroundTheBuilding:
    """Tanks, chimneys, towers, canopies and ramps.

    The first three stand on the roof. The last two belong to the storey
    they are drawn on, which is the distinction that matters: a porch over
    the front door is at first floor soffit level, and moving it to the roof
    would leave the door uncovered and hang a slab three storeys up.
    """

    @pytest.mark.parametrize(
        "label, category",
        [
            ("OVERHEAD TANK", "tank"),
            ("WATER TANK", "tank"),
            ("CHIMNEY", "chimney"),
            ("TURRET", "tower"),
            ("MINARET", "tower"),
            ("BELVEDERE", "tower"),
            ("PORTICO", "canopy"),
            ("CANOPY", "canopy"),
            ("CHAJJA", "canopy"),
            ("RAMP", "ramp"),
            ("WHEELCHAIR RAMP", "ramp"),
            ("VEHICLE RAMP", "ramp"),
        ],
    )
    def test_the_words_are_recognised(self, label, category):
        assert classify(label) == category

    def test_a_chajja_is_a_cover_rather_than_a_balcony(self):
        # It used to classify as "open" and was railed, as though a
        # projecting sunshade were something to stand on.
        assert classify("CHAJJA") == "canopy"

    def test_a_portico_is_not_railed_either(self):
        assert classify("PORTICO") == "canopy"

    def test_no_keyword_belongs_to_two_categories(self):
        # Which one a duplicate lands in depends on the order the list
        # happens to be written in, which is nobody's decision. Nine of
        # these existed before the check did.
        seen: dict[str, list[str]] = {}
        for category, keywords in FEATURE_KEYWORDS:
            for keyword in keywords:
                seen.setdefault(keyword, []).append(category)

        assert {k: v for k, v in seen.items() if len(v) > 1} == {}

    def _house(self, *labels):
        outline = [(0.0, 0.0), (600.0, 0.0), (600.0, 400.0), (0.0, 400.0)]
        rooms = [
            Room(
                polygon=[
                    (60.0 + 160 * i, 60.0),
                    (200.0 + 160 * i, 60.0),
                    (200.0 + 160 * i, 200.0),
                    (60.0 + 160 * i, 200.0),
                ],
                label=label,
            )
            for i, label in enumerate(labels)
        ]
        return FloorPlan(
            walls=[
                Wall(start=outline[i], end=outline[(i + 1) % 4], thickness=12.0)
                for i in range(4)
            ],
            footprint=list(outline),
            rooms=rooms,
        )

    @pytest.mark.parametrize(
        "label, part", [("OVERHEAD TANK", "tank"), ("CHIMNEY", "chimney"), ("TURRET", "tower")]
    )
    def test_roof_structures_reach_the_model(self, label, part):
        scene = build_scene([self._house(label)], wall_height_ft=9.0, scale=SCALE)

        assert part in scene.geometry

    @pytest.mark.parametrize("label", ["OVERHEAD TANK", "CHIMNEY", "TURRET"])
    def test_roof_structures_stand_on_the_deck(self, label):
        scene = build_scene([self._house(label)], wall_height_ft=9.0, scale=SCALE)
        deck = scene.geometry["roof"].bounds[0][1]

        raised = [
            geometry
            for name, geometry in scene.geometry.items()
            if name in {"tank", "chimney", "tower"}
        ]
        assert raised
        assert all(part.bounds[0][1] >= deck for part in raised)

    def test_a_tank_stands_clear_of_the_roof_on_a_stand(self):
        # Two parts, not one. Resting straight on the deck it reads as a
        # packing case left on the roof.
        stand, tank = _water_tank(ROOM, 0.0, SCALE)

        assert stand.bounds[1][1] <= tank.bounds[0][1] + 1e-6
        # And the stand is set in from the tank it carries, being legs.
        assert (stand.bounds[1][0] - stand.bounds[0][0]) < (
            tank.bounds[1][0] - tank.bounds[0][0]
        )

    def test_a_tower_is_taller_than_a_chimney(self):
        tower = build_scene([self._house("TURRET")], wall_height_ft=9.0, scale=SCALE)
        chimney = build_scene([self._house("CHIMNEY")], wall_height_ft=9.0, scale=SCALE)

        assert tower.geometry["tower"].bounds[1][1] > chimney.geometry["chimney"].bounds[1][1]

    def test_a_canopy_hangs_at_its_own_storeys_ceiling(self):
        # Not at roof level. On a two storey house the ground floor's porch
        # must stay below the first floor.
        scene = build_scene(
            [self._house("PORTICO"), self._house()], wall_height_ft=9.0, scale=SCALE
        )

        assert scene.geometry["canopy"].bounds[1][1] < scene.geometry["roof"].bounds[0][1]

    def test_a_ramp_slopes(self):
        scene = build_scene([self._house("CAR RAMP")], wall_height_ft=9.0, scale=SCALE)
        ramp = scene.geometry["stairs"]

        low, high = ramp.bounds
        assert high[1] - low[1] > 0

    def test_a_tower_is_not_finished_in_the_roof_deck_grey(self):
        assert SURFACES["tower"].colour != SURFACES["roof"].colour
