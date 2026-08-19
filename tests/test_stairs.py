"""Stairs, without which a storey has no visible way to reach the next."""

import pytest

from planto3d.extrude import RISER_HEIGHT_FT, _stair_parts, floors_to_parts
from planto3d.features import classify
from planto3d.geometry_types import FloorPlan, Room, Wall

SCALE = 20.0
FEET_TO_METRES = 0.3048
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
# A stairwell 12ft along the climb by 4ft across, at 20px/ft.
STAIRWELL = [(100.0, 100.0), (340.0, 100.0), (340.0, 180.0), (100.0, 180.0)]


@pytest.mark.parametrize(
    "label", ["STAIRCASE", "STAIR", "STAIRS", "STEPS", "LANDING", "UP STAIR"]
)
def test_stair_labels_are_recognised(label):
    assert classify(label) == "stairs"


@pytest.mark.parametrize("label", ["UP", "DN", "DOWN", "UP ", "* UP"])
def test_the_bare_stair_marks_are_recognised(label):
    # A flight is often annotated with nothing but an arrow and "UP", which
    # is all OCR returns from the reference sheets.
    assert classify(label) == "stairs"


@pytest.mark.parametrize("label", ["GROUP ROOM", "CUP BOARD", "DINING", "UPPER HALL"])
def test_short_marks_do_not_match_inside_longer_words(label):
    # "UP" and "DN" are only meaningful as whole words.
    assert classify(label) != "stairs"


def test_a_longer_phrase_still_wins_over_a_bare_mark():
    # "DOUBLE HEIGHT UP" is a void with a stair mark nearby, not a stairwell.
    assert classify("DOUBLE HEIGHT UP") == "void"


class TestFlightGeometry:
    def _flight(self, polygon=None, rise=9.0):
        return _stair_parts(polygon or STAIRWELL, base_ft=0.0, rise_ft=rise, scale=SCALE)

    def test_a_flight_has_a_step_per_riser(self):
        steps = self._flight()

        assert len(steps) == pytest.approx(9.0 / RISER_HEIGHT_FT, abs=1)

    def test_a_taller_storey_gets_more_steps(self):
        # Sized from a comfortable riser rather than a fixed count.
        assert len(self._flight(rise=12.0)) > len(self._flight(rise=9.0))

    def test_the_flight_reaches_the_storey_above(self):
        steps = self._flight(rise=9.0)

        top = max(s.bounds[1][1] for s in steps)
        assert top == pytest.approx(9.0 * FEET_TO_METRES, abs=0.1)

    def test_the_flight_starts_at_the_floor(self):
        steps = self._flight()

        assert min(s.bounds[0][1] for s in steps) == pytest.approx(0.0, abs=0.01)

    def test_steps_rise_monotonically_along_the_flight(self):
        steps = self._flight()
        heights = [s.bounds[1][1] for s in steps]

        assert heights == sorted(heights)

    def test_the_flight_climbs_the_stairwell_s_long_axis(self):
        # A straight flight needs length for the run; the width only has to
        # fit a person, so the climb must follow the longer dimension.
        wide = _stair_parts(
            [(0.0, 0.0), (400.0, 0.0), (400.0, 80.0), (0.0, 80.0)], 0.0, 9.0, SCALE
        )
        tall = _stair_parts(
            [(0.0, 0.0), (80.0, 0.0), (80.0, 400.0), (0.0, 400.0)], 0.0, 9.0, SCALE
        )

        wide_span = max(s.bounds[1][0] for s in wide) - min(s.bounds[0][0] for s in wide)
        tall_span = max(s.bounds[1][2] for s in tall) - min(s.bounds[0][2] for s in tall)
        assert wide_span > tall_span * 0.9  # each climbs its own long axis

    def test_the_flight_stays_within_its_stairwell(self):
        steps = self._flight()

        assert min(s.bounds[0][0] for s in steps) >= 100.0 / SCALE * FEET_TO_METRES - 0.1
        assert max(s.bounds[1][0] for s in steps) <= 340.0 / SCALE * FEET_TO_METRES + 0.1

    def test_a_degenerate_outline_produces_no_flight(self):
        assert _stair_parts([(0.0, 0.0), (10.0, 0.0)], 0.0, 9.0, SCALE) == []

    def test_no_rise_produces_no_flight(self):
        assert self._flight(rise=0.0) == []


class TestStairsInTheModel:
    def _floor(self, rooms=None):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        return FloorPlan(walls=walls, footprint=list(FOOTPRINT), rooms=rooms or [])

    def test_a_labelled_stairwell_becomes_a_flight(self):
        parts = floors_to_parts(
            [self._floor([Room(polygon=STAIRWELL, label="STAIRCASE")])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert "stairs" in parts

    def test_a_plan_without_stairs_gets_none(self):
        parts = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)

        assert "stairs" not in parts

    def test_an_upper_storey_s_flight_starts_at_its_own_floor(self):
        from planto3d.extrude import SLAB_THICKNESS_FT
        from planto3d.site import PLINTH_HEIGHT_FT

        parts = floors_to_parts(
            [self._floor(), self._floor([Room(polygon=STAIRWELL, label="STAIR")])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        # Measured from the plinth the building stands on, not from the
        # site. A storey occupies its wall height plus the slab it stands
        # on, so the upper floor's own slab sits two slabs up: its own, and
        # the one the storey below stands on.
        expected = PLINTH_HEIGHT_FT + 9.0 + 2 * SLAB_THICKNESS_FT
        lowest = min(s.bounds[0][1] for s in parts["stairs"])
        assert lowest == pytest.approx(expected * FEET_TO_METRES, abs=0.1)
