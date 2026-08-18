"""A drawing whose dimensions cannot be read must still produce a model."""

import numpy as np
import pytest

from planto3d.calibrate import ASSUMED_DRAWING_RATIO, TextBox, assumed_scale, estimate_scale
from planto3d.geometry_types import Room
from planto3d.ingest import WORKING_DPI


class TestAssumedScale:
    def test_follows_from_the_drafting_ratio_and_resolution(self):
        # At 1:150, one foot occupies 12/150 inch of paper; at 400 dpi that
        # is 32 pixels. Derived, not invented.
        assert assumed_scale(400, ratio=150.0) == pytest.approx(32.0)

    def test_lands_near_the_measured_scale_of_a_real_sheet(self):
        # The reference sheet measures 28.15 px/ft. The assumption must be
        # close enough that the model is a believable size.
        measured = 28.15
        assumed = assumed_scale(WORKING_DPI)

        assert 0.7 < assumed / measured < 1.5

    def test_a_finer_ratio_gives_more_pixels_per_foot(self):
        assert assumed_scale(400, ratio=100.0) > assumed_scale(400, ratio=200.0)

    def test_higher_resolution_gives_more_pixels_per_foot(self):
        assert assumed_scale(600) > assumed_scale(300)

    def test_the_default_ratio_is_used_when_none_is_given(self):
        assert assumed_scale(400) == assumed_scale(400, ratio=ASSUMED_DRAWING_RATIO)


class TestMeasuredScaleStillWins:
    def test_a_readable_drawing_is_measured_not_assumed(self):
        # The fallback must never displace a real measurement.
        room = Room(polygon=[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)])
        box = TextBox(text="10'0\"X10'0\"", bbox=(100, 100, 10, 5), confidence=90.0)

        assert estimate_scale([room], [box]) == pytest.approx(20.0)

    def test_an_unreadable_drawing_reports_no_measurement(self):
        # estimate_scale stays honest and returns None; substituting a guess
        # is the caller's decision, so it can be reported as assumed.
        room = Room(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

        assert estimate_scale([room], []) is None
