"""A drawing whose dimensions cannot be read must still produce a model."""

import numpy as np
import pytest

from planto3d.calibrate import ASSUMED_DRAWING_RATIO, TextBox, assumed_scale, estimate_scale
from planto3d.geometry_types import Room
from planto3d.pipeline import MAX_SCALE_DISAGREEMENT, PipelineResult
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


class TestScaleFromStandardElements:
    def _door(self, width):
        from planto3d.geometry_types import Opening

        return Opening(wall_id=0, position=100.0, width=width, type="door")

    def _window(self, width):
        from planto3d.geometry_types import Opening

        return Opening(wall_id=0, position=100.0, width=width, type="window")

    def test_doors_recover_the_scale_of_the_real_sheet(self):
        from planto3d.calibrate import scale_from_doors

        # 23 doors with a median of 68px were detected on the reference
        # sheet, whose printed dimensions give 28.15 px/ft.
        doors = [self._door(w) for w in [55, 62, 66, 68, 68, 70, 74, 88, 96]]

        scale = scale_from_doors(doors)

        assert scale == pytest.approx(28.15, rel=0.1)

    def test_the_median_resists_a_wide_main_door(self):
        from planto3d.calibrate import scale_from_doors

        ordinary = [self._door(68) for _ in range(8)]
        scale = scale_from_doors(ordinary + [self._door(400)])

        assert scale == pytest.approx(scale_from_doors(ordinary))

    def test_windows_are_not_used_as_door_references(self):
        # Window widths vary hugely; only doors are standard enough.
        from planto3d.calibrate import scale_from_doors

        assert scale_from_doors([self._window(300) for _ in range(9)]) is None

    def test_too_few_doors_gives_no_estimate(self):
        from planto3d.calibrate import scale_from_doors

        assert scale_from_doors([self._door(68), self._door(70)]) is None

    def test_wall_thickness_lands_in_the_right_region(self):
        from planto3d.calibrate import scale_from_walls
        from planto3d.geometry_types import Wall

        # Median wall on the reference sheet is 23.5px against 28.15 px/ft.
        walls = [
            Wall(start=(0.0, 0.0), end=(100.0, 0.0), thickness=t)
            for t in [12, 18, 22, 23, 23.5, 24, 26, 30, 34, 40]
        ]

        assert scale_from_walls(walls) == pytest.approx(28.15, rel=0.25)

    def test_too_few_walls_gives_no_estimate(self):
        from planto3d.calibrate import scale_from_walls
        from planto3d.geometry_types import Wall

        walls = [Wall(start=(0.0, 0.0), end=(10.0, 0.0), thickness=20.0)]

        assert scale_from_walls(walls) is None


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


class TestScaleConfidence:
    """Two estimates disagreeing is the drawing arguing with itself.

    Doors and the wall gauge measure different things and are wrong in
    different ways, so the gap between them says something neither says
    alone. Measured over 48 plans: where they agree the size lands within
    a fifth of true 22 times in 24, and where they do not, 11 times in 24.
    """

    def _result(self, **kwargs):
        return PipelineResult(floors=[], scale=30.0, model_path=None, **kwargs)

    def test_a_printed_size_is_believed_without_a_second_opinion(self):
        # The architect wrote it down. It has already been checked against
        # the geometry by the gate in calibrate; nothing more is needed.
        for source in ("dimensions", "areas"):
            assert self._result(scale_source=source, scale_agreement=None).scale_confident

    def test_estimates_that_agree_are_trusted(self):
        assert self._result(scale_source="doors", scale_agreement=0.03).scale_confident

    def test_estimates_that_argue_are_not(self):
        assert not self._result(scale_source="doors", scale_agreement=0.45).scale_confident

    def test_one_estimate_alone_is_not_confidence(self):
        # Nothing corroborated it. That is not the same as it being wrong,
        # and it is not the same as it being checked either.
        assert not self._result(scale_source="walls", scale_agreement=None).scale_confident

    def test_the_flag_does_not_touch_the_size(self):
        # Confidence is reported alongside the answer, never instead of it.
        # Combining the two estimates was measured and does not help, so
        # the chosen scale is exactly what it was before this existed.
        doubted = self._result(scale_source="doors", scale_agreement=0.9)
        assert doubted.scale == 30.0
        assert doubted.scale_source == "doors"

    def test_the_threshold_sits_out_in_the_tail(self):
        # The estimates usually agree closely -- median disagreement 0.062,
        # 90th percentile 0.207. A threshold below the median would flag
        # half of everything and mean nothing.
        assert 0.10 < MAX_SCALE_DISAGREEMENT < 0.25
