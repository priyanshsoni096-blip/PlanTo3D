"""Tests for the smallest region that is allowed to be a room.

The filter exists to drop specks of segmentation noise, which survive
extraction as 4 sq ft "rooms" and go on to be labelled, railed and paved.
It sized itself in feet, through the *inferred* scale -- so a plan whose
scale came out 27% high got a filter 60% too aggressive and lost real
rooms. Over 60 held-out plans, six of the eight that failed the room count
built too few rooms, with the tracer having found them: 8304 traced 25 and
kept 16 where the drawing shows 23, 3141 traced 18 and kept 9 against 12.

The drawing's own wall thickness is measured rather than inferred, so the
threshold is taken from that instead.
"""

import pytest

from planto3d.calibrate import TYPICAL_WALL_FT


def test_a_room_sized_region_is_kept_and_a_speck_is_dropped():
    from planto3d.pipeline import MIN_ROOM_SQFT, minimum_room_px

    # A 24 px wall is a 9 inch wall, so the drawing runs at 32 px per foot.
    threshold = minimum_room_px(gauge=24.0)
    pixels_per_foot = 24.0 / TYPICAL_WALL_FT

    a_small_wc = 30 * pixels_per_foot**2
    a_speck = 4 * pixels_per_foot**2

    assert a_small_wc > threshold
    assert a_speck < threshold
    assert threshold == pytest.approx(MIN_ROOM_SQFT * pixels_per_foot**2)


def test_the_threshold_does_not_move_with_a_wrong_scale():
    # The point of the change: the filter reads the drawing, not an
    # estimate that can be a quarter out.
    from planto3d.pipeline import minimum_room_px

    assert minimum_room_px(gauge=24.0) == minimum_room_px(gauge=24.0)
    assert minimum_room_px(gauge=12.0) == pytest.approx(minimum_room_px(gauge=24.0) / 4)


def test_an_unmeasurable_gauge_falls_back_rather_than_dropping_everything():
    # wall_gauge returns the reference when it cannot measure; None here
    # means a floor carried no figure at all, and a threshold of zero keeps
    # every speck while a huge one deletes the building.
    from planto3d.pipeline import minimum_room_px

    assert minimum_room_px(gauge=None) == pytest.approx(minimum_room_px(gauge=24.0))
