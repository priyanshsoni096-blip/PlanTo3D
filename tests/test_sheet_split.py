"""Forcing a split on a sheet the detector reads wrongly.

Measured against CubiCasa's annotations, automatic splitting is right on
58 of 60 sheets and never splits a single plan wrongly. The two it misses
are a sheet whose thin floor the acceptance gates reject, and two units
sharing a party wall with no gutter at all. The first is recoverable by
using the cut that was already proposed; the second is not recoverable
without inventing a divide, so it must fail loudly rather than guess.
"""

import numpy as np
import pytest

from planto3d.ingest import split_sheet


def _two_plans_side_by_side() -> np.ndarray:
    """A white sheet with two ink blocks and a wide blank gutter between."""
    sheet = np.full((600, 1200, 3), 255, dtype=np.uint8)
    # Each "plan" is a hollow rectangle, so it encloses area like a real one.
    for left in (60, 700):
        sheet[80:520, left : left + 440] = 0
        sheet[120:480, left + 40 : left + 400] = 255
    return sheet


def _one_plan() -> np.ndarray:
    sheet = np.full((600, 1200, 3), 255, dtype=np.uint8)
    sheet[80:520, 100:1100] = 0
    sheet[120:480, 140:1060] = 255
    return sheet


def test_force_none_keeps_todays_behaviour():
    assert len(split_sheet(_two_plans_side_by_side(), force=None)) == 2


def test_force_one_keeps_the_sheet_whole():
    # The escape hatch for a sheet that is one plan but splits wrongly.
    assert len(split_sheet(_two_plans_side_by_side(), force=1)) == 1


def test_force_two_splits_a_sheet_the_gates_would_reject():
    # A thin strip of a second plan: real, but too little ink to pass
    # MIN_PLAN_INK_SHARE. Automatic reading leaves it whole; forcing splits it.
    sheet = np.full((600, 1200, 3), 255, dtype=np.uint8)
    sheet[80:520, 60:900] = 0
    sheet[120:480, 100:860] = 255
    sheet[300:340, 1000:1160] = 0  # the thin strip, past a wide gutter
    assert len(split_sheet(sheet, force=None)) == 1
    assert len(split_sheet(sheet, force=2)) == 2


def test_forcing_a_split_with_no_cut_available_says_so():
    # The party-wall case. There is no gutter, so there is nothing to force,
    # and guessing a divide would cut a real plan in half.
    with pytest.raises(ValueError, match="no dividing line"):
        split_sheet(_one_plan(), force=2)


def test_forcing_more_pieces_than_there_are_cuts_says_so():
    with pytest.raises(ValueError, match="no dividing line"):
        split_sheet(_two_plans_side_by_side(), force=5)
