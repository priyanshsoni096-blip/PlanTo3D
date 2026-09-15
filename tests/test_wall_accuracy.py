"""Tests for scoring extracted walls against the annotation.

Pinned here: a wall built straight through a drawn door or window is not
invented wall. The build cuts the opening back out of it, and scoring it as
invented made every opening a wall is read through count against the plan.
"""

import numpy as np
import pytest

from planto3d.classes import BACKGROUND, DOOR, WALL, WINDOW
from planto3d.geometry_types import Wall


def _plan_with_an_opening(opening):
    truth = np.full((200, 200), BACKGROUND, dtype=np.int64)
    truth[100:106, 10:190] = WALL
    truth[100:106, 70:130] = opening
    return truth


@pytest.mark.parametrize("opening", [WINDOW, DOOR])
def test_a_wall_built_through_a_drawn_opening_is_not_invented(opening):
    from scripts.wall_accuracy import score

    built = [Wall(start=(10.0, 103.0), end=(190.0, 103.0), thickness=6.0)]

    coverage, agreement = score(_plan_with_an_opening(opening), built)

    assert agreement > 0.95
    assert coverage > 0.95


def test_a_wall_where_nothing_is_drawn_is_still_invented():
    from scripts.wall_accuracy import score

    truth = _plan_with_an_opening(WINDOW)
    built = [
        Wall(start=(10.0, 103.0), end=(190.0, 103.0), thickness=6.0),
        Wall(start=(10.0, 30.0), end=(190.0, 30.0), thickness=6.0),
    ]

    _, agreement = score(truth, built)

    assert agreement == pytest.approx(0.5, abs=0.05)


def test_coverage_still_asks_only_about_the_annotated_wall():
    # Counting openings as wall on the built side must not make an opening
    # something the model is required to build a wall over.
    from scripts.wall_accuracy import score

    truth = _plan_with_an_opening(WINDOW)
    built = [Wall(start=(10.0, 103.0), end=(70.0, 103.0), thickness=6.0),
             Wall(start=(130.0, 103.0), end=(190.0, 103.0), thickness=6.0)]

    coverage, agreement = score(truth, built)

    assert coverage > 0.95
    assert agreement > 0.95
