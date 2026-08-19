import numpy as np
import pytest

from planto3d.classes import BACKGROUND, WALL
from planto3d.extract import extract_walls


# Fixture walls are drawn at the gauge the thresholds are expressed
# against, so the gaps below mean what they say: a 24 pixel wall is about
# nine inches and a 60 pixel gap about a doorway. Drawn thinner, the same
# gaps would be five feet across and rightly refuse to merge -- the sizes
# in this module follow the drawing's own wall thickness now, not the
# resolution the tests were first written at.
WALL_PX = 24


def _blank(size=400):
    return np.full((size, size), BACKGROUND, dtype=np.int64)


def _horizontal(walls):
    return [w for w in walls if abs(w.end[0] - w.start[0]) > abs(w.end[1] - w.start[1])]


def _vertical(walls):
    return [w for w in walls if abs(w.end[1] - w.start[1]) > abs(w.end[0] - w.start[0])]


class TestMergingCollinearRuns:
    def test_a_wall_split_by_a_doorway_becomes_one_run(self):
        # One continuous wall interrupted by a door. Left apart, the two
        # stubs extrude as abutting boxes with a seam at the joint, and an
        # opening measured along a stub sits at the wrong distance.
        mask = _blank()
        mask[100:124, 40:180] = WALL
        mask[100:124, 240:360] = WALL  # same line, doorway-sized gap

        walls = _horizontal(extract_walls(mask))

        assert len(walls) == 1
        assert min(walls[0].start[0], walls[0].end[0]) == pytest.approx(40, abs=4)
        assert max(walls[0].start[0], walls[0].end[0]) == pytest.approx(359, abs=4)

    def test_walls_on_different_lines_stay_separate(self):
        mask = _blank()
        mask[100:124, 40:360] = WALL
        mask[300:324, 40:360] = WALL

        assert len(_horizontal(extract_walls(mask))) == 2

    def test_a_wide_gap_is_not_bridged(self):
        # Two genuinely separate walls on one line, far apart, must not be
        # joined into a single run spanning the space between them.
        mask = _blank()
        mask[100:124, 20:80] = WALL
        mask[100:124, 320:380] = WALL

        assert len(_horizontal(extract_walls(mask))) == 2

    def test_vertical_runs_merge_too(self):
        mask = _blank()
        mask[40:180, 100:124] = WALL
        mask[240:360, 100:124] = WALL

        assert len(_vertical(extract_walls(mask))) == 1

    def test_merging_reduces_fragments_without_losing_extent(self):
        mask = _blank()
        for start in range(40, 340, 60):
            mask[100:124, start : start + 40] = WALL

        unmerged = _horizontal(extract_walls(mask, merge=False))
        merged = _horizontal(extract_walls(mask, merge=True))

        assert len(merged) < len(unmerged)
        span = lambda walls: (
            min(min(w.start[0], w.end[0]) for w in walls),
            max(max(w.start[0], w.end[0]) for w in walls),
        )
        assert span(merged) == pytest.approx(span(unmerged), abs=4)

    def test_merging_keeps_the_thicker_wall_thickness(self):
        # Both drawn around the gauge, so the doorway between them is a
        # doorway. A thin partition meeting a thick external wall is the
        # ordinary case; the merged run has to keep the thicker figure or
        # the wall is built too light where it carries most.
        mask = _blank()
        mask[100:118, 40:180] = WALL  # 18px thick
        mask[100:130, 240:360] = WALL  # 30px thick, same line

        walls = _horizontal(extract_walls(mask))

        assert len(walls) == 1
        assert walls[0].thickness >= 29

    def test_merging_can_be_turned_off(self):
        mask = _blank()
        mask[100:124, 40:180] = WALL
        mask[100:124, 240:360] = WALL

        assert len(_horizontal(extract_walls(mask, merge=False))) == 2

    def test_a_single_wall_is_unaffected(self):
        mask = _blank()
        mask[100:124, 40:360] = WALL

        walls = extract_walls(mask)

        assert len(_horizontal(walls)) == 1

    def test_a_corner_keeps_both_of_its_walls(self):
        # Perpendicular runs meeting at a corner must not collapse together.
        mask = _blank()
        mask[100:124, 100:300] = WALL
        mask[100:300, 100:124] = WALL

        walls = extract_walls(mask)

        assert len(_horizontal(walls)) == 1
        assert len(_vertical(walls)) == 1

    def test_an_empty_mask_still_returns_nothing(self):
        assert extract_walls(_blank()) == []
