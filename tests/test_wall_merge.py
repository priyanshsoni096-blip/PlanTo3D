import numpy as np
import pytest

from planto3d.classes import BACKGROUND, WALL
from planto3d.extract import extract_walls


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
        mask[100:108, 40:180] = WALL
        mask[100:108, 240:360] = WALL  # same line, doorway-sized gap

        walls = _horizontal(extract_walls(mask))

        assert len(walls) == 1
        assert min(walls[0].start[0], walls[0].end[0]) == pytest.approx(40, abs=4)
        assert max(walls[0].start[0], walls[0].end[0]) == pytest.approx(359, abs=4)

    def test_walls_on_different_lines_stay_separate(self):
        mask = _blank()
        mask[100:108, 40:360] = WALL
        mask[300:308, 40:360] = WALL

        assert len(_horizontal(extract_walls(mask))) == 2

    def test_a_wide_gap_is_not_bridged(self):
        # Two genuinely separate walls on one line, far apart, must not be
        # joined into a single run spanning the space between them.
        mask = _blank()
        mask[100:108, 20:80] = WALL
        mask[100:108, 320:380] = WALL

        assert len(_horizontal(extract_walls(mask))) == 2

    def test_vertical_runs_merge_too(self):
        mask = _blank()
        mask[40:180, 100:108] = WALL
        mask[240:360, 100:108] = WALL

        assert len(_vertical(extract_walls(mask))) == 1

    def test_merging_reduces_fragments_without_losing_extent(self):
        mask = _blank()
        for start in range(40, 340, 60):
            mask[100:108, start : start + 40] = WALL

        unmerged = _horizontal(extract_walls(mask, merge=False))
        merged = _horizontal(extract_walls(mask, merge=True))

        assert len(merged) < len(unmerged)
        span = lambda walls: (
            min(min(w.start[0], w.end[0]) for w in walls),
            max(max(w.start[0], w.end[0]) for w in walls),
        )
        assert span(merged) == pytest.approx(span(unmerged), abs=4)

    def test_merging_keeps_the_thicker_wall_thickness(self):
        mask = _blank()
        mask[100:106, 40:180] = WALL  # 6px thick
        mask[100:114, 240:360] = WALL  # 14px thick, same line

        walls = _horizontal(extract_walls(mask))

        assert len(walls) == 1
        assert walls[0].thickness >= 13

    def test_merging_can_be_turned_off(self):
        mask = _blank()
        mask[100:108, 40:180] = WALL
        mask[100:108, 240:360] = WALL

        assert len(_horizontal(extract_walls(mask, merge=False))) == 2

    def test_a_single_wall_is_unaffected(self):
        mask = _blank()
        mask[100:108, 40:360] = WALL

        walls = extract_walls(mask)

        assert len(_horizontal(walls)) == 1

    def test_a_corner_keeps_both_of_its_walls(self):
        # Perpendicular runs meeting at a corner must not collapse together.
        mask = _blank()
        mask[100:108, 100:300] = WALL
        mask[100:300, 100:108] = WALL

        walls = extract_walls(mask)

        assert len(_horizontal(walls)) == 1
        assert len(_vertical(walls)) == 1

    def test_an_empty_mask_still_returns_nothing(self):
        assert extract_walls(_blank()) == []
