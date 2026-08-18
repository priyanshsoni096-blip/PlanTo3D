"""Window frames and mullions, which make glazing legible on a facade."""

import pytest

from planto3d.extrude import MAX_PANE_WIDTH_M, floors_to_parts, opening_frames
from planto3d.geometry_types import FloorPlan, Opening, Wall

SCALE = 20.0
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
LONG_WALL = Wall(start=(0.0, 0.0), end=(600.0, 0.0), thickness=10.0)


def _window(width, position=300.0):
    return Opening(wall_id=0, position=position, width=width, type="window")


def _frames(openings, height_ft=9.0):
    return opening_frames(
        LONG_WALL, openings, height_ft * 0.3048, SCALE, base_m=0.0
    )


class TestFrames:
    def test_a_window_gets_a_frame(self):
        assert _frames([_window(80.0)])

    def test_a_door_gets_no_frame(self):
        # Doorways are left open, so there is nothing to frame.
        door = Opening(wall_id=0, position=300.0, width=80.0, type="door")

        assert _frames([door]) == []

    def test_a_wide_window_is_divided_by_mullions(self):
        # Glass is not made in arbitrary widths, and an undivided expanse
        # reads as a hole rather than a window.
        narrow = _frames([_window(60.0)])
        wide = _frames([_window(600.0)])

        assert len(wide) > len(narrow)

    def test_mullion_count_follows_the_opening_width(self):
        # Roughly one division per pane's worth of width.
        span_m = 600.0 / SCALE * 0.3048
        expected = max(int(span_m / MAX_PANE_WIDTH_M), 1)

        members = _frames([_window(600.0)])
        # Two horizontal members plus one vertical per division boundary.
        assert len(members) == 2 + expected + 1

    def test_frames_sit_within_the_opening_not_the_wall(self):
        members = _frames([_window(80.0, position=300.0)])

        left = min(m.bounds[0][0] for m in members)
        right = max(m.bounds[1][0] for m in members)
        centre = 300.0 / SCALE * 0.3048
        half = (80.0 / SCALE * 0.3048) / 2

        assert left == pytest.approx(centre - half, abs=0.15)
        assert right == pytest.approx(centre + half, abs=0.15)

    def test_a_sliver_of_an_opening_is_skipped(self):
        assert _frames([_window(1.0)]) == []

    def test_frames_reach_no_higher_than_the_opening_head(self):
        members = _frames([_window(80.0)])
        head_m = 7.0 * 0.3048

        assert max(m.bounds[1][1] for m in members) <= head_m + 0.01


class TestFramesInTheModel:
    def _floor(self, openings):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        return FloorPlan(walls=walls, footprint=list(FOOTPRINT), openings=openings)

    def test_glazed_openings_produce_frame_geometry(self):
        parts = floors_to_parts(
            [self._floor([_window(120.0, position=200.0)])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert "frame" in parts
        assert "glass" in parts

    def test_a_plan_without_windows_has_no_frames(self):
        parts = floors_to_parts([self._floor([])], wall_height_ft=9.0, scale=SCALE)

        assert "frame" not in parts
