"""Cleanup that keeps the model looking like a building rather than a blob."""

import numpy as np
import pytest

from planto3d.classes import BACKGROUND, ROOM, WALL
from planto3d.extract import extract_footprint, extract_walls


def _bounds(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


class TestFootprintCleanup:
    # Walls at the gauge the cleanup spans are expressed against, so a
    # "thin spur" below is thinner than a wall and reads as the tendril it
    # is meant to be. Drawn at 8 pixels, the same 10 pixel spur was wider
    # than the building's own walls -- the cleanup is sized by the drawing
    # now, and rightly kept it.
    WALL_PX = 24

    def _building(self, size=400):
        mask = np.full((size, size), BACKGROUND, dtype=np.int64)
        mask[100:300, 100:300] = ROOM
        mask[100 - self.WALL_PX : 100, 100:300] = WALL
        mask[300 : 300 + self.WALL_PX, 100:300] = WALL
        return mask

    def test_a_thin_spur_does_not_reach_into_the_outline(self):
        # Segmentation bleeds along paving and planting beside the building.
        # Those tendrils used to end up in the footprint, giving slabs jagged
        # fingers reaching out over open ground.
        mask = self._building()
        mask[195:205, 300:390] = ROOM  # a 10px-wide spike out to the right

        _, _, right, _ = _bounds(extract_footprint(mask))

        assert right < 330

    def test_a_detached_patch_cannot_drag_the_outline_across_the_site(self):
        mask = self._building()
        # Clear of the building, including its walls.
        patch = 300 + self.WALL_PX + 30
        mask[patch : patch + 40, patch : patch + 40] = ROOM

        _, _, right, bottom = _bounds(extract_footprint(mask))

        assert right < patch
        assert bottom < patch

    def test_the_building_itself_survives_the_cleanup(self):
        # Stated against the fixture rather than as fixed numbers, so
        # redrawing it at another wall thickness cannot silently make this
        # test about something else.
        left, top, right, bottom = _bounds(extract_footprint(self._building()))

        assert left == pytest.approx(100, abs=15)
        assert right == pytest.approx(300, abs=15)
        assert bottom - top == pytest.approx(200 + 2 * self.WALL_PX, abs=25)

    def test_an_empty_mask_yields_no_footprint(self):
        assert extract_footprint(np.full((100, 100), BACKGROUND, dtype=np.int64)) == []


class TestWallsAreSquare:
    def _tilted_wall(self, drop: int, size=300):
        """A wall running left to right that drifts `drop` pixels downward."""
        mask = np.full((size, size), BACKGROUND, dtype=np.int64)
        for x in range(40, 260):
            y = 150 + int(drop * (x - 40) / 220)
            mask[y - 3 : y + 4, x] = WALL
        return mask

    def test_a_noisy_wall_edge_still_yields_an_axis_aligned_wall(self):
        # Endpoints sit on each run's bounding-box centreline, so a ragged or
        # drifting edge cannot tilt the extruded masonry.
        walls = extract_walls(self._tilted_wall(drop=6))

        assert walls
        for wall in walls:
            horizontal = abs(wall.end[1] - wall.start[1]) < 0.01
            vertical = abs(wall.end[0] - wall.start[0]) < 0.01
            assert horizontal or vertical

    def test_walls_from_the_real_extractor_are_never_skewed(self):
        mask = np.full((300, 300), BACKGROUND, dtype=np.int64)
        mask[100:106, 40:260] = WALL
        mask[40:260, 200:206] = WALL

        for wall in extract_walls(mask):
            horizontal = abs(wall.end[1] - wall.start[1]) < 0.01
            vertical = abs(wall.end[0] - wall.start[0]) < 0.01
            assert horizontal or vertical

    def test_a_diagonal_wall_is_not_recovered(self):
        # A documented limitation, pinned so it cannot change unnoticed: the
        # orientation filters erase diagonal runs entirely.
        mask = np.full((300, 300), BACKGROUND, dtype=np.int64)
        for step in range(200):
            mask[50 + step - 2 : 50 + step + 3, 50 + step] = WALL

        assert extract_walls(mask) == []
