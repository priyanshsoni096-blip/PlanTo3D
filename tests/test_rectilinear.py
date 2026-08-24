"""Rooms are drawn with square corners, so they are built with square corners.

Measured over twelve CubiCasa plans, the annotated rooms run 98.5% square
on four vertices while the traced ones ran 72.9% on ten. A quarter of every
room's perimeter was diagonal, and that is what made a finished model read
as approximate rather than as a building.

Walls have been forced axis-aligned since the beginning, for the same
reason. This is that rule applied to rooms.
"""

import math

import numpy as np
import pytest

from planto3d.classes import BACKGROUND, ROOM
from planto3d.extract import _rectilinear, extract_rooms


def square_share(polygon, tolerance=8.0) -> float:
    """Share of the perimeter running horizontal or vertical."""
    points = np.asarray(polygon, dtype=float)
    total = square = 0.0
    for index in range(len(points)):
        start, end = points[index], points[(index + 1) % len(points)]
        delta = end - start
        length = float(np.hypot(*delta))
        if not length:
            continue
        angle = math.degrees(math.atan2(abs(delta[1]), abs(delta[0])))
        total += length
        if angle <= tolerance or angle >= 90 - tolerance:
            square += length
    return square / total if total else 1.0


def area(polygon) -> float:
    points = np.asarray(polygon, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class TestSquaringACorner:
    def test_a_ragged_rectangle_comes_back_square(self):
        ragged = [(0, 0), (98, 3), (201, 1), (199, 97), (202, 202), (3, 198), (1, 101)]

        assert square_share(_rectilinear(ragged)) == pytest.approx(1.0)

    def test_a_clean_rectangle_is_left_alone(self):
        clean = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]

        assert area(_rectilinear(clean)) == pytest.approx(area(clean), rel=0.01)

    def test_the_room_keeps_its_size(self):
        # Squaring must not quietly shrink or inflate a room; the model is
        # measured in feet and a room's area is the thing being measured.
        ragged = [(0, 0), (98, 3), (201, 1), (199, 97), (202, 202), (3, 198), (1, 101)]

        assert area(_rectilinear(ragged)) == pytest.approx(area(ragged), rel=0.15)

    def test_an_l_shape_stays_an_l_shape(self):
        # Six corners, not four. Squaring is not the same as taking a
        # bounding box, and plenty of rooms are L-shaped.
        shape = [(0, 0), (200, 2), (198, 100), (100, 98), (102, 200), (2, 199)]

        squared = _rectilinear(shape)

        assert len(squared) >= 6
        assert area(squared) == pytest.approx(area(shape), rel=0.15)

    def test_a_diagonal_room_is_left_as_traced(self):
        # Squaring a genuinely diagonal room would move its area a long
        # way, and a wrong room is worse than a rough one.
        diamond = [(100, 0), (200, 100), (100, 200), (0, 100)]

        assert _rectilinear(diamond) == diamond

    def test_too_few_corners_to_square(self):
        triangle = [(0.0, 0.0), (100.0, 0.0), (50.0, 80.0)]

        assert _rectilinear(triangle) == triangle


class TestThroughExtraction:
    def test_rooms_come_out_of_the_pipeline_square(self):
        # A room with a deliberately ragged edge, as segmentation produces.
        mask = np.full((300, 300), BACKGROUND, dtype=np.int64)
        mask[60:240, 60:240] = ROOM
        for offset, row in enumerate(range(60, 240, 12)):
            mask[row : row + 6, 60 + (offset % 3) : 63 + (offset % 3)] = BACKGROUND

        rooms = extract_rooms(mask)

        assert rooms
        assert square_share(rooms[0].polygon) > 0.95

    def test_a_plain_room_is_still_four_corners(self):
        mask = np.full((300, 300), BACKGROUND, dtype=np.int64)
        mask[60:240, 40:260] = ROOM

        polygon = extract_rooms(mask)[0].polygon

        assert len(polygon) == 4
        assert square_share(polygon) == pytest.approx(1.0)
