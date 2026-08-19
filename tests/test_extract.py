import numpy as np
import pytest

from planto3d.classes import BACKGROUND, BEDROOM, KITCHEN, OUTDOOR, ROOM, WALL
from planto3d.extract import (
    MAX_THICKNESS_RATIO,
    _drop_impossibly_thick,
    _typical_thickness,
    extract_rooms,
    extract_walls,
)
from planto3d.geometry_types import Wall


def _blank(size: int = 60) -> np.ndarray:
    return np.full((size, size), BACKGROUND, dtype=np.int64)


def _horizontal(mask, y, x0, x1, thickness=3):
    mask[y : y + thickness, x0:x1] = WALL


def _vertical(mask, x, y0, y1, thickness=3):
    mask[y0:y1, x : x + thickness] = WALL


def _axis(wall) -> str:
    dx = abs(wall.end[0] - wall.start[0])
    dy = abs(wall.end[1] - wall.start[1])
    return "horizontal" if dx >= dy else "vertical"


def test_extract_walls_finds_a_single_horizontal_wall():
    mask = _blank()
    _horizontal(mask, y=20, x0=10, x1=50, thickness=3)

    walls = extract_walls(mask)

    assert len(walls) == 1
    wall = walls[0]
    assert _axis(wall) == "horizontal"
    assert wall.length() == pytest.approx(40, abs=2)
    assert wall.thickness == pytest.approx(3, abs=1)


def test_extract_walls_splits_a_closed_loop_into_separate_segments():
    # Four walls meeting at corners form one connected region. Contour
    # detection alone returns the ring as a single shape, so the extractor
    # has to decompose it by orientation.
    mask = _blank()
    _horizontal(mask, y=10, x0=10, x1=50)
    _horizontal(mask, y=47, x0=10, x1=50)
    _vertical(mask, x=10, y0=10, y1=50)
    _vertical(mask, x=47, y0=10, y1=50)

    walls = extract_walls(mask)

    assert len(walls) == 4
    orientations = sorted(_axis(w) for w in walls)
    assert orientations == ["horizontal", "horizontal", "vertical", "vertical"]
    for wall in walls:
        assert wall.length() == pytest.approx(40, abs=4)


def test_extract_walls_ignores_speckle_noise():
    mask = _blank()
    _horizontal(mask, y=20, x0=10, x1=50)
    mask[40, 5] = WALL
    mask[45, 30] = WALL
    mask[50:52, 20:22] = WALL

    walls = extract_walls(mask)

    assert len(walls) == 1


def test_extract_walls_returns_nothing_for_a_mask_with_no_walls():
    assert extract_walls(_blank()) == []


def test_extract_rooms_finds_one_polygon_per_enclosed_region():
    mask = _blank()
    mask[10:30, 10:30] = ROOM
    mask[35:55, 35:55] = ROOM

    rooms = extract_rooms(mask)

    assert len(rooms) == 2
    for room in rooms:
        assert room.label == ""


def test_extract_rooms_simplifies_a_rectangle_to_its_corners():
    # A raw contour traces every boundary pixel; a rectangle should reduce to
    # roughly four vertices so downstream geometry stays manageable.
    mask = _blank()
    mask[10:50, 10:50] = ROOM

    rooms = extract_rooms(mask)

    assert len(rooms) == 1
    assert len(rooms[0].polygon) <= 6


def test_extract_rooms_ignores_regions_below_the_area_floor():
    mask = _blank()
    mask[10:40, 10:40] = ROOM
    mask[50:53, 50:53] = ROOM

    rooms = extract_rooms(mask)

    assert len(rooms) == 1


def test_extract_rooms_returns_nothing_for_a_mask_with_no_rooms():
    assert extract_rooms(_blank()) == []


class TestRoomTypesComeThroughExtraction:
    def _mask(self):
        mask = np.zeros((200, 200), dtype=np.int64)
        mask[10:90, 10:90] = ROOM
        mask[10:90, 110:190] = BEDROOM
        mask[110:190, 10:90] = KITCHEN
        mask[110:190, 110:190] = OUTDOOR
        return mask

    def test_each_room_carries_the_class_it_came_from(self):
        rooms = extract_rooms(self._mask())

        assert sorted(room.category for room in rooms) == [
            "",
            "bedroom",
            "kitchen",
            "outdoor",
        ]

    def test_the_generic_class_stays_uncategorised(self):
        # "room" as a category would be indistinguishable from a positive
        # identification, and the pipeline needs to know the model had no
        # opinion so a printed name can supply one.
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:90, 10:90] = ROOM

        assert extract_rooms(mask)[0].category == ""

    def test_adjoining_rooms_of_different_types_stay_separate(self):
        # An open kitchen has no wall between it and the dining area it
        # opens onto. Tracing both classes together would return one room
        # and one floor finish for what are plainly two spaces.
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[10:90, 10:100] = ROOM
        mask[10:90, 100:190] = KITCHEN

        rooms = extract_rooms(mask)

        assert len(rooms) == 2
        assert {room.category for room in rooms} == {"", "kitchen"}

    def test_a_single_class_can_still_be_asked_for(self):
        rooms = extract_rooms(self._mask(), room_class=KITCHEN)

        assert len(rooms) == 1
        assert rooms[0].category == "kitchen"


class TestRunsTooThickToBeWalls:
    """Boundary hatching and dimension bands segment as enormous walls.

    On the reference sheet the median wall run measures 10 inches and the
    fattest nearly ten feet. Left in, that becomes a solid slab across the
    plan -- and worse, it drags the wall-thickness scale estimate with it,
    so the whole building is built the wrong size.

    Tested on the rule directly rather than through a rasterized mask. A
    mask of bands also yields runs the fixture did not intend, and a test
    that trips over those is measuring the fixture.
    """

    def _walls(self, thicknesses, length=400.0):
        return [
            Wall(start=(0.0, index * 50.0), end=(length, index * 50.0), thickness=float(t))
            for index, t in enumerate(thicknesses)
        ]

    def _kept(self, thicknesses, **kwargs):
        walls = self._walls(thicknesses, **kwargs)
        return sorted(round(wall.thickness) for wall in _drop_impossibly_thick(walls))

    def test_a_band_far_thicker_than_the_walls_is_dropped(self):
        assert self._kept([10] * 8 + [120]) == [10] * 8

    def test_ordinary_walls_all_survive(self):
        # Real walls span a narrow range and every one has to be kept: a 4
        # inch partition beside an 18 inch external wall.
        bands = [8, 10, 12, 16, 20, 24]

        assert self._kept(bands) == sorted(bands)

    def test_nothing_is_judged_on_too_few_runs(self):
        # A handful of runs could easily be mostly hatching, and taking
        # their reference would discard the walls and keep the hatching.
        assert 200 in self._kept([10, 10, 200])

    def test_a_drawing_of_thick_walls_is_left_alone(self):
        # The limit moves with the drawing, so a plan drawn at a larger
        # scale is not gutted for having thicker lines.
        bands = [40, 44, 48, 52, 56, 60, 64]

        assert self._kept(bands) == sorted(bands)

    def test_the_walls_are_kept_when_too_much_would_go(self):
        # Beyond a small fraction the reference was measuring the wrong
        # thing. Allowing half to go once left a drawing with a two pixel
        # median wall, putting the building at a thirtieth of its size.
        kept = self._kept([6, 6, 90, 90, 90, 90, 90, 90])

        assert kept == [6, 6, 90, 90, 90, 90, 90, 90]

    def test_the_reference_follows_the_longest_walls_not_the_most_numerous(self):
        # Short specks of noise outnumbering the walls must not set the
        # thickness: that is what took a real drawing down to 3 px/ft.
        walls = self._walls([6] * 10, length=8.0) + self._walls([24] * 4, length=900.0)

        assert _typical_thickness(walls) == pytest.approx(24.0)

    def test_it_runs_through_extraction(self):
        # And the rule is actually wired into extract_walls, not merely
        # available beside it.
        # As many walls as a real storey has. A single thick band is found
        # by both the horizontal and the vertical pass, so it costs two
        # drops, and on a drawing of only a few runs that is a large enough
        # share to trip the guard against over-eager removal.
        mask = np.full((1200, 1200), BACKGROUND, dtype=np.int64)
        row = 30
        for index, thickness in enumerate([10] * 16 + [200]):
            left = 30 + index * 8
            mask[row : row + thickness, left : left + 420] = WALL
            row += thickness + 30

        assert max(wall.thickness for wall in extract_walls(mask, merge=False)) < 200
