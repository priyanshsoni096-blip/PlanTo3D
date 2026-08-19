import numpy as np
import pytest

from planto3d.classes import BACKGROUND, BEDROOM, KITCHEN, OUTDOOR, ROOM, WALL
from planto3d.extract import extract_rooms, extract_walls


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
