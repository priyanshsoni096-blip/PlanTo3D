import numpy as np
import pytest

from planto3d.classical import ROOM_FILL, WALL_FILL, classical_mask, room_mask, wall_mask
from planto3d.classes import BACKGROUND, ROOM, WALL

BG_FILL = 255


def _plan(size: int = 200) -> np.ndarray:
    """A room enclosed by walls, drawn with the template's grey levels."""
    image = np.full((size, size, 3), BG_FILL, dtype=np.uint8)
    image[20:180, 20:180] = WALL_FILL  # wall block
    image[28:172, 28:172] = ROOM_FILL  # room interior, leaving 8px walls
    return image


def test_wall_mask_finds_the_wall_ring():
    mask = wall_mask(_plan())

    assert mask[24, 100] == 1  # inside the top wall
    assert mask[100, 24] == 1  # inside the left wall
    assert mask[100, 100] == 0  # room interior is not wall
    assert mask[5, 5] == 0  # outside is not wall


def test_room_mask_finds_the_interior():
    mask = room_mask(_plan())

    assert mask[100, 100] == 1
    assert mask[24, 100] == 0
    assert mask[5, 5] == 0


def test_classical_mask_labels_each_class():
    mask = classical_mask(_plan())

    assert mask[100, 100] == ROOM
    assert mask[24, 100] == WALL
    assert mask[5, 5] == BACKGROUND
    assert mask.dtype == np.int64
    assert mask.shape == (200, 200)


def test_classical_mask_output_feeds_the_extractors():
    from planto3d.extract import extract_rooms, extract_walls

    mask = classical_mask(_plan())

    walls = extract_walls(mask)
    rooms = extract_rooms(mask)

    assert len(walls) == 4  # a closed ring decomposes into four segments
    assert len(rooms) == 1


def test_thin_lines_are_not_mistaken_for_walls():
    # Furniture outlines and text are dark but hairline-thin. Only solid
    # bodies are walls, so the cleanup must erase these.
    image = np.full((200, 200, 3), BG_FILL, dtype=np.uint8)
    image[100, 20:180] = 0  # a one-pixel black rule
    image[20:180, 100] = 30  # another, slightly lighter

    assert wall_mask(image).sum() == 0


def test_greyscale_input_is_accepted():
    grey = np.full((100, 100), BG_FILL, dtype=np.uint8)
    grey[20:80, 20:80] = WALL_FILL

    assert wall_mask(grey).any()


def test_rooms_split_by_a_hairline_door_line_stay_separate():
    # Two rooms joined by a doorway, with the door leaf drawn across the
    # opening as a single-pixel line. That line is all that separates them,
    # so any smoothing of the room mask merges the rooms -- which is exactly
    # what happened on the real ground floor, collapsing 15 rooms into 1.
    image = np.full((200, 200, 3), BG_FILL, dtype=np.uint8)
    image[20:180, 20:180] = WALL_FILL
    image[28:96, 28:172] = ROOM_FILL  # upper room
    image[104:172, 28:172] = ROOM_FILL  # lower room
    image[96:104, 80:120] = ROOM_FILL  # doorway joining them
    image[100, 80:120] = 0  # door leaf across the opening

    import cv2

    count, _, stats, _ = cv2.connectedComponentsWithStats(room_mask(image), connectivity=4)
    big = [a for a in stats[1:, cv2.CC_STAT_AREA] if a >= 500]

    assert len(big) == 2


def test_a_blank_page_yields_all_background():
    blank = np.full((100, 100, 3), BG_FILL, dtype=np.uint8)

    mask = classical_mask(blank)

    assert (mask == BACKGROUND).all()


@pytest.mark.parametrize("fill", [WALL_FILL - 6, WALL_FILL, WALL_FILL + 6])
def test_wall_detection_tolerates_small_shading_variation(fill):
    image = np.full((200, 200, 3), BG_FILL, dtype=np.uint8)
    image[20:180, 20:60] = fill

    assert wall_mask(image).sum() > 0
