"""What the pipeline does when the page holds no drawing.

Found by handing it the files a real user will: a blank page, a black
scan, a one-pixel image, and a text file renamed .png. Each came back
claiming a model had been built, with no walls, no rooms, and a scale of
exactly 32.0 px/ft attributed to "walls" -- the assumed figure, reported
as though the drawing's own walls had been measured.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from planto3d.classes import BACKGROUND, ROOM, WALL


def _blank_page(tmp_path, name="blank.png", colour=255):
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((600, 800, 3), colour, np.uint8))
    return path


def _nothing_segmenter(image):
    return np.full(image.shape[:2], BACKGROUND, dtype=np.int64)


def test_a_page_with_no_drawing_on_it_is_refused(tmp_path):
    from planto3d.pipeline import run

    with tempfile.TemporaryDirectory() as workdir:
        with pytest.raises(ValueError, match="no walls or rooms"):
            run(_blank_page(tmp_path), Path(workdir), segmenter=_nothing_segmenter)


def test_the_refusal_names_the_file(tmp_path):
    from planto3d.pipeline import run

    page = _blank_page(tmp_path, name="holiday-snap.png")
    with tempfile.TemporaryDirectory() as workdir:
        with pytest.raises(ValueError, match="holiday-snap.png"):
            run(page, Path(workdir), segmenter=_nothing_segmenter)


def test_a_drawing_with_rooms_but_no_walls_still_builds(tmp_path):
    # Refusing must mean "nothing here", not "nothing I recognise as a
    # wall": a sheet whose partitions are drawn too faint to segment still
    # has rooms worth building.
    from planto3d.pipeline import run

    def rooms_only(image):
        mask = np.full(image.shape[:2], BACKGROUND, dtype=np.int64)
        mask[100:300, 100:400] = ROOM
        mask[350:550, 100:400] = ROOM
        return mask

    with tempfile.TemporaryDirectory() as workdir:
        result = run(_blank_page(tmp_path), Path(workdir), segmenter=rooms_only)

    assert result.room_count >= 1


def test_a_scale_nobody_measured_is_not_credited_to_the_walls(tmp_path):
    # wall_gauge falls back to the reference 24 px when it cannot measure,
    # and 24 px over a 9 inch wall is exactly the 32.0 px/ft the assumed
    # ratio gives -- reported as "walls", which claims evidence that does
    # not exist.
    from planto3d.pipeline import run

    def rooms_only(image):
        mask = np.full(image.shape[:2], BACKGROUND, dtype=np.int64)
        mask[100:300, 100:400] = ROOM
        return mask

    with tempfile.TemporaryDirectory() as workdir:
        result = run(_blank_page(tmp_path), Path(workdir), segmenter=rooms_only)

    assert result.scale_source != "walls"
    assert result.scale_assumed


def test_a_wall_that_was_measured_is_still_credited(tmp_path):
    from planto3d.pipeline import run

    def one_room_with_walls(image):
        mask = np.full(image.shape[:2], BACKGROUND, dtype=np.int64)
        mask[100:400, 100:500] = ROOM
        mask[100:112, 100:500] = WALL
        mask[388:400, 100:500] = WALL
        mask[100:400, 100:112] = WALL
        mask[100:400, 488:500] = WALL
        return mask

    with tempfile.TemporaryDirectory() as workdir:
        result = run(_blank_page(tmp_path), Path(workdir), segmenter=one_room_with_walls)

    assert result.scale_source == "walls"


def test_a_file_that_is_not_an_image_says_so(tmp_path):
    # It raised FileNotFoundError: "could not read page image", which sends
    # the reader looking for a missing file that is sitting right there.
    from planto3d.pipeline import run

    notes = tmp_path / "notes.png"
    notes.write_text("this is not a png")

    with tempfile.TemporaryDirectory() as workdir:
        with pytest.raises(ValueError, match="not an image"):
            run(notes, Path(workdir), segmenter=_nothing_segmenter)
