import cv2
import numpy as np
import pytest

from planto3d.ingest import crop_pages, detect_drawing_region

WHITE, BLACK = 255, 0
SHEET_W, SHEET_H = 400, 500
# Template frame: border at the edges, drawing area between, title block below.
FRAME_TOP, FRAME_BOTTOM = 20, 380
FRAME_LEFT, FRAME_RIGHT = 15, 385
TITLE_BLOCK_TOP = 430


def _sheet(feature_rows: list[int]) -> np.ndarray:
    """A template sheet with optional full-width plan features inside the drawing."""
    sheet = np.full((SHEET_H, SHEET_W, 3), WHITE, dtype=np.uint8)

    # Sheet frame — identical on every page.
    for row in (FRAME_TOP, FRAME_BOTTOM):
        sheet[row, FRAME_LEFT:FRAME_RIGHT + 1] = BLACK
    for col in (FRAME_LEFT, FRAME_RIGHT):
        sheet[FRAME_TOP:FRAME_BOTTOM + 1, col] = BLACK

    # Title block below the drawing — also part of the template.
    sheet[TITLE_BLOCK_TOP, FRAME_LEFT:FRAME_RIGHT + 1] = BLACK
    sheet[TITLE_BLOCK_TOP:TITLE_BLOCK_TOP + 30, FRAME_LEFT] = BLACK

    # Plan content that spans the full drawing width, unique to this page.
    for row in feature_rows:
        sheet[row, FRAME_LEFT:FRAME_RIGHT + 1] = BLACK

    return sheet


def test_detect_drawing_region_ignores_page_specific_full_width_features():
    # Page 1 has a plot boundary running just inside the bottom frame line —
    # the case that fooled per-page detection on the real sheets.
    pages = [_sheet([FRAME_BOTTOM - 7]), _sheet([]), _sheet([])]

    left, top, right, bottom = detect_drawing_region(pages)

    assert FRAME_TOP <= top <= FRAME_TOP + 5
    assert FRAME_BOTTOM - 5 <= bottom <= FRAME_BOTTOM
    assert FRAME_LEFT <= left <= FRAME_LEFT + 5
    assert FRAME_RIGHT - 5 <= right <= FRAME_RIGHT


def test_detect_drawing_region_excludes_the_title_block():
    pages = [_sheet([]), _sheet([]), _sheet([])]

    _, _, _, bottom = detect_drawing_region(pages)

    assert bottom < TITLE_BLOCK_TOP


def test_crop_pages_gives_every_page_the_same_frame(tmp_path):
    paths = []
    for i, features in enumerate([[FRAME_BOTTOM - 7], [], [200]], start=1):
        path = tmp_path / f"page-{i}.png"
        cv2.imwrite(str(path), _sheet(features))
        paths.append(path)

    cropped_paths = crop_pages(paths)

    shapes = {cv2.imread(str(p)).shape for p in cropped_paths}
    assert len(shapes) == 1, f"pages cropped to differing frames: {shapes}"
    assert [p.name for p in cropped_paths] == [
        "page-1_cropped.png",
        "page-2_cropped.png",
        "page-3_cropped.png",
    ]


def test_detect_drawing_region_rejects_an_empty_page_set():
    with pytest.raises(ValueError):
        detect_drawing_region([])


def test_detect_drawing_region_rejects_a_blank_sheet():
    blank = np.full((SHEET_H, SHEET_W, 3), WHITE, dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_drawing_region([blank])
