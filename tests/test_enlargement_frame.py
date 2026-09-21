"""Tests for keeping an enlarged drawing's geometry in one frame.

A drawing whose walls are too thin to measure is enlarged before it is read,
so its walls and rooms come out in the enlarged frame. Two things kept using
the page as it was on disk: the overlay drew the geometry over the original,
smaller image, and the plot size was taken from the original page while the
scale was measured on the enlarged one. On the reference house, read by a
checkpoint that sees its walls 8 px thick, the page was enlarged 1.25x: the
overlay ran off the right and bottom edges, and the plot came out 1.25x too
small for the building standing on it.
"""

import cv2
import numpy as np

from planto3d.classes import BACKGROUND, WALL
from planto3d.geometry_types import FloorPlan, Wall


def _grid_segmenter(thickness):
    """A segmenter whose walls are ``thickness`` px on any image it is given."""

    def segment(image):
        mask = np.full(image.shape[:2], BACKGROUND, dtype=np.int64)
        for start in range(0, mask.shape[0], 40):
            mask[start : start + thickness, :] = WALL
        for start in range(0, mask.shape[1], 40):
            mask[:, start : start + thickness] = WALL
        return mask

    return segment


def test_a_drawing_that_is_fine_is_not_enlarged():
    from planto3d.pipeline import _enlarge_if_unmeasurable

    image = np.full((200, 300, 3), 255, np.uint8)
    segment = _grid_segmenter(16)

    _, _, factor = _enlarge_if_unmeasurable(image, segment(image), segment)

    assert factor == 1.0


def test_an_enlarged_drawing_reports_how_much():
    from planto3d.pipeline import _enlarge_if_unmeasurable

    image = np.full((200, 300, 3), 255, np.uint8)
    segment = _grid_segmenter(4)

    enlarged, mask, factor = _enlarge_if_unmeasurable(image, segment(image), segment)

    assert factor > 1.0
    assert enlarged.shape[1] == round(300 * factor)
    assert mask.shape == enlarged.shape[:2]


def _floor(tmp_path, enlargement):
    from planto3d.pipeline import FloorResult

    page = tmp_path / "page.png"
    cv2.imwrite(str(page), np.full((40, 50, 3), 255, np.uint8))
    return FloorResult(
        index=0,
        image_path=page,
        plan=FloorPlan(walls=[Wall(start=(10.0, 70.0), end=(90.0, 70.0), thickness=4.0)]),
        enlargement=enlargement,
    )


def test_the_overlay_is_drawn_in_the_frame_the_geometry_was_read_in(tmp_path):
    from planto3d.pipeline import draw_overlay

    overlay = draw_overlay(_floor(tmp_path, enlargement=2.0))

    # The page is 50 x 40; read at twice the size, so is its geometry.
    assert overlay.shape[:2] == (80, 100)
    # The wall at y=70 is on the canvas rather than off its bottom edge.
    assert (overlay[70, 50] != 255).any()


def test_an_unenlarged_overlay_is_the_page_as_it_was(tmp_path):
    from planto3d.pipeline import draw_overlay

    assert draw_overlay(_floor(tmp_path, enlargement=1.0)).shape[:2] == (40, 50)


def test_the_plot_is_measured_in_the_same_frame_as_the_scale(tmp_path):
    from planto3d.pipeline import _page_size

    assert _page_size(_floor(tmp_path, enlargement=1.0)) == (50, 40)
    assert _page_size(_floor(tmp_path, enlargement=2.0)) == (100, 80)
