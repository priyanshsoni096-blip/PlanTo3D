"""Tests for reading CVC-FP annotations into masks.

CVC-FP is the second drafting tradition every generalisation claim rests on,
so its masks are ground truth for those claims. What is pinned here is that a
shape is painted whole however its polygons happen to overlap.
"""

from planto3d.classes import BACKGROUND, WALL, WINDOW
from planto3d.cvc_fp import svg_to_mask

SVG_HEADER = '<svg xmlns="http://www.w3.org/2000/svg">'


def _write_svg(tmp_path, body: str):
    path = tmp_path / "1_gt_1.svg"
    path.write_text(f"{SVG_HEADER}{body}</svg>")
    return path


def _polygon(cls: str, x0, y0, x1, y1) -> str:
    return f'<polygon class="{cls}" points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'


def test_overlapping_walls_leave_no_hole(tmp_path):
    # Two wall pieces meeting at a corner overlap. Filled together in one
    # call, OpenCV treated the overlap as a hole and left it background:
    # measured over the 122 plans, 468,747 pixels of wall were missing,
    # 1.04% of all wall.
    svg = _write_svg(
        tmp_path,
        _polygon("Wall", 0, 0, 60, 20) + _polygon("Wall", 40, 0, 100, 20),
    )

    mask = svg_to_mask(svg, (40, 110))

    # The overlap, x 40-60, is wall like the rest of both pieces.
    assert (mask[3:17, 3:97] == WALL).all()
    assert mask[30, 50] == BACKGROUND


def test_a_window_still_paints_over_its_wall(tmp_path):
    # Openings go last so a window reads as a hole in its wall, not the wall
    # swallowing it. The fill change must not disturb that order.
    svg = _write_svg(
        tmp_path,
        _polygon("Wall", 0, 0, 100, 20) + _polygon("Window", 30, 5, 70, 15),
    )

    mask = svg_to_mask(svg, (40, 110))

    assert (mask[7:14, 32:69] == WINDOW).all()
    assert mask[2, 10] == WALL
