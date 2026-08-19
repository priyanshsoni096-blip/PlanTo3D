"""Reading the room sizes CubiCasa hides inside its annotations.

Scale decides how big the finished house is, and until these were found it
could not be checked against anything but the one drawing that printed its
dimensions. They are marked ``display: none`` and never render, so they are
not a shortcut the pipeline could take on a real drawing -- only a way to
score what it infers.
"""

import pytest

from planto3d.cubicasa import ground_truth_scale, parse_feet


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12'4\"", 12 + 4 / 12),
        ("9'", 9.0),
        ("7' 6\"", 7.5),
        ("15'11\"", 15 + 11 / 12),
        ("nothing measurable", None),
    ],
)
def test_feet_and_inches_parse(text, expected):
    assert parse_feet(text) == pytest.approx(expected) if expected else parse_feet(text) is None


def _svg(tmp_path, spaces: str):
    path = tmp_path / "model.svg"
    path.write_text(
        '<?xml version="1.0"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">'
        f"{spaces}</svg>"
    )
    return path


def _space(x0, y0, x1, y1, measurement):
    return (
        f'<g class="Space Bedroom">'
        f'<polygon points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1} "/>'
        f'<g class="Dimension"><g class="SpaceDimensionsLabel">'
        f'<g class="TextLabel DimensionMeasureLabel" style="display: none;">'
        f"<text>{measurement}</text></g></g></g></g>"
    )


def test_scale_comes_from_a_rooms_stated_size(tmp_path):
    # 300px across a room stated as 10 ft is 30 pixels per foot.
    svg = _svg(tmp_path, _space(0, 0, 300, 300, "10' x 10'"))

    assert ground_truth_scale(svg) == pytest.approx(30.0)


def test_a_rooms_orientation_does_not_matter(tmp_path):
    # The label does not say which measurement runs which way, so both
    # pairings are tried and the one that agrees is believed.
    across = _svg(tmp_path, _space(0, 0, 300, 600, "10' x 20'"))
    (tmp_path / "b").mkdir()
    down = _svg(tmp_path / "b", _space(0, 0, 300, 600, "20' x 10'"))

    assert ground_truth_scale(across) == pytest.approx(30.0)
    assert ground_truth_scale(down) == pytest.approx(30.0)


def test_rooms_that_do_not_agree_are_left_out(tmp_path):
    # An L-shaped room's bounding box overstates it, and believing the
    # label against that box would report a scale far off the truth.
    svg = _svg(
        tmp_path,
        _space(0, 0, 300, 300, "10' x 10'") + _space(400, 0, 900, 100, "10' x 10'"),
    )

    assert ground_truth_scale(svg) == pytest.approx(30.0)


def test_cupboards_are_too_small_to_measure(tmp_path):
    # A foot of rounding on a 3 ft cupboard is a third of the answer.
    svg = _svg(
        tmp_path,
        _space(0, 0, 300, 300, "10' x 10'") + _space(400, 0, 460, 460, "2' x 2'"),
    )

    assert ground_truth_scale(svg) == pytest.approx(30.0)


def test_the_median_survives_one_bad_room(tmp_path):
    svg = _svg(
        tmp_path,
        _space(0, 0, 300, 300, "10' x 10'")
        + _space(0, 400, 300, 700, "10' x 10'")
        + _space(0, 800, 900, 1700, "10' x 10'"),
    )

    assert ground_truth_scale(svg) == pytest.approx(30.0)


def test_an_annotation_stating_no_sizes_returns_nothing(tmp_path):
    svg = _svg(tmp_path, '<g class="Space Bedroom"><polygon points="0,0 10,0 10,10 "/></g>')

    assert ground_truth_scale(svg) is None
