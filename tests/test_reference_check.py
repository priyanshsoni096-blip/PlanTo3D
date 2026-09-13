"""Tests for the reference-house check.

The check compares a built model against measurements taken off a real
building, so what is pinned here is the reading of those measurements and
the arithmetic of the comparison -- not any particular house. The house's
own figures stay in a git-ignored file and never appear in a test.
"""

from types import SimpleNamespace

import pytest


def _room(left, top, right, bottom):
    return SimpleNamespace(bounds=lambda: (left, top, right, bottom))


def test_reads_the_same_syntax_the_correction_flags_use():
    from scripts.reference_check import read_truth

    truth = read_truth([
        "# comments and blank lines are ignored",
        "",
        "overall=40x60",
        "1:5=12.5x14  # a trailing comment too",
        "2:0=10x10",
    ])
    assert truth.overall == (40.0, 60.0)
    # Stored zero-based, exactly as --correct stores them.
    assert truth.rooms == {(0, 5): (12.5, 14.0), (1, 0): (10.0, 10.0)}


def test_overall_is_optional():
    from scripts.reference_check import read_truth

    assert read_truth(["1:0=10x12"]).overall is None


def test_a_bad_line_is_refused_rather_than_skipped():
    # A measurement that silently does nothing would report a pass on a
    # room nobody checked.
    from scripts.reference_check import read_truth

    with pytest.raises(ValueError):
        read_truth(["1:5=twelve by ten"])


def test_error_ignores_which_way_round_the_sizes_were_written():
    # A plan does not say which side is the width, so 12x10 and 10x12
    # describe the same room.
    from scripts.reference_check import size_error

    assert size_error((12.0, 10.0), (10.0, 12.0)) == 0.0
    # The worse side decides it: 11 against 10 is 10% out.
    assert size_error((12.0, 11.0), (12.0, 10.0)) == pytest.approx(0.10)


def test_a_room_is_measured_from_its_bounds_at_the_model_scale():
    from scripts.reference_check import measured_size

    # 250 x 300 px at 25 px/ft is 10 x 12 ft.
    assert measured_size(_room(0, 0, 250, 300), scale=25.0) == (10.0, 12.0)


def test_evaluate_applies_the_spec_gates():
    from scripts.reference_check import Truth, evaluate

    result = SimpleNamespace(
        scale=25.0,
        floors=[SimpleNamespace(
            plan=SimpleNamespace(
                rooms=[_room(0, 0, 250, 300), _room(0, 0, 300, 300)],
                footprint=[(0, 0), (1000, 0), (1000, 1500), (0, 1500)],
            ),
        )],
    )
    truth = Truth(
        overall=(40.0, 60.0),                       # exact: 1000/25, 1500/25
        rooms={(0, 0): (10.0, 12.0), (0, 1): (10.0, 12.0)},  # second is 12x12
    )
    rows = evaluate(result, truth)
    by_name = {row.name: row for row in rows}

    assert by_name["overall"].error == 0.0
    assert by_name["overall"].passed          # gate 5%
    assert by_name["1:0"].passed              # exact
    assert by_name["1:1"].error == pytest.approx(0.20)
    assert not by_name["1:1"].passed          # gate 10%


def test_a_room_number_the_model_does_not_have_is_reported_not_crashed():
    from scripts.reference_check import Truth, evaluate

    result = SimpleNamespace(
        scale=25.0,
        floors=[SimpleNamespace(plan=SimpleNamespace(rooms=[], footprint=[]))],
    )
    rows = evaluate(result, Truth(overall=None, rooms={(0, 7): (10.0, 10.0)}))
    assert len(rows) == 1
    assert rows[0].measured is None
    assert not rows[0].passed
