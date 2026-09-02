"""Which drafting tradition a drawing belongs to, where it says so.

Scale rests on assumed element sizes, and those differ by tradition.
Measured against CubiCasa's own annotations over 30 sheets, the implied
real wall thickness has a median of 0.648 ft against the 0.75 the code
assumes -- and correcting only that takes the pooled error from 17.7% to
9.9%. Correcting the door constant instead makes it worse, 20.1%.

So the wall value is changed, the door value is not, and both only where
the drawing identifies its tradition. Everything unrecognised keeps the
shipped defaults, because a constant fitted to the one corpus with
ground truth would silently degrade every corpus without it.
"""

import pytest

from planto3d.calibrate import (
    TYPICAL_DOOR_FT,
    TYPICAL_WALL_FT,
    detect_convention,
    element_sizes,
)


class _Box:
    """The only part of a TextBox this reads."""

    def __init__(self, text: str) -> None:
        self.text = text


def _boxes(*words: str) -> list:
    return [_Box(word) for word in words]


@pytest.mark.parametrize(
    "label", ["PARVEKE", "KYLPYHUONE", "KEITTIO", "OLOHUONE", "BALKONG"]
)
def test_finnish_and_swedish_names_identify_the_nordic_tradition(label):
    # Two distinct Nordic words are required to corroborate a tradition
    # (MIN_CONVENTION_HITS = 2), so each case pairs the parametrized word
    # with a second, different Nordic word not already in this list.
    assert detect_convention(_boxes(label, "MAKUUHUONE")) == "nordic"


def test_english_names_are_not_claimed_for_any_tradition():
    # English is the default the shipped constants already serve; claiming
    # it as a tradition would change behaviour for the plans that work.
    assert detect_convention(_boxes("BEDROOM", "KITCHEN", "BALCONY")) is None


def test_nothing_readable_identifies_nothing():
    # The common case on a scan OCR cannot read. It must not guess.
    assert detect_convention([]) is None
    assert detect_convention(_boxes("", "5 GE", "~ a")) is None


def test_one_stray_word_does_not_decide_a_sheet():
    # A single match is a misread waiting to happen; a tradition is a
    # property of the whole drawing, so it takes corroboration.
    assert detect_convention(_boxes("PARVEKE", "BEDROOM", "KITCHEN")) is None


def test_the_nordic_tradition_uses_its_measured_wall_thickness():
    door_ft, wall_ft = element_sizes(_boxes("PARVEKE", "KEITTIO"))
    assert wall_ft == pytest.approx(0.648)
    # The door constant is deliberately unchanged: correcting it measured
    # worse, 20.1% against 17.7% shipped.
    assert door_ft == pytest.approx(TYPICAL_DOOR_FT)


def test_an_unrecognised_drawing_keeps_the_shipped_defaults():
    door_ft, wall_ft = element_sizes(_boxes("BEDROOM", "KITCHEN"))
    assert (door_ft, wall_ft) == (TYPICAL_DOOR_FT, TYPICAL_WALL_FT)
