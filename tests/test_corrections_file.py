"""A plan annotated once should stay annotated.

The segmenter finds only about 15% of the spaces that are open to the air,
and 30% of rooms carry neither a label nor a predicted type, so a human has
to say which spaces are terraces. Making them retype that on every run is
how a correction feature goes unused.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from correct_and_build import (  # noqa: E402
    corrections_from_lines,
    corrections_to_lines,
    parse_correction,
)


def test_corrections_round_trip():
    original = {(0, 5): "open", (1, 2): "paving", (0, 11): "lawn"}
    assert corrections_from_lines(corrections_to_lines(original)) == original


def test_saved_lines_use_the_same_syntax_as_the_flag():
    # Whatever a user can type at --correct they can also read in the file.
    lines = corrections_to_lines({(0, 5): "open"})
    assert "1:5=open" in lines  # floors print from 1, rooms index from 0


def test_comments_and_blank_lines_are_ignored():
    lines = ["# ground floor", "", "1:5=open", "   ", "# done"]
    assert corrections_from_lines(lines) == {(0, 5): "open"}


def test_an_unknown_category_is_refused_not_guessed():
    with pytest.raises(ValueError, match="not a feature category"):
        corrections_from_lines(["1:5=outdoor"])


def test_a_malformed_line_names_itself():
    with pytest.raises(ValueError, match="oops"):
        corrections_from_lines(["oops"])


class TestFloorAndRoomIndicesAreValidated:
    """Unchecked, ``floor - 1`` on a floor below 1 wraps to a negative
    Python index and silently corrects a different room than the one
    named: ``0:5=open`` hit floor index -1, the *last* floor, not floor 0.
    A negative room index does the same without even the subtraction:
    ``1:-3=open`` silently hit the third-from-last room. Off-by-one on
    floor numbering is the expected mistake once corrections live in a
    hand-editable file (Task 3's whole point), so it must be caught here,
    not quietly relabel the wrong room.
    """

    def test_floor_zero_is_rejected(self):
        with pytest.raises(ValueError, match="floor 0"):
            parse_correction("0:5=open")

    def test_a_negative_floor_is_rejected(self):
        with pytest.raises(ValueError, match="floor -2"):
            parse_correction("-2:5=open")

    def test_a_negative_room_is_rejected(self):
        with pytest.raises(ValueError, match="room -3"):
            parse_correction("1:-3=open")

    def test_floor_one_room_zero_are_the_valid_minimum(self):
        # 1:0 must still work -- the fix must not reject legitimate input.
        assert parse_correction("1:0=open") == ((0, 0), "open")

    def test_the_same_validation_applies_when_reading_a_corrections_file(self):
        with pytest.raises(ValueError, match="floor 0"):
            corrections_from_lines(["0:5=open"])
