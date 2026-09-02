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
