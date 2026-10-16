"""The vocabulary read against ordinary room names.

Matching is by substring, so a three-letter keyword fires inside any longer
word that happens to contain it. Audited across all 460 keywords, these are
the ones that actually misread names a real drawing carries -- each was run
through `classify` before being written down:

    BALLROOM        -> open    (BAL)   and lost its roof
    BALANCE SPACE   -> open    (BAL)
    POOL TABLE ROOM -> water   (POOL)  and became a pond, unroofed
    SPARE ROOM      -> wet     (SPA)
    SPACE           -> wet     (SPA)
    DOMESTIC HELP   -> dome    (DOME)  and grew a dome
    PRODUCTION ROOM -> void    (DUCT)  and lost its floor
    FOOTBALL COURT  -> void    (OTB)
    STEPNEY STORE   -> stairs  (STEP)

Separately, six keywords could never match at all: labels are normalised
before matching -- punctuation becomes a space -- and the keywords were
not, so anything with an apostrophe or a hyphen was unreachable.
"""

import pytest

from planto3d.features import classify, finish_for, is_open_to_sky


class Room:
    def __init__(self, label):
        self.label = label
        self.category = ""


@pytest.mark.parametrize("label", [
    "BALLROOM", "BALANCE SPACE", "SPARE ROOM", "SPARE BEDROOM", "SPACE",
    "DOMESTIC HELP", "DOMESTIC STORE", "PRODUCTION ROOM", "FOOTBALL COURT",
    "STEPNEY STORE", "POOL TABLE ROOM", "BILLIARDS ROOM", "DUCTED AC ROOM",
])
def test_an_ordinary_room_is_not_read_as_a_feature(label):
    assert classify(label) is None, label
    assert not is_open_to_sky(Room(label)), label


@pytest.mark.parametrize("label,expected", [
    ("BAL", "open"),            # Indian plans abbreviate balcony
    ("BAL 2", "open"),
    ("SPA", "wet"),
    ("SPA ROOM", "wet"),
    ("DOME", "dome"),
    ("DOMED CEILING", "dome"),
    ("DUCT", "void"),
    ("SHAFT DUCT", "void"),
    ("STEP", "stairs"),
    ("STEPS", "stairs"),
    ("OTB", "void"),
    ("POOL", "water"),
    ("SWIMMING POOL", "water"),
    ("POOL DECK", "open"),
    ("BATHROOM", "wet"),        # still a substring match, correctly
    ("BALCONY", "open"),
    ("WASH AREA", "wet"),
])
def test_the_words_themselves_still_read_correctly(label, expected):
    assert classify(label) == expected, label


@pytest.mark.parametrize("label,expected", [
    ("BUTLER'S PANTRY", "wet"),
    ("DOUBLE-HEIGHT", "void"),
    ("TRIPLE-HEIGHT", "void"),
])
def test_a_keyword_written_with_punctuation_can_still_match(label, expected):
    assert classify(label) == expected


@pytest.mark.parametrize("label,expected", [
    ("OWNER'S SUITE", "timber"),
    ("CHEF'S KITCHEN", "tile"),
    ("BUTLER'S KITCHEN", "tile"),
])
def test_punctuated_finish_keywords_match_too(label, expected):
    assert finish_for(label) == expected
