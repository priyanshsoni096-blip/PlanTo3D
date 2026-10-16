"""Paved does not mean open to the sky.

Every ground cover counted as open to the sky, and a garage is paved, so
BRIDGE sheet 116794 -- whose single room is labelled "3-Car Garage" -- came
out with 90% of its roof missing. A driveway and a courtyard are genuinely
open; a garage, a carport and stilt parking are roofed, and the label says
which.

Floor finishes come from a separate map, so nothing here changes what any
of these are paved with.
"""

import pytest

from planto3d.features import finish_for, is_open_to_sky


class Room:
    def __init__(self, label="", category=""):
        self.label = label
        self.category = category


@pytest.mark.parametrize("label", [
    "GARAGE", "3-Car Garage", "CARPORT", "CAR PORT", "CAR PORCH",
    "COVERED PARKING", "BASEMENT PARKING", "STILT PARKING", "COVERED WALKWAY",
])
def test_a_roofed_paved_space_keeps_its_roof(label):
    assert not is_open_to_sky(Room(label=label))


@pytest.mark.parametrize("label", [
    "DRIVEWAY", "COURTYARD", "OPEN PARKING", "VISITOR PARKING", "PATIO",
    "TERRACE", "BALCONY", "LANDSCAPE",
])
def test_what_is_genuinely_open_stays_open(label):
    assert is_open_to_sky(Room(label=label))


def test_the_floor_is_paved_either_way():
    # The fix must not turn a garage floor into carpet.
    assert finish_for("GARAGE") == finish_for("DRIVEWAY")


def test_an_unlabelled_outdoor_room_is_still_open():
    # Most plans name nothing; the predicted type has to keep working.
    assert is_open_to_sky(Room(category="outdoor"))
