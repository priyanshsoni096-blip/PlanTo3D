"""Words filed under the wrong meaning, found by reading all 15 categories.

Three were plainly wrong when the list was read against what each category
does to the building:

- "OPEN TO SKY" sat under `void`, which is a hole through a *floor* and is
  deliberately not open to the sky. A room whose printed name says it is
  open to the sky was built with a roof over it.
- "GREEN WALL", "VERTICAL GARDEN" and "LIVING WALL" sat under `lawn`, a
  ground cover: planting on a wall was laid on the floor as a lawn, and
  the roof above it removed.
- "LOGGIA", "ARCADE", "COLONNADE", "CLOISTER", "SCREENED PORCH", "GAZEBO",
  "PAVILION" and "CABANA" sat under `open`. Each is a roofed structure --
  that is what the word means -- so each lost its roof.

Left alone deliberately: a verandah, a porch and a sit-out are paved and
open here, which is a judgement rather than an error, and on the reference
house the rooms opened to the sky are exactly the right ones.
"""

import pytest

from planto3d.features import classify, is_open_to_sky


class Room:
    def __init__(self, label):
        self.label = label
        self.category = ""


def test_open_to_sky_stays_a_void_on_purpose():
    # The name argues for the open category, and that was tried. On Indian
    # drawings "OTS -- open to sky" marks a light well: a shaft through the
    # floors, which is what `void` builds. As `open` it would keep every
    # floor and remove the roof, turning the shaft into a terrace on each
    # storey, so the roof over it is the lesser error.
    assert classify("OPEN TO SKY") == "void"
    assert classify("OTS") == "void"


@pytest.mark.parametrize("label", ["OPEN TO BELOW", "DOUBLE HEIGHT", "LIGHT WELL", "VOID"])
def test_a_hole_through_a_floor_is_still_a_void(label):
    assert classify(label) == "void"
    assert not is_open_to_sky(Room(label))


@pytest.mark.parametrize("label", ["GREEN WALL", "VERTICAL GARDEN", "LIVING WALL"])
def test_planting_on_a_wall_is_not_ground(label):
    assert classify(label) is None
    assert not is_open_to_sky(Room(label))


@pytest.mark.parametrize("label", ["GARDEN", "LAWN", "TERRACE GARDEN", "PLANTER"])
def test_planting_on_the_ground_still_is(label):
    assert classify(label) == "lawn"


@pytest.mark.parametrize("label", [
    "LOGGIA", "ARCADE", "COLONNADE", "CLOISTER", "SCREENED PORCH",
    "GAZEBO", "PAVILION", "CABANA",
])
def test_a_roofed_structure_keeps_its_roof(label):
    assert classify(label) == "canopy"
    assert not is_open_to_sky(Room(label))


@pytest.mark.parametrize("label", ["BALCONY", "TERRACE", "SUN DECK", "PERGOLA", "OPEN TERRACE"])
def test_what_is_open_sided_and_unroofed_stays_open(label):
    assert classify(label) == "open"
    assert is_open_to_sky(Room(label))
