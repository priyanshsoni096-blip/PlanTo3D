import pytest

from planto3d.features import GROUND_COVERS, classify, group_by_feature
from planto3d.geometry_types import Room


@pytest.mark.parametrize(
    "label, expected",
    [
        # Water, which is a hole in the ground rather than a mat laid on it.
        ("SWIMMING POOL", "water"),
        ("POOL 20'0\"X10'0\"", "water"),
        ("JACUZZI", "water"),
        ("WATER BODY", "water"),
        # Planting.
        ("LANDSCAPE 49'0\"X13'2\"", "lawn"),
        ("TERRACE GARDEN 2130 SQ.FT.", "lawn"),
        ("LAWN", "lawn"),
        ("PLANTER", "lawn"),
        # Openings through a floor.
        ("DOUBLE HEIGHT", "void"),
        ("OTS", "void"),
        ("OPEN TO SKY", "void"),
        ("SHAFT", "void"),
        # Hard landscaping.
        ("PARKING 20'10\"X25'4\"", "paving"),
        ("CAR PORCH", "paving"),
        ("COURTYARD", "paving"),
        ("SIT OUT", "paving"),
        # Open-edged, needing a railing.
        ("BALCONY 20'6\"X6'6\"", "open"),
        ("BAL", "open"),
        ("VERANDAH 8'10\"X5'8\"", "open"),
        # Wet areas.
        ("DRESS/TOILET", "wet"),
        ("W.C.", "wet"),
        ("CHEF'S KITCHEN/WASH AREA", "wet"),
        ("UTILITY", "wet"),
    ],
)
def test_labels_map_to_the_right_feature(label, expected):
    assert classify(label) == expected


@pytest.mark.parametrize(
    "label", ["BEDROOM", "KITCHEN", "STORE", "TEMPLE", "STUDY", "MULTI-PURPOSE HALL"]
)
def test_ordinary_rooms_have_no_special_feature(label):
    assert classify(label) is None


@pytest.mark.parametrize("label", ["", "5 GE", "~ a", "UFT", "PLAV-AREA"])
def test_noise_and_misreads_are_not_guessed_at(label):
    # A misread label must not sink a swimming pool into a bedroom.
    assert classify(label) is None


def test_a_terrace_garden_is_planting_not_an_open_deck():
    # "TERRACE GARDEN" contains "TERRACE"; the more specific phrase must win,
    # or a lawn becomes a bare deck.
    assert classify("TERRACE GARDEN") == "lawn"
    assert classify("TERRACE") == "open"


def test_matching_ignores_case_and_surrounding_text():
    assert classify("Swimming Pool (proposed)") == "water"


def test_ground_covers_are_the_outdoor_categories():
    assert GROUND_COVERS == {"water", "lawn", "paving"}
    assert "void" not in GROUND_COVERS  # a void removes floor, not covers it


class TestGrouping:
    def _room(self, label, offset=0.0):
        return Room(
            polygon=[
                (offset, offset),
                (offset + 50, offset),
                (offset + 50, offset + 50),
                (offset, offset + 50),
            ],
            label=label,
        )

    def test_rooms_group_by_feature(self):
        grouped = group_by_feature(
            [
                self._room("SWIMMING POOL"),
                self._room("LANDSCAPE", 100),
                self._room("BALCONY", 200),
                self._room("BEDROOM", 300),
            ]
        )

        assert set(grouped) == {"water", "lawn", "open"}
        assert len(grouped["water"]) == 1

    def test_a_plan_of_plain_rooms_groups_to_nothing(self):
        assert group_by_feature([self._room("BEDROOM"), self._room("KITCHEN", 100)]) == {}
