import pytest

from planto3d.features import (
    GROUND_COVERS,
    classify,
    group_by_feature,
    regions_from_labels,
)
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
        # A verandah is a covered space at ground level, not an elevated
        # balcony -- railing one puts a balustrade across the front door.
        ("VERANDAH 8'10\"X5'8\"", "paving"),
        ("VERANDA", "paving"),
        ("PORCH", "paving"),
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


class TestRegionsFromLabels:
    """Outdoor areas are not rooms, so their labels are the only source."""

    def _box(self, text, x=500.0, y=400.0):
        from planto3d.calibrate import TextBox

        return TextBox(text=text, bbox=(int(x) - 40, int(y) - 6, 80, 12), confidence=90.0)

    def test_a_dimensioned_label_becomes_a_region_of_that_size(self):
        # LANDSCAPE 49'0"X13'2" is 645 sq ft, whatever the hatching covers.
        regions = regions_from_labels([self._box("LANDSCAPE 49'0\"X13'2\"")], scale=20.0)

        polygon = regions["lawn"][0]
        width = max(p[0] for p in polygon) - min(p[0] for p in polygon)
        height = max(p[1] for p in polygon) - min(p[1] for p in polygon)
        assert width / 20.0 == pytest.approx(49.0, abs=0.5)
        assert height / 20.0 == pytest.approx(13.17, abs=0.5)

    def test_the_region_is_centred_on_its_label(self):
        # Drafters centre a label in the space it names.
        regions = regions_from_labels(
            [self._box("PARKING 20'0\"X25'0\"", x=900, y=700)], scale=20.0
        )

        polygon = regions["paving"][0]
        centre_x = sum(p[0] for p in polygon) / 4
        centre_y = sum(p[1] for p in polygon) / 4
        assert centre_x == pytest.approx(900, abs=2)
        assert centre_y == pytest.approx(700, abs=2)

    def test_a_pool_label_places_water(self):
        regions = regions_from_labels([self._box("SWIMMING POOL 30'0\"X12'0\"")], scale=20.0)

        assert "water" in regions

    def test_labels_without_dimensions_are_skipped(self):
        # Nothing to size the region with; a guessed extent is worse than none.
        assert regions_from_labels([self._box("TERRACE GARDEN")], scale=20.0) == {}

    def test_a_size_printed_under_its_name_is_paired_with_it(self):
        name = self._box("LANDSCAPE", x=500, y=400)
        size = self._box("49'0\"X13'2\"", x=500, y=418)

        regions = regions_from_labels([name, size], scale=20.0)

        assert "lawn" in regions

    def test_a_size_belonging_to_another_room_is_not_stolen(self):
        # "TERRACE GARDEN 2130 SQ.FT." states an area, not a width. Pairing
        # it with whichever dimension sits nearest builds the garden at some
        # other room's size, in the wrong place.
        name = self._box("TERRACE GARDEN", x=500, y=400)
        elsewhere = self._box("21'6\"X27'6\"", x=1400, y=402)

        assert regions_from_labels([name, elsewhere], scale=20.0) == {}

    def test_a_size_printed_above_a_name_is_not_paired_with_it(self):
        # Dimensions are set beneath their name; text above belongs to the
        # room before it.
        name = self._box("LANDSCAPE", x=500, y=400)
        above = self._box("15'0\"X18'0\"", x=500, y=380)

        assert regions_from_labels([name, above], scale=20.0) == {}

    def test_ordinary_rooms_produce_no_region(self):
        assert regions_from_labels([self._box("BEDROOM 15'0\"X18'0\"")], scale=20.0) == {}

    def test_an_unusable_scale_produces_nothing(self):
        assert regions_from_labels([self._box("LANDSCAPE 49'0\"X13'2\"")], scale=0.0) == {}
