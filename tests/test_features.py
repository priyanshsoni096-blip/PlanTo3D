import pytest

from planto3d.features import (
    DEFAULT_FINISH,
    GROUND_COVERS,
    classify,
    feature_for,
    finish_for_room,
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
        # Roof forms and site features -- previously zero coverage.
        ("STAIRCASE", "stairs"),
        ("GRAND STAIRCASE", "stairs"),
        ("SPIRAL STAIRCASE", "stairs"),
        ("DOME", "dome"),
        ("ONION DOME", "dome"),
        ("PITCHED ROOF", "pitched"),
        ("GABLE ROOF", "pitched"),
        ("SKYLIGHT", "glazed"),
        ("GLASS ROOF", "glazed"),
        ("OVERHEAD TANK", "tank"),
        ("WATER TANK", "tank"),
        ("CHIMNEY", "chimney"),
        ("CHIMNEY STACK", "chimney"),
        ("TOWER", "tower"),
        ("CLOCK TOWER", "tower"),
        ("CANOPY", "canopy"),
        ("PORTICO", "canopy"),
        ("RAMP", "ramp"),
        ("WHEELCHAIR RAMP", "ramp"),
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


class TestRoomFunctionWithoutAName:
    """Most plans print no room names, so the predicted type carries them.

    Across sixty CubiCasa plans OCR read a name on three. Everything that
    makes a model look like a house -- tiled wet floors, railed balconies,
    boarded bedrooms -- hung off those names until the segmenter began
    predicting the room type instead.
    """

    def _room(self, label="", category=""):
        return Room(
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            label=label,
            category=category,
        )

    def test_a_predicted_bath_is_wet_without_any_label(self):
        assert feature_for(self._room(category="bath")) == "wet"

    def test_a_predicted_outdoor_room_earns_a_railing(self):
        assert feature_for(self._room(category="outdoor")) == "open"

    def test_a_printed_name_beats_the_prediction(self):
        # The verandah case: the segmenter is trained on Finnish apartments
        # and calls it an outdoor space, which would rail it like a balcony.
        # The drawing says VERANDAH, and the drawing is right.
        room = self._room(label="VERANDAH", category="outdoor")

        assert feature_for(room) == "paving"

    def test_a_room_with_neither_implies_nothing(self):
        assert feature_for(self._room()) is None

    def test_a_plain_predicted_type_implies_no_feature(self):
        # A bedroom builds nothing a nameless room would not; it only
        # changes the floor finish.
        assert feature_for(self._room(category="bedroom")) is None

    def test_predicted_types_choose_the_floor_finish(self):
        assert finish_for_room(self._room(category="kitchen")) == "tile"
        assert finish_for_room(self._room(category="bath")) == "tile"
        assert finish_for_room(self._room(category="bedroom")) == "timber"

    def test_an_unknown_type_falls_back_to_the_default_finish(self):
        assert finish_for_room(self._room(category="observatory")) == DEFAULT_FINISH

    def test_a_named_room_keeps_its_own_finish(self):
        assert finish_for_room(self._room(label="KITCHEN", category="bedroom")) == "tile"

    def test_grouping_uses_predictions_where_names_are_missing(self):
        grouped = group_by_feature(
            [
                self._room(category="bath"),
                self._room(category="outdoor"),
                self._room(label="BALCONY"),
                self._room(category="bedroom"),
            ]
        )

        assert len(grouped["wet"]) == 1
        assert len(grouped["open"]) == 2


class TestTheLanguageTheTrainingDataIsDrawnIn:
    """The segmenter is trained on a Finnish corpus.

    Those are the drawings it reads best, and none of their words were
    understood. A parveke -- a balcony -- was built as a sealed room,
    walling in the windows that open onto it.
    """

    @pytest.mark.parametrize(
        "label, category",
        [
            ("PARVEKE", "open"),
            ("PARV", "open"),
            ("TERASSI", "open"),
            ("KATTOTERASSI", "open"),
            ("BALKONG", "open"),
            ("KYLPYHUONE", "wet"),
            ("KPH", "wet"),
            ("PESUHUONE", "wet"),
            ("KODINHOITOHUONE", "wet"),
            ("KHH", "wet"),
            ("KEITTIO", "wet"),
        ],
    )
    def test_finnish_and_swedish_room_names(self, label, category):
        assert classify(label) == category

    def test_short_marks_are_matched_whole(self):
        # As substrings these would match half the English vocabulary.
        assert classify("PARVIS") is None
        assert classify("PARV") == "open"

    def test_a_stray_single_letter_tiles_nothing(self):
        # OCR litters a dense drawing with single letters, and one landing
        # in a room would tile its floor. "K" for keittio is left out for
        # that reason, at the cost of the plans that abbreviate that far.
        assert classify("K") is None

    def test_rooms_that_only_change_the_finish_imply_no_feature(self):
        # A makuuhuone is a bedroom and an olohuone a living room. Neither
        # builds anything a plain room would not.
        for label in ("OH", "MH", "ET"):
            assert classify(label) is None


class TestSpanishAndPortuguese:
    """Today's real coverage of the two languages Phase 7 will add.

    Two of these already return an answer, and neither is real support:
    "balcón" matches "open" only because "BAL" -- kept for OCR truncation
    of the English word -- happens to be a substring of "balcon", and
    "garagem" matches "paving" only because "GARAGE" is a substring of
    it. "patio" is a genuine hit: PATIO is already an explicit English/
    Spanish loanword keyword. Everything else below returns nothing,
    which is the honest baseline -- if Messi uploads a plan labelled
    "varanda" or "cochera", the vocabulary does not know either word.
    This class documents that gap; it is not meant to pass differently
    until the vocabulary itself is extended.
    """

    @pytest.mark.parametrize(
        "label",
        [
            "VARANDA",
            "COCHERA",
            "JARDIN",
            "JARDIM",
            "TERRAZA",
            "TERRACO",
            "SOTANO",
            "PORAO",
        ],
    )
    def test_unsupported_spanish_and_portuguese_terms(self, label):
        assert classify(label) is None

    def test_two_terms_already_match_by_accident_not_design(self):
        # Documented so a future reader does not mistake these for real
        # Portuguese/Spanish support and leave them out of Phase 7's work.
        assert classify("BALCON") == "open"  # via "BAL", not "balcón"
        assert classify("GARAGEM") == "paving"  # via "GARAGE", not design
        assert classify("PATIO") == "paving"  # this one is genuine
