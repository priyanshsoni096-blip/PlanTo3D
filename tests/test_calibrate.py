import numpy as np
import pytest

from planto3d.calibrate import (
    MAX_OCR_UPSCALE,
    MAX_PRINTED_DISAGREEMENT,
    TextBox,
    corroborated,
    estimate_scale,
    parse_area_text,
    parse_dimension_text,
    read_text_boxes,
    scale_from_areas,
)
from planto3d.geometry_types import Room


class TestParseDimensionText:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("15'0\"X18'0\"", (15.0, 18.0)),
            ("BEDROOM 15'0\"X18'0\"", (15.0, 18.0)),
            ("7'6\"X27'0\"", (7.5, 27.0)),
            ("13'10\"X16'6\"", (13 + 10 / 12, 16.5)),
        ],
    )
    def test_reads_feet_and_inches(self, text, expected):
        parsed = parse_dimension_text(text)
        assert parsed == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text",
        [
            "15’0” X 18’0”",  # curly quotes, as OCR often returns
            "15' 0\" x 18' 0\"",  # lowercase x and loose spacing
            "15'0X18'0",  # inch marks dropped entirely
            # Degree signs substituted for foot/inch marks. Seen on the real
            # first-floor sheet, where these were the only reason otherwise
            # clean dimensions failed to parse.
            "15'°0\"X18'0\"",
            "15'0°X18'0\"",
        ],
    )
    def test_tolerates_ocr_variations(self, text):
        assert parse_dimension_text(text) == pytest.approx((15.0, 18.0))

    @pytest.mark.parametrize("text", ["KITCHEN", "", "ASILE", "3,050 SQ. FT.", "OPTION 04"])
    def test_returns_none_for_text_without_dimensions(self, text):
        assert parse_dimension_text(text) is None


class TestEstimateScale:
    def _room_with_label(self, size_px: float, dimension: str, offset: float = 0.0):
        # A square room of size_px pixels labelled with its real dimensions.
        room = Room(
            polygon=[
                (offset, offset),
                (offset + size_px, offset),
                (offset + size_px, offset + size_px),
                (offset, offset + size_px),
            ]
        )
        centre = offset + size_px / 2
        box = TextBox(text=dimension, bbox=(int(centre), int(centre), 10, 5), confidence=90.0)
        return room, box

    def test_derives_pixels_per_foot_from_a_labelled_room(self):
        # A 10ft x 10ft room drawn 200px across is 20 px/ft.
        room, box = self._room_with_label(200.0, "10'0\"X10'0\"")

        assert estimate_scale([room], [box]) == pytest.approx(20.0)

    def test_matches_long_side_to_long_dimension(self):
        # The label does not say which dimension is which, so the extractor
        # pairs longest-with-longest rather than assuming an order.
        room = Room(polygon=[(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)])
        box = TextBox(text="10'0\"X20'0\"", bbox=(200, 100, 10, 5), confidence=90.0)

        assert estimate_scale([room], [box]) == pytest.approx(20.0)

    def test_takes_the_median_so_one_misread_does_not_skew_the_scale(self):
        rooms, boxes = [], []
        for i in range(3):
            room, box = self._room_with_label(200.0, "10'0\"X10'0\"", offset=i * 300)
            rooms.append(room)
            boxes.append(box)
        # A fourth room whose OCR misread makes it look ten times the scale.
        bad_room, bad_box = self._room_with_label(200.0, "1'0\"X1'0\"", offset=1200)
        rooms.append(bad_room)
        boxes.append(bad_box)

        assert estimate_scale(rooms, boxes) == pytest.approx(20.0)

    def test_ignores_a_dimension_outside_every_room(self):
        room, _ = self._room_with_label(200.0, "10'0\"X10'0\"")
        stray = TextBox(text="10'0\"X10'0\"", bbox=(9000, 9000, 10, 5), confidence=90.0)

        assert estimate_scale([room], [stray]) is None

    def test_returns_none_when_nothing_is_measurable(self):
        room = Room(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        box = TextBox(text="KITCHEN", bbox=(5, 5, 10, 5), confidence=90.0)

        assert estimate_scale([room], [box]) is None
        assert estimate_scale([], []) is None


class TestReadTextBoxes:
    def test_groups_words_into_lines_so_split_dimensions_stay_together(self, monkeypatch):
        # Tesseract splits "15'0"X18'0"" across word tokens; a dimension is
        # only usable if its pieces are rejoined with a single location.
        import planto3d.calibrate as calibrate

        def fake_image_to_data(image, output_type):
            return {
                "text": ["BEDROOM", "15'0\"", "X18'0\"", "KITCHEN"],
                "left": [100, 100, 140, 400],
                "top": [200, 220, 220, 500],
                "width": [60, 35, 40, 70],
                "height": [12, 10, 10, 12],
                "conf": ["95", "88", "86", "91"],
                "block_num": [1, 1, 1, 2],
                "par_num": [1, 1, 1, 1],
                "line_num": [1, 2, 2, 1],
            }

        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", fake_image_to_data)

        # Comfortably over MIN_OCR_LONG_EDGE, so nothing is enlarged and
        # the faked coordinates come back as given. Enlargement has its own
        # tests below.
        boxes = read_text_boxes(np.zeros((1400, 1400, 3), dtype=np.uint8))

        texts = [b.text for b in boxes]
        assert "BEDROOM" in texts
        assert "KITCHEN" in texts
        dimension = next(b for b in boxes if parse_dimension_text(b.text))
        assert parse_dimension_text(dimension.text) == pytest.approx((15.0, 18.0))
        # The merged box spans both words.
        assert dimension.bbox[0] == 100
        assert dimension.bbox[2] == 80

    def test_splits_a_line_when_words_are_far_apart(self, monkeypatch):
        # Tesseract puts two distant room labels on one "line". Merged, the
        # second dimension is lost and the box centre lands between rooms.
        # Observed on the real first-floor sheet.
        import planto3d.calibrate as calibrate

        def fake_image_to_data(image, output_type):
            return {
                "text": ["14'6\"X11'0\"", "15'0\"X22'0\""],
                "left": [100, 1400],
                "top": [800, 800],
                "width": [90, 90],
                "height": [14, 14],
                "conf": ["85", "84"],
                "block_num": [1, 1],
                "par_num": [1, 1],
                "line_num": [1, 1],
            }

        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", fake_image_to_data)

        boxes = read_text_boxes(np.zeros((1600, 2275, 3), dtype=np.uint8))

        assert len(boxes) == 2
        parsed = [parse_dimension_text(b.text) for b in boxes]
        assert parsed[0] == pytest.approx((14.5, 11.0))
        assert parsed[1] == pytest.approx((15.0, 22.0))
        # Each box sits over its own room rather than between them.
        assert boxes[0].centre[0] < 300
        assert boxes[1].centre[0] > 1300

    def test_keeps_words_of_one_label_together(self, monkeypatch):
        # The flip side: a dimension split across adjacent word tokens must
        # still merge, or it never parses at all.
        import planto3d.calibrate as calibrate

        def fake_image_to_data(image, output_type):
            return {
                "text": ["15'0\"", "X18'0\""],
                "left": [100, 142],
                "top": [220, 220],
                "width": [38, 40],
                "height": [12, 12],
                "conf": ["88", "86"],
                "block_num": [1, 1],
                "par_num": [1, 1],
                "line_num": [1, 1],
            }

        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", fake_image_to_data)

        boxes = read_text_boxes(np.zeros((600, 600, 3), dtype=np.uint8))

        assert len(boxes) == 1
        assert parse_dimension_text(boxes[0].text) == pytest.approx((15.0, 18.0))

    def test_drops_low_confidence_and_empty_tokens(self, monkeypatch):
        import planto3d.calibrate as calibrate

        def fake_image_to_data(image, output_type):
            return {
                "text": ["GOOD", "", "noise"],
                "left": [10, 50, 90],
                "top": [10, 10, 10],
                "width": [40, 0, 40],
                "height": [12, 0, 12],
                "conf": ["95", "-1", "5"],
                "block_num": [1, 1, 1],
                "par_num": [1, 1, 1],
                "line_num": [1, 2, 3],
            }

        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", fake_image_to_data)

        boxes = read_text_boxes(np.zeros((100, 200, 3), dtype=np.uint8))

        assert [b.text for b in boxes] == ["GOOD"]


def test_textbox_centre_is_the_middle_of_its_bounds():
    assert TextBox(text="X", bbox=(10, 20, 30, 40), confidence=90.0).centre == (25.0, 40.0)


class TestPrintedAreas:
    """Scale from an area the drawing prints on itself.

    3DPlanNet (Park & Kim, *Electronics* 2021, 10, 2729, equation 1)
    calibrates this way and reaches 97% on drawings that print an area.
    Indian residential sheets print them routinely; the Finnish corpus
    does not print any, so this path is opportunistic and gated.
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("TERRACE GARDEN 2130 SQ.FT.", 2130.0),
            ("600 SQ.FT.", 600.0),
            ("PANTRY/DECK SEATING 600 SQ FT", 600.0),
            ("1,250 SQFT", 1250.0),
            ("AREA 450 SQ. FT", 450.0),
            ("125 m2", 1345.5),
            ("98.5 SQM", 1060.2),
        ],
    )
    def test_it_reads_an_area_however_the_sheet_writes_it(self, text, expected):
        parsed = parse_area_text(text)
        assert parsed is not None, text
        assert parsed == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize(
        "text",
        [
            "BEDROOM 3",       # a room number
            "ROOM 101",        # a door tag
            "2130",            # a bare number is not an area
            "13'0\" X 10'0\"",  # a dimension pair, handled elsewhere
            "5 SQ.FT.",        # too small to be a room
            "99000 SQ.FT.",    # a development, not a room
        ],
    )
    def test_it_refuses_what_is_not_an_area(self, text):
        # The unit is required and the size is bounded. A bare number read
        # as an area would resize the whole building.
        assert parse_area_text(text) is None

    def test_a_label_is_charged_to_the_space_not_a_fragment_of_it(self):
        # The segmenter emits a polygon per class, so outlines overlap and
        # a point sits inside several. Taking whichever came first put the
        # reference sheet's terrace label on a region a third its size and
        # the building out by a third.
        big = Room(
            polygon=[(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)],
            label="TERRACE",
        )
        fragment = Room(
            polygon=[(150.0, 150.0), (250.0, 150.0), (250.0, 250.0), (150.0, 250.0)],
            category="outdoor",
        )
        label = TextBox(text="1600 SQ.FT.", bbox=(195, 195, 10, 10), confidence=90.0)

        # 400x400 px over 1600 sq ft is 10 px/ft; the fragment would say 2.5.
        for rooms in ([big, fragment], [fragment, big]):
            assert scale_from_areas(rooms, [label]) == pytest.approx(10.0, rel=0.01)

    def test_no_area_on_the_sheet_is_not_an_error(self):
        room = Room(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert scale_from_areas([room], []) is None


class TestTheGateOnPrintedScale:
    """Text read off a drawing is better evidence than a door width -- but
    only once it has been checked. OCR on a busy sheet drops a foot mark or
    takes a tag for a size, and an ungated reading resizes the building."""

    def test_a_reading_close_to_the_geometry_is_believed(self):
        assert corroborated(26.3, 28.4)

    def test_a_reading_out_by_a_multiple_is_not(self):
        # A dropped digit, a transposed pair, a room number read as a size.
        assert not corroborated(3.0, 32.0)
        assert not corroborated(64.0, 32.0)

    def test_with_nothing_to_check_against_it_is_taken_on_trust(self):
        # Refusing it would leave the drawing with no scale at all, and it
        # is still the best evidence available.
        assert corroborated(32.0, None)

    def test_the_gate_is_looser_than_the_geometry_it_checks(self):
        # The measured estimate is itself only good to about a fifth, so a
        # tighter gate would reject correct readings. This guards against
        # someone tightening it to look rigorous.
        assert MAX_PRINTED_DISAGREEMENT > 0.2


class TestSmallSheetsAreEnlargedBeforeReading:
    """OCR has a resolution floor, and plenty of plans sit under it.

    A sheet that fits a whole house into 600 pixels prints its dimensions
    at a size Tesseract cannot resolve: the letters are there and come back
    as nothing. Measured over 30 such sheets, enlarging past the floor took
    the dimension pairs recovered from 1 to 8.
    """

    def _fake(self, seen):
        def image_to_data(image, output_type):
            seen.append(image.shape[:2])
            return {
                "text": ["12'-6\"", "x 13'-8\""],
                "left": [200, 300],
                "top": [400, 400],
                "width": [80, 90],
                "height": [20, 20],
                "conf": ["90", "90"],
                "block_num": [1, 1],
                "par_num": [1, 1],
                "line_num": [1, 1],
            }
        return image_to_data

    def test_a_small_sheet_is_enlarged(self, monkeypatch):
        import planto3d.calibrate as calibrate

        seen = []
        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", self._fake(seen))
        read_text_boxes(np.zeros((400, 600, 3), dtype=np.uint8))

        assert seen[0] == (800, 1200)

    def test_a_large_sheet_is_left_alone(self, monkeypatch):
        import planto3d.calibrate as calibrate

        seen = []
        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", self._fake(seen))
        read_text_boxes(np.zeros((1600, 2000, 3), dtype=np.uint8))

        assert seen[0] == (1600, 2000)

    def test_boxes_come_back_in_the_original_frame(self, monkeypatch):
        # Callers match these against room polygons drawn in the original
        # image's coordinates. A box left in the enlarged frame lands in the
        # wrong room, or off the page.
        import planto3d.calibrate as calibrate

        monkeypatch.setattr(calibrate.pytesseract, "image_to_data", self._fake([]))
        boxes = read_text_boxes(np.zeros((400, 600, 3), dtype=np.uint8))

        assert boxes, "the line should still be read"
        left, top, width, _ = boxes[0].bbox
        assert (left, top) == (100, 200)
        assert width == pytest.approx(95, abs=2)

    def test_enlargement_is_capped(self):
        # Interpolation cannot invent strokes that were never sampled: at
        # three times, the same 30 sheets yielded the same 8 dimensions as
        # at two. The cap says so rather than trusting a future reader to
        # rediscover it.
        assert MAX_OCR_UPSCALE <= 3.0
