import pytest

from planto3d.extrude import floors_to_parts
from planto3d.geometry_types import FloorPlan, Room, Wall
from planto3d.site import boundary_walls, classify_cover, outdoor_rooms, site_outline

SCALE = 20.0
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def _room(label, offset=0.0, size=80.0):
    return Room(
        polygon=[
            (offset, offset),
            (offset + size, offset),
            (offset + size, offset + size),
            (offset, offset + size),
        ],
        label=label,
    )


class TestClassifyCover:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("LANDSCAPE", "lawn"),
            ("TERRACE GARDEN", "lawn"),
            ("PARKING", "paving"),
            ("VERANDAH", "paving"),
            ("DECK SEATING", "paving"),
            ("6'4\" WIDE PASSAGE", "paving"),
        ],
    )
    def test_recognises_outdoor_labels(self, label, expected):
        assert classify_cover(label) == expected

    @pytest.mark.parametrize("label", ["BEDROOM", "KITCHEN", "TEMPLE", "DRESS/TOILET"])
    def test_indoor_rooms_get_no_ground_cover(self, label):
        assert classify_cover(label) is None

    @pytest.mark.parametrize("label", ["", "PLAV-AREA", "UFT", "5 GE"])
    def test_unrecognised_or_misread_labels_fall_back_to_nothing(self, label):
        # OCR mangles labels; guessing a cover from noise would paint lawn
        # across the middle of the house.
        assert classify_cover(label) is None

    def test_matching_is_case_insensitive(self):
        assert classify_cover("Landscape") == "lawn"


class TestOutdoorRooms:
    def test_groups_rooms_by_the_cover_they_imply(self):
        grouped = outdoor_rooms(
            [_room("LANDSCAPE"), _room("PARKING", 200), _room("BEDROOM", 400)]
        )

        assert set(grouped) == {"lawn", "paving"}
        assert len(grouped["lawn"]) == 1

    def test_a_plan_with_no_outdoor_rooms_yields_nothing(self):
        assert outdoor_rooms([_room("BEDROOM"), _room("KITCHEN", 200)]) == {}


class TestPlotFromTheDrawing:
    def test_the_page_extent_is_preferred_over_a_margin(self):
        # The sheet is cropped to the drawing frame, and that frame encloses
        # the whole site -- setbacks, driveway and garden. It is the plot,
        # measured rather than assumed.
        outline = site_outline([FOOTPRINT], margin_px=50.0, page_size=(2275, 1570))

        assert max(p[0] for p in outline) == pytest.approx(2275, abs=10)
        assert max(p[1] for p in outline) == pytest.approx(1570, abs=10)

    def test_the_plot_is_inset_from_the_sheet_edge(self):
        outline = site_outline([], margin_px=0.0, page_size=(1000, 800))

        assert min(p[0] for p in outline) > 0
        assert max(p[0] for p in outline) < 1000

    def test_a_boundary_wall_runs_right_around_the_plot(self):
        outline = site_outline([], margin_px=0.0, page_size=(1000, 800))

        walls = boundary_walls(outline, thickness_px=16.0)

        assert len(walls) == 4
        assert all(wall.length() > 0 for wall in walls)

    def test_a_degenerate_outline_yields_no_boundary(self):
        assert boundary_walls([(0.0, 0.0), (1.0, 1.0)], thickness_px=16.0) == []


class TestSiteOutline:
    def test_encloses_every_floor_with_a_margin(self):
        outline = site_outline([FOOTPRINT], margin_px=50.0)

        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        assert min(xs) == pytest.approx(-50)
        assert max(xs) == pytest.approx(450)
        assert min(ys) == pytest.approx(-50)
        assert max(ys) == pytest.approx(350)

    def test_covers_the_largest_floor_when_they_differ(self):
        small = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]

        outline = site_outline([small, FOOTPRINT], margin_px=0.0)

        assert max(p[0] for p in outline) == pytest.approx(400)

    def test_no_footprint_means_no_site(self):
        assert site_outline([], margin_px=50.0) == []
        assert site_outline([[]], margin_px=50.0) == []


class TestSiteInTheModel:
    def _floor(self, rooms=None):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        return FloorPlan(walls=walls, footprint=list(FOOTPRINT), rooms=rooms or [])

    def test_the_building_gets_ground_to_stand_on(self):
        parts = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)

        assert "ground" in parts

    def test_labelled_outdoor_rooms_become_lawn_and_paving(self):
        parts = floors_to_parts(
            [self._floor([_room("LANDSCAPE", 20), _room("PARKING", 200)])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert "lawn" in parts
        assert "paving" in parts

    def test_indoor_rooms_do_not_become_ground_cover(self):
        parts = floors_to_parts(
            [self._floor([_room("BEDROOM", 20), _room("KITCHEN", 200)])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        assert "lawn" not in parts
        assert "paving" not in parts

    def test_the_ground_sits_below_the_building(self):
        parts = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)

        ground_top = max(mesh.bounds[1][1] for mesh in parts["ground"])
        building_base = min(mesh.bounds[0][1] for mesh in parts["floor"])

        assert ground_top <= building_base + 1e-6

    def test_the_plot_gets_a_boundary_wall(self):
        parts = floors_to_parts(
            [self._floor()], wall_height_ft=9.0, scale=SCALE, page_size=(800, 600)
        )

        assert "boundary" in parts

    def test_the_boundary_stands_lower_than_the_house(self):
        parts = floors_to_parts(
            [self._floor()], wall_height_ft=9.0, scale=SCALE, page_size=(800, 600)
        )

        boundary_top = max(mesh.bounds[1][1] for mesh in parts["boundary"])
        house_top = max(mesh.bounds[1][1] for mesh in parts["roof"])

        assert boundary_top < house_top

    def test_the_ground_extends_past_the_walls(self):
        parts = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)

        ground_width = parts["ground"][0].bounds[1][0] - parts["ground"][0].bounds[0][0]
        wall_width = max(m.bounds[1][0] for m in parts["wall"]) - min(
            m.bounds[0][0] for m in parts["wall"]
        )

        assert ground_width > wall_width
