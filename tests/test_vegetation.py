import cv2
import numpy as np

from planto3d.classical import vegetation_regions
from planto3d.extrude import floors_to_parts
from planto3d.geometry_types import FloorPlan, Wall

SCALE = 20.0
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]

# BGR, matching how the plan draws each ink.
WHITE = (255, 255, 255)
GREEN = (60, 170, 90)
CYAN = (220, 190, 90)
GREY = (200, 200, 200)


def _sheet(size=400):
    return np.full((size, size, 3), WHITE, dtype=np.uint8)


class TestVegetationRegions:
    def test_finds_a_planted_bed(self):
        image = _sheet()
        image[50:200, 50:250] = GREEN

        regions = vegetation_regions(image)

        assert len(regions) == 1
        assert len(regions[0]) >= 3

    def test_scattered_planting_symbols_close_into_one_bed(self):
        # Lawns are drawn as loose tree and shrub symbols, not a solid fill.
        image = _sheet()
        for x in range(60, 240, 22):
            for y in range(60, 190, 22):
                cv2.circle(image, (x, y), 7, GREEN, -1)

        regions = vegetation_regions(image)

        assert len(regions) == 1

    def test_glazing_is_not_mistaken_for_planting(self):
        # Cyan window strips are the only other coloured ink on the sheet.
        image = _sheet()
        image[100:106, 40:360] = CYAN

        assert vegetation_regions(image) == []

    def test_greyscale_drawing_content_is_ignored(self):
        image = _sheet()
        image[50:250, 50:250] = GREY

        assert vegetation_regions(image) == []

    def test_specks_below_the_area_floor_are_ignored(self):
        image = _sheet()
        cv2.circle(image, (100, 100), 4, GREEN, -1)

        assert vegetation_regions(image) == []

    def test_separate_beds_stay_separate(self):
        image = _sheet(600)
        image[50:200, 50:200] = GREEN
        image[400:550, 400:550] = GREEN

        assert len(vegetation_regions(image)) == 2

    def test_a_blank_sheet_has_no_planting(self):
        assert vegetation_regions(_sheet()) == []

    def test_greyscale_input_is_handled_without_error(self):
        assert vegetation_regions(np.full((100, 100), 255, dtype=np.uint8)) == []


class TestPlantingInTheModel:
    def _floor(self, planting=None):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        return FloorPlan(
            walls=walls, footprint=list(FOOTPRINT), planting=planting or []
        )

    def test_planted_regions_become_lawn(self):
        bed = [(500.0, 0.0), (700.0, 0.0), (700.0, 200.0), (500.0, 200.0)]

        parts = floors_to_parts([self._floor([bed])], wall_height_ft=9.0, scale=SCALE)

        assert "lawn" in parts

    def test_a_plan_without_planting_gets_no_lawn(self):
        parts = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)

        assert "lawn" not in parts

    def test_a_planted_terrace_is_not_roofed_over(self):
        # A garden open to the sky must stay open; roofing it hides the
        # planting entirely and the top storey reads as a sealed box.
        bed = [(40.0, 40.0), (360.0, 40.0), (360.0, 260.0), (40.0, 260.0)]

        roofed = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)
        opened = floors_to_parts([self._floor([bed])], wall_height_ft=9.0, scale=SCALE)

        roofed_area = sum(m.area for m in roofed["roof"])
        opened_area = sum(m.area for m in opened["roof"])
        assert opened_area < roofed_area

    def test_planting_covering_the_whole_roof_does_not_erase_the_building(self):
        # A cut that consumes the entire slab must cost only the roof.
        bed = [(-100.0, -100.0), (900.0, -100.0), (900.0, 900.0), (-100.0, 900.0)]

        parts = floors_to_parts([self._floor([bed])], wall_height_ft=9.0, scale=SCALE)

        assert parts["wall"]

    def test_a_roof_garden_sits_at_its_own_storey(self):
        # Terrace planting belongs on the terrace, not on the ground.
        bed = [(50.0, 50.0), (250.0, 50.0), (250.0, 250.0), (50.0, 250.0)]

        parts = floors_to_parts(
            [self._floor(), self._floor(), self._floor([bed])],
            wall_height_ft=9.0,
            scale=SCALE,
        )

        highest_lawn = max(mesh.bounds[1][1] for mesh in parts["lawn"])
        assert highest_lawn > 9.0 * 0.3048
