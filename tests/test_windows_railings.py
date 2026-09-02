import numpy as np
import pytest

from planto3d.classes import BACKGROUND, DOOR, WALL, WINDOW
from planto3d.classical import refine_windows, window_mask
from planto3d.extrude import floors_to_parts
from planto3d.geometry_types import FloorPlan, Room, Wall

SCALE = 20.0
FOOTPRINT = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]

WHITE = (255, 255, 255)
CYAN = (220, 190, 90)  # BGR
GREEN = (60, 170, 90)


def _sheet(size=400):
    return np.full((size, size, 3), WHITE, dtype=np.uint8)


class TestWindowStrips:
    def test_a_cyan_band_is_read_as_a_window(self):
        image = _sheet()
        image[100:106, 60:300] = CYAN

        assert window_mask(image).any()

    def test_round_planting_symbols_are_not_windows(self):
        # Both inks are saturated; only shape separates them.
        import cv2

        image = _sheet()
        for x in range(80, 300, 30):
            cv2.circle(image, (x, 150), 12, GREEN, -1)

        assert not window_mask(image).any()

    def test_a_mullioned_run_closes_into_one_opening(self):
        image = _sheet()
        for start in range(60, 300, 40):
            image[100:106, start : start + 30] = CYAN

        import cv2

        count = cv2.connectedComponentsWithStats(window_mask(image), 8)[0] - 1
        assert count == 1

    def test_a_stubby_mark_is_rejected(self):
        image = _sheet()
        image[100:112, 100:112] = CYAN

        assert not window_mask(image).any()

    def test_greyscale_input_is_handled(self):
        assert not window_mask(np.full((80, 80), 255, np.uint8)).any()


class TestRefineWindows:
    def _mask(self):
        mask = np.full((400, 400), BACKGROUND, dtype=np.int64)
        mask[98:108, 0:400] = WALL
        mask[98:108, 20:60] = WINDOW  # the model's guess, in the wrong place
        mask[98:108, 300:340] = DOOR
        return mask

    def test_colour_replaces_the_model_s_windows(self):
        # The model scores 0.12 IoU on windows and finds roughly twice as
        # many blobs as there are windows. Where a drawing clearly marks them
        # in colour, that is the better source.
        image = _sheet()
        for start in range(150, 400, 60):
            image[100:106, start : start + 40] = CYAN

        refined = refine_windows(self._mask(), image)

        assert refined[102, 170] == WINDOW  # found by colour
        assert refined[102, 40] != WINDOW  # the model's guess dropped

    def test_doors_are_left_alone(self):
        image = _sheet()
        image[100:106, 150:260] = CYAN

        refined = refine_windows(self._mask(), image)

        assert refined[102, 320] == DOOR

    def test_a_sheet_without_coloured_windows_keeps_the_model_s(self):
        # Nothing to substitute; a weak guess beats none.
        refined = refine_windows(self._mask(), _sheet())

        assert refined[102, 40] == WINDOW

    def test_one_stray_coloured_mark_does_not_discard_the_model_s_windows(self):
        # A sheet using the convention shows glazing on every elevation. One
        # mark is not evidence of a convention, and treating it as such threw
        # away everything the model found on plans that never used colour.
        image = _sheet()
        image[100:106, 150:230] = CYAN  # a single strip

        refined = refine_windows(self._mask(), image)

        assert refined[102, 40] == WINDOW  # the model's kept
        assert refined[102, 190] == WINDOW  # and the coloured one added

    def test_a_sheet_that_clearly_uses_colour_overrides_the_model(self):
        image = _sheet()
        for start in range(60, 380, 80):
            image[100:106, start : start + 50] = CYAN

        refined = refine_windows(self._mask(), image)

        assert refined[102, 40] != WINDOW  # the model's guess dropped

    def test_walls_survive_refinement(self):
        image = _sheet()
        image[100:106, 150:260] = CYAN

        refined = refine_windows(self._mask(), image)

        assert refined[102, 380] == WALL


class TestRailings:
    def _floor(self, rooms=None):
        walls = [
            Wall(start=FOOTPRINT[i], end=FOOTPRINT[(i + 1) % 4], thickness=10.0)
            for i in range(4)
        ]
        return FloorPlan(walls=walls, footprint=list(FOOTPRINT), rooms=rooms or [])

    def test_a_balcony_gets_a_railing_in_the_model(self):
        balcony = Room(
            polygon=[(420.0, 0.0), (520.0, 0.0), (520.0, 120.0), (420.0, 120.0)],
            label="BALCONY",
        )

        parts = floors_to_parts(
            [self._floor(), self._floor([balcony])], wall_height_ft=9.0, scale=SCALE
        )

        assert "railing" in parts

    def test_a_plan_without_balconies_gets_no_railings(self):
        parts = floors_to_parts([self._floor()], wall_height_ft=9.0, scale=SCALE)

        assert "railing" not in parts

    def test_a_railing_is_waist_high_not_full_height(self):
        balcony = Room(polygon=list(FOOTPRINT), label="BALCONY")

        parts = floors_to_parts([self._floor([balcony])], wall_height_ft=9.0, scale=SCALE)

        rail_height = max(m.bounds[1][1] for m in parts["railing"]) - min(
            m.bounds[0][1] for m in parts["railing"]
        )
        assert rail_height < 9.0 * 0.3048
