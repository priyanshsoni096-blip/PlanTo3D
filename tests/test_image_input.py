"""Plans arrive as images at least as often as PDFs."""

import cv2
import numpy as np
import pytest

from planto3d.classical import ROOM_FILL, WALL_FILL
from planto3d.ingest import read_image
from planto3d.pipeline import IMAGE_SUFFIXES, _load_pages, extract, run


def _plan_image(size=320):
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[40:280, 40:280] = WALL_FILL
    image[52:268, 52:268] = ROOM_FILL
    return image


class TestLoadPages:
    def test_a_single_image_is_taken_as_one_page(self, tmp_path):
        path = tmp_path / "plan.png"
        cv2.imwrite(str(path), _plan_image())

        assert _load_pages(path, tmp_path / "pages") == [path]

    def test_a_directory_becomes_one_page_per_image_in_order(self, tmp_path):
        folder = tmp_path / "floors"
        folder.mkdir()
        for name in ("3-terrace.png", "1-ground.png", "2-first.png"):
            cv2.imwrite(str(folder / name), _plan_image())

        pages = _load_pages(folder, tmp_path / "pages")

        assert [p.name for p in pages] == ["1-ground.png", "2-first.png", "3-terrace.png"]

    def test_non_image_files_in_a_directory_are_ignored(self, tmp_path):
        folder = tmp_path / "floors"
        folder.mkdir()
        cv2.imwrite(str(folder / "plan.png"), _plan_image())
        (folder / "notes.txt").write_text("not a plan")

        assert len(_load_pages(folder, tmp_path / "pages")) == 1

    def test_an_empty_directory_is_an_error(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()

        with pytest.raises(ValueError):
            _load_pages(folder, tmp_path / "pages")

    def test_common_image_formats_are_accepted(self):
        assert {".png", ".jpg", ".jpeg"} <= IMAGE_SUFFIXES


class TestRunFromImage:
    def test_a_single_image_produces_a_model(self, tmp_path):
        source = tmp_path / "plan.png"
        cv2.imwrite(str(source), _plan_image())

        result = run(source, tmp_path / "out", crop=False)

        assert result.model_path is not None
        assert result.model_path.exists()
        assert result.wall_count > 0

    def test_cropping_is_skipped_for_a_lone_image(self, tmp_path):
        # The crop looks for borders common to every page. On one tight image
        # there are none to find, so it could only take content away.
        source = tmp_path / "plan.png"
        cv2.imwrite(str(source), _plan_image())

        result = run(source, tmp_path / "out")

        assert result.wall_count > 0

    def test_scale_falls_back_when_an_image_has_no_dimensions(self, tmp_path):
        # A bare plan image carries no printed dimensions; the model must
        # still be built, and must say the size was inferred.
        source = tmp_path / "plan.png"
        cv2.imwrite(str(source), _plan_image())

        result = run(source, tmp_path / "out", crop=False)

        assert result.scale is not None
        assert result.scale_assumed
        assert result.scale_source in {"doors", "walls", "ratio"}


class TestSplitIsIgnoredOnceAlreadyMultiPage:
    # ``split`` only has a single sheet to act on. A directory of images is
    # already one file per storey -- there is nothing left for --split N to
    # cut -- so extract() must not silently pretend it applied.
    def test_a_directory_of_images_ignores_split_and_warns(self, tmp_path, caplog):
        folder = tmp_path / "floors"
        folder.mkdir()
        for name in ("1-ground.png", "2-first.png"):
            cv2.imwrite(str(folder / name), _plan_image())

        with caplog.at_level("WARNING"):
            result = extract(folder, tmp_path / "out", crop=False, split=2)

        assert len(result.floors) == 2  # unaffected: one floor per image
        assert any("ignored" in record.message for record in caplog.records)

    def test_a_single_image_still_honours_split(self, tmp_path, caplog):
        # The other half of the contract: split=1 on a lone sheet is
        # meaningful (nothing to split) and must not warn.
        source = tmp_path / "plan.png"
        cv2.imwrite(str(source), _plan_image())

        with caplog.at_level("WARNING"):
            result = extract(source, tmp_path / "out", crop=False, split=1)

        assert len(result.floors) == 1
        assert not any("ignored" in record.message for record in caplog.records)


class TestReversedPrints:
    """A blueprint carries the same drawing with its tones the other way up.

    The segmenter has never seen one: reversing 15 sheets took wall IoU
    from 0.747 to 0.014 and left 4 of them reconstructable. It is the most
    damaging thing that can be done to a drawing without changing a line
    of it, and the cheapest to undo.
    """

    def _plan(self):
        sheet = np.full((400, 500, 3), 255, dtype=np.uint8)
        sheet[60:340, 80:420] = 30
        sheet[80:320, 100:400] = 255
        return sheet

    def test_a_light_on_dark_sheet_is_turned_the_right_way_up(self, tmp_path):
        path = tmp_path / "blueprint.png"
        cv2.imwrite(str(path), 255 - self._plan())

        read = read_image(path)

        assert read is not None
        # Paper is light again, and the drawing survives the turn.
        assert np.median(cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)) > 128
        assert np.array_equal(read, self._plan())

    def test_an_ordinary_sheet_is_left_exactly_alone(self, tmp_path):
        path = tmp_path / "ordinary.png"
        plan = self._plan()
        cv2.imwrite(str(path), plan)

        assert np.array_equal(read_image(path), plan)

    def test_a_heavily_inked_sheet_is_not_mistaken_for_a_reversed_one(self, tmp_path):
        # A dense drawing is still mostly paper. The two populations are far
        # apart -- the lowest ordinary median across 66 sheets is 195 and the
        # highest reversed one is 60 -- and this guards the gap between them.
        sheet = np.full((400, 500, 3), 255, dtype=np.uint8)
        sheet[:, ::3] = 20           # ink over a third of the sheet
        path = tmp_path / "dense.png"
        cv2.imwrite(str(path), sheet)

        assert np.array_equal(read_image(path), sheet)
