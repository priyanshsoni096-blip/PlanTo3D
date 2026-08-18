"""Plans arrive as images at least as often as PDFs."""

import cv2
import numpy as np
import pytest

from planto3d.classical import ROOM_FILL, WALL_FILL
from planto3d.pipeline import IMAGE_SUFFIXES, _load_pages, run


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
