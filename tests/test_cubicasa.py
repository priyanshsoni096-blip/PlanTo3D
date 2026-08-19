import numpy as np
import pytest

from planto3d.classes import (
    BACKGROUND,
    BATH,
    BEDROOM,
    DOOR,
    KITCHEN,
    OUTDOOR,
    ROOM,
    WALL,
    WINDOW,
)
from planto3d.cubicasa import class_distribution, sample_paths, svg_to_mask

SVG_HEADER = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'


def _write_svg(tmp_path, body: str, name: str = "model.svg"):
    path = tmp_path / name
    path.write_text(f"{SVG_HEADER}{body}</svg>")
    return path


def _rect(x0, y0, x1, y1) -> str:
    return f'<polygon points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'


class TestSvgToMask:
    def test_maps_each_annotation_class(self, tmp_path):
        svg = _write_svg(
            tmp_path,
            f'<g class="Wall">{_rect(0, 0, 20, 20)}</g>'
            f'<g class="Door">{_rect(30, 0, 50, 20)}</g>'
            f'<g class="Window">{_rect(60, 0, 80, 20)}</g>'
            f'<g class="Space Bedroom">{_rect(0, 40, 20, 60)}</g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert mask[10, 10] == WALL
        assert mask[10, 40] == DOOR
        assert mask[10, 70] == WINDOW
        assert mask[50, 10] == BEDROOM

    def test_room_types_are_kept_apart(self, tmp_path):
        # Room type is the only route to room function on a plan that prints
        # no names, which is most of them. Collapsing these to one class --
        # as this did before -- costs every finish and every railing.
        svg = _write_svg(
            tmp_path,
            f'<g class="Space Bedroom">{_rect(0, 0, 20, 20)}</g>'
            f'<g class="Space Kitchen">{_rect(30, 0, 50, 20)}</g>'
            f'<g class="Space Bath Shower">{_rect(60, 0, 80, 20)}</g>'
            f'<g class="Space Outdoor Balcony">{_rect(0, 30, 20, 50)}</g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert mask[10, 10] == BEDROOM
        assert mask[10, 40] == KITCHEN
        assert mask[10, 70] == BATH
        assert mask[40, 10] == OUTDOOR

    def test_types_grouped_by_what_they_change_share_a_class(self, tmp_path):
        # A sauna is not a bathroom, but both want a floor built to get wet,
        # and that is the only distinction the model makes use of.
        svg = _write_svg(
            tmp_path,
            f'<g class="Space Sauna">{_rect(0, 0, 20, 20)}</g>'
            f'<g class="Space Bath">{_rect(30, 0, 50, 20)}</g>'
            f'<g class="Space Kitchen Kitchenette">{_rect(60, 0, 80, 20)}</g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert mask[10, 10] == BATH
        assert mask[10, 40] == BATH
        assert mask[10, 70] == KITCHEN

    def test_an_unknown_room_type_stays_a_room(self, tmp_path):
        # Dropping a type CubiCasa adds later would punch a hole in the
        # floor. Losing its finish is the smaller failure.
        svg = _write_svg(
            tmp_path, f'<g class="Space SomeTypeAddedLater">{_rect(0, 0, 20, 20)}</g>'
        )

        assert svg_to_mask(svg, (100, 100))[10, 10] == ROOM

    def test_unmapped_categories_become_background(self, tmp_path):
        # Furniture and fixtures must not be mistaken for structure.
        svg = _write_svg(
            tmp_path,
            f'<g class="Furniture Sofa">{_rect(0, 0, 20, 20)}</g>'
            f'<g class="Stairs">{_rect(30, 0, 50, 20)}</g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert (mask == BACKGROUND).all()

    def test_openings_paint_over_the_wall_they_interrupt(self, tmp_path):
        # A door sits inside its wall. Painted in the wrong order the door
        # disappears and the model never learns openings exist.
        svg = _write_svg(
            tmp_path,
            f'<g class="Wall">{_rect(0, 0, 100, 20)}</g>'
            f'<g class="Door">{_rect(40, 0, 60, 20)}</g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert mask[10, 50] == DOOR
        assert mask[10, 10] == WALL

    def test_group_translation_is_applied(self, tmp_path):
        svg = _write_svg(
            tmp_path,
            f'<g class="Wall" transform="translate(50, 50)">{_rect(0, 0, 20, 20)}</g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert mask[60, 60] == WALL
        assert mask[10, 10] == BACKGROUND

    def test_degenerate_polygons_are_ignored(self, tmp_path):
        svg = _write_svg(
            tmp_path,
            '<g class="Wall"><polygon points="10,10 20,20"/></g>'
            '<g class="Wall"><polygon points=""/></g>',
        )

        mask = svg_to_mask(svg, (100, 100))

        assert (mask == BACKGROUND).all()

    def test_mask_shape_and_dtype_match_the_model_contract(self, tmp_path):
        svg = _write_svg(tmp_path, f'<g class="Wall">{_rect(0, 0, 10, 10)}</g>')

        mask = svg_to_mask(svg, (64, 128))

        assert mask.shape == (64, 128)
        assert mask.dtype == np.int64


class TestSamplePaths:
    def _sample(self, root, folder, image_name="F1_scaled.png"):
        directory = root / folder
        directory.mkdir(parents=True)
        (directory / image_name).write_bytes(b"png")
        (directory / "model.svg").write_text(f"{SVG_HEADER}</svg>")
        return directory

    def test_reads_folders_from_a_split_file(self, tmp_path):
        self._sample(tmp_path, "high_quality_architectural/2003")
        self._sample(tmp_path, "colorful/17")
        split = tmp_path / "train.txt"
        split.write_text("/high_quality_architectural/2003/\n/colorful/17/\n")

        pairs = sample_paths(tmp_path, split)

        assert len(pairs) == 2
        assert all(image.is_file() and svg.is_file() for image, svg in pairs)

    def test_falls_back_to_the_original_image_when_scaled_is_absent(self, tmp_path):
        self._sample(tmp_path, "colorful/9", image_name="F1_original.png")
        split = tmp_path / "train.txt"
        split.write_text("/colorful/9/\n")

        (image, _), = sample_paths(tmp_path, split)

        assert image.name == "F1_original.png"

    def test_incomplete_samples_are_skipped_not_fatal(self, tmp_path):
        self._sample(tmp_path, "colorful/1")
        (tmp_path / "colorful/2").mkdir(parents=True)  # no image, no svg
        split = tmp_path / "train.txt"
        split.write_text("/colorful/1/\n/colorful/2/\n\n")

        pairs = sample_paths(tmp_path, split)

        assert len(pairs) == 1


def test_class_distribution_sums_to_one():
    mask = np.array([[WALL, WALL], [ROOM, BACKGROUND]], dtype=np.int64)

    shares = class_distribution(mask)

    assert shares[WALL] == pytest.approx(0.5)
    assert sum(shares.values()) == pytest.approx(1.0)
