"""Regions built from a printed label have to fit on the page they came from.

Found by sweeping the BRIDGE sheets: 116794 is 498 x 372 px, and a region
built from one of its labels came out 336 x 568 px -- taller than the sheet
-- which then cut a hole in the roof 276% of the building's own footprint.

A label states a size; the scale converting it to pixels is an estimate,
and the dimension text itself can be misread. Neither is reliable enough to
draw past the edge of the drawing.
"""

from planto3d.features import regions_from_labels


class Box:
    """The shape read_text_boxes returns, as much of it as this needs."""

    def __init__(self, text, centre, bbox=(0, 0, 60, 12)):
        self.text = text
        self.centre = centre
        self.bbox = bbox


def _extent(region):
    xs = [point[0] for point in region]
    ys = [point[1] for point in region]
    return max(xs) - min(xs), max(ys) - min(ys)


def test_a_region_that_fits_is_drawn_at_its_stated_size():
    boxes = [Box("LANDSCAPE 20'0\" X 10'0\"", centre=(400, 300))]

    regions = regions_from_labels(boxes, scale=10.0, page_size=(800, 600))

    assert _extent(regions["lawn"][0]) == (200, 100)


def test_a_region_running_off_the_page_is_cut_back_to_it():
    # The label sits near the edge, so half its rectangle would hang off.
    boxes = [Box("LANDSCAPE 40'0\" X 10'0\"", centre=(60, 300))]

    regions = regions_from_labels(boxes, scale=10.0, page_size=(800, 600))
    region = regions["lawn"][0]

    assert min(point[0] for point in region) >= 0
    assert max(point[0] for point in region) <= 800
    assert _extent(region)[1] == 100          # the side that fitted is untouched


def test_a_region_larger_than_the_whole_sheet_is_dropped():
    # 100 ft by 80 ft at 16 px/ft is 1600 x 1280 on a 498 x 372 sheet: the
    # label was misread, or the scale is far out, and either way the region
    # says nothing about where anything is.
    boxes = [Box("LANDSCAPE 100'0\" X 80'0\"", centre=(250, 180))]

    regions = regions_from_labels(boxes, scale=16.0, page_size=(498, 372))

    assert regions.get("lawn", []) == []


def test_without_a_page_size_nothing_is_clipped():
    # Callers that do not know the page keep the old behaviour rather than
    # silently losing regions.
    boxes = [Box("LANDSCAPE 40'0\" X 10'0\"", centre=(60, 300))]

    regions = regions_from_labels(boxes, scale=10.0)

    assert min(point[0] for point in regions["lawn"][0]) < 0
