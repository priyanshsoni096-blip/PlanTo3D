"""Read the drawing's own text to convert pixels into real-world feet.

Architectural sheets label each room with its dimensions, so the drawing
carries its own scale. Pairing a room's pixel size with its printed feet
gives pixels-per-foot without needing a scale bar or assumed DPI.

OCR is noisy: it splits a dimension across word tokens, swaps straight
quotes for curly ones, and drops inch marks. Text is therefore rejoined by
line before parsing, the pattern is written loosely, and the final scale is
a median across every labelled room so a single misread cannot skew it.
"""

import logging
import re
from dataclasses import dataclass
from statistics import median

import numpy as np
import pytesseract

from planto3d.geometry_types import Room
from planto3d.tools import configure_tesseract

logger = logging.getLogger(__name__)

configure_tesseract()

# Feet-and-inches pairs such as 15'0"X18'0", tolerating curly quotes, missing
# inch marks, stray spaces, and either case of the separating x. Degree signs
# are accepted as foot and inch marks because Tesseract substitutes them
# routinely -- on the reference first-floor sheet that alone was the
# difference between a dimension parsing and being discarded.
FOOT_MARK = r"['’´°]+"
INCH_MARK = r"[\"”“'’°]*"
DIMENSION_PATTERN = re.compile(
    rf"(\d{{1,3}})\s*{FOOT_MARK}\s*(\d{{1,2}})\s*{INCH_MARK}"
    r"\s*[xX×]\s*"
    rf"(\d{{1,3}})\s*{FOOT_MARK}\s*(\d{{1,2}})"
)

INCHES_PER_FOOT = 12
MIN_CONFIDENCE = 40.0
# Fallback when the drawing carries no readable dimensions. Residential plans
# are typically drafted around 1:150, so at a known rasterization resolution
# the scale follows from the ratio rather than being invented: on the
# reference sheet this gives 32 px/ft against 28.15 measured, close enough
# for a model of believable size while clearly flagged as assumed.
ASSUMED_DRAWING_RATIO = 150.0
# Standard element sizes used to recover scale when a drawing carries no
# dimensions. A house is mostly interior doors at about 2'6", so the median
# door is a far better reference than the widest one. Wall thickness is
# weaker: a plan mixes thin partitions with thick external walls.
TYPICAL_DOOR_FT = 2.5
TYPICAL_WALL_FT = 0.75  # 9 inches
# Too few of either and the median means nothing.
MIN_DOORS_FOR_SCALE = 4
MIN_WALLS_FOR_SCALE = 8
# Words separated by more than this multiple of their height belong to
# different labels. Tesseract assigns one "line" to text at the same
# vertical position however far apart it sits, which on a floor plan merges
# unrelated room labels -- losing one dimension and putting the merged box
# between the two rooms instead of over either.
WORD_GAP_RATIO = 2.0


@dataclass(frozen=True)
class TextBox:
    """A line of text OCR found, with where it sits on the page."""

    text: str
    bbox: tuple[int, int, int, int]  # left, top, width, height
    confidence: float

    @property
    def centre(self) -> tuple[float, float]:
        left, top, width, height = self.bbox
        return (left + width / 2, top + height / 2)


def parse_dimension_text(text: str) -> tuple[float, float] | None:
    """Extract a (feet, feet) pair from a dimension label, or None."""
    match = DIMENSION_PATTERN.search(text)
    if not match:
        return None
    feet_a, inches_a, feet_b, inches_b = (int(g) for g in match.groups())
    return (
        feet_a + inches_a / INCHES_PER_FOOT,
        feet_b + inches_b / INCHES_PER_FOOT,
    )


def read_text_boxes(image: np.ndarray, min_confidence: float = MIN_CONFIDENCE) -> list[TextBox]:
    """OCR an image, returning one box per line of text.

    Grouping by line matters: Tesseract emits `15'0"` and `X18'0"` as
    separate words, and neither parses as a dimension alone.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        if float(data["conf"][i]) < min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    boxes = []
    for indices in lines.values():
        for group in _split_on_gaps(data, indices):
            boxes.append(_to_text_box(data, group))

    logger.info("read %d text line(s)", len(boxes))
    return boxes


def _split_on_gaps(data: dict, indices: list[int]) -> list[list[int]]:
    """Break one OCR line into groups of words that actually belong together."""
    ordered = sorted(indices, key=lambda i: data["left"][i])

    groups: list[list[int]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        gap = data["left"][current] - (data["left"][previous] + data["width"][previous])
        allowed = WORD_GAP_RATIO * max(data["height"][previous], data["height"][current])
        if gap > allowed:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def _to_text_box(data: dict, indices: list[int]) -> TextBox:
    lefts = [data["left"][i] for i in indices]
    tops = [data["top"][i] for i in indices]
    rights = [data["left"][i] + data["width"][i] for i in indices]
    bottoms = [data["top"][i] + data["height"][i] for i in indices]

    return TextBox(
        text=" ".join(data["text"][i].strip() for i in indices),
        bbox=(min(lefts), min(tops), max(rights) - min(lefts), max(bottoms) - min(tops)),
        confidence=float(min(float(data["conf"][i]) for i in indices)),
    )


def estimate_scale(rooms: list[Room], text_boxes: list[TextBox]) -> float | None:
    """Estimate pixels per foot from rooms carrying dimension labels.

    Sheets in one drawing set share a scale, so rooms and text from several
    floors can be pooled into a single call. That matters for sparsely
    labelled floors: the reference terrace sheet yields only one dimension on
    its own, against eight from the ground floor.

    Returns None when no room can be measured, leaving the caller to decide
    rather than guessing a scale.
    """
    samples: list[float] = []

    for room in rooms:
        dimensions = next(
            (
                parsed
                for box in text_boxes
                if room.contains(box.centre) and (parsed := parse_dimension_text(box.text))
            ),
            None,
        )
        if dimensions is None:
            continue

        left, top, right, bottom = room.bounds()
        pixel_sides = sorted((right - left, bottom - top))
        feet_sides = sorted(dimensions)
        if min(pixel_sides) <= 0 or min(feet_sides) <= 0:
            continue

        # The label does not say which dimension runs which way, so pair the
        # long side with the long dimension and the short with the short.
        samples.extend(px / ft for px, ft in zip(pixel_sides, feet_sides))

    if not samples:
        logger.warning("no room could be matched to a dimension label; scale unknown")
        return None

    scale = median(samples)
    logger.info("scale estimated at %.2f px/ft from %d measurement(s)", scale, len(samples))
    return scale


def scale_from_doors(openings: list, typical_width_ft: float = TYPICAL_DOOR_FT) -> float | None:
    """Estimate scale from door widths.

    Doors are the most standardised element in a building: a house is mostly
    interior doors at about 2'6", whatever the drawing conventions or the
    language on the sheet. That makes them a scale reference on plans that
    carry no dimensions at all.

    Measured against the reference sheet, whose printed dimensions give
    28.15 px/ft: 23 detected doors have a median width of 68 px, which at
    2'6" gives 27.2 px/ft -- within 4%.

    The median resists the wide main door and any misdetected opening.
    """
    widths = [o.width for o in openings if o.type == "door" and o.width > 0]
    if len(widths) < MIN_DOORS_FOR_SCALE:
        return None

    scale = median(widths) / typical_width_ft
    logger.info(
        "scale %.2f px/ft from %d door(s), median %.1f px", scale, len(widths), median(widths)
    )
    return scale


def scale_from_walls(walls: list, typical_thickness_ft: float = TYPICAL_WALL_FT) -> float | None:
    """Estimate scale from wall thickness.

    Weaker than doors, because a plan mixes thin partitions with thick
    external walls and the median lands somewhere between. Useful only when
    no doors were found -- on the reference sheet it lands within about 11%.
    """
    thicknesses = [w.thickness for w in walls if w.thickness > 0]
    if len(thicknesses) < MIN_WALLS_FOR_SCALE:
        return None

    scale = median(thicknesses) / typical_thickness_ft
    logger.info("scale %.2f px/ft from %d wall thickness(es)", scale, len(thicknesses))
    return scale


def assumed_scale(dpi: int, ratio: float = ASSUMED_DRAWING_RATIO) -> float:
    """Pixels per foot implied by a drafting ratio at a known resolution.

    Used only when the drawing carries no readable dimensions -- a scanned
    sheet, or one whose labels OCR cannot resolve. The model is then correctly
    proportioned but its absolute size is an assumption, which callers must
    say out loud rather than present as measured.
    """
    inches_per_foot_on_paper = INCHES_PER_FOOT / ratio
    return dpi * inches_per_foot_on_paper
