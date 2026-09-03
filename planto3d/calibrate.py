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
import math
import re
from dataclasses import dataclass, replace
from statistics import median

import cv2
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
# What may sit between the feet and the inches. Half the drawing offices in
# the world write 12'-6" and the other half write 12'6", and the hyphenated
# form was silently unreadable -- every dimension on a US or Indian sheet in
# that style, thrown away, on the metric that sets the building's size.
FEET_INCH_JOIN = r"[\s\-‐-―]*"

# Inches are optional. A room written 14' is fourteen feet exactly, and
# demanding the zero lost the whole pair -- "12'-8\" x 14'" parsed as
# nothing at all rather than as one dimension it could read.
DIMENSION_PATTERN = re.compile(
    rf"(\d{{1,3}})\s*{FOOT_MARK}{FEET_INCH_JOIN}(\d{{1,2}})?\s*{INCH_MARK}"
    r"\s*[xX×]\s*"
    rf"(\d{{1,3}})\s*{FOOT_MARK}{FEET_INCH_JOIN}(\d{{1,2}})?"
)

# An area printed on the drawing: "2130 SQ.FT.", "600 SQ. FT.", "125 m2",
# "125 SQM". Indian and Korean residential sheets print these routinely and
# they are the most direct scale reference a drawing can carry -- an area is
# two dimensions at once, and unlike a room's bounding box it does not care
# which way round the label reads.
#
# The unit is required. A bare number on a plan is a room number, a level,
# a door tag or a note, and reading one as an area would resize the whole
# building.
# The superscript two, named rather than typed, so the pattern survives
# every editor and shell it passes through.
SUPERSCRIPT_TWO = "²"

AREA_PATTERN = re.compile(
    r"(\d{1,3}(?:[,\s]\d{3}|\d*)(?:\.\d+)?)\s*"
    r"(sq\s*\.?\s*(?:ft|feet|m|metres|meters)|sqft|sqm"
    r"|m\s*[" + SUPERSCRIPT_TWO + r"2]|ft\s*[" + SUPERSCRIPT_TWO + r"2])",
    re.IGNORECASE,
)

SQUARE_FEET_PER_SQUARE_METRE = 10.7639

# Smallest and largest printed area worth believing, in square feet. Below
# the first a label is more likely a door tag than a room; above the second
# it is a plot or a whole development rather than anything the polygon under
# the text encloses.
MIN_PRINTED_AREA_SQFT = 20.0
MAX_PRINTED_AREA_SQFT = 20000.0

INCHES_PER_FOOT = 12
MIN_CONFIDENCE = 40.0

# Below this, on its longest side, a sheet is enlarged before OCR reads it.
# A drawing that fits a house into 600 pixels prints its dimensions at a
# size Tesseract cannot resolve, and the text is lost rather than misread.
# 1200 is where the measured gain arrives; the cap is there because
# interpolation cannot recover strokes that were never sampled, and three
# times reads no more than two.
MIN_OCR_LONG_EDGE = 1200
MAX_OCR_UPSCALE = 2.0
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
MIN_DOORS_FOR_SCALE = 3

# Room names that place a drawing in a drafting tradition. Only the Nordic
# set is listed, because it is the only tradition whose element sizes can
# be checked: CubiCasa is Finnish residential and carries metric ground
# truth. Most of these words are already recognised by planto3d/features.py,
# which is where they came from; OLOHUONE, MAKUUHUONE and ETEINEN are not
# in that file's keyword lists but are legitimate Finnish room names
# (living room, bedroom, hall) added for this detector specifically.
CONVENTION_KEYWORDS = {
    "nordic": (
        "PARVEKE",
        "TERASSI",
        "KATTOTERASSI",
        "BALKONG",
        "TERRASS",
        "KYLPYHUONE",
        "PESUHUONE",
        "KODINHOITOHUONE",
        "KEITTIO",
        "OLOHUONE",
        "MAKUUHUONE",
        "ETEINEN",
    ),
}

# How many of those words must appear before a sheet is claimed for a
# tradition. One is a misread waiting to happen -- OCR turns "BEDROOM 2"
# into all sorts of things -- and a tradition is a property of the whole
# drawing rather than of one label.
MIN_CONVENTION_HITS = 2

# Element sizes per tradition, as (door_ft, wall_ft).
#
# Nordic's wall thickness is the measured median implied by CubiCasa's own
# ground truth over 30 sheets -- 0.648 ft against the 0.75 assumed
# worldwide. Substituting it alone moves the pooled error from 17.7% to
# 9.9% and the sheets within a fifth from 20/30 to 23/30.
#
# The door stays at the shipped 2.5 ft deliberately. Correcting it to
# CubiCasa's own 2'3" was measured and made things worse -- 20.1% median,
# 15/30 -- because the detector measures an opening span rather than the
# leaf the annotation records.
CONVENTIONS: dict[str, tuple[float, float]] = {
    "nordic": (TYPICAL_DOOR_FT, 0.648),
}


def detect_convention(text_boxes: list) -> str | None:
    """Which drafting tradition a drawing announces, or None if it does not.

    Deliberately conservative. Claiming a tradition changes how big the
    finished building is, so an unrecognised drawing keeps the defaults
    rather than being assigned a best guess.
    """
    words = {
        word
        for box in text_boxes
        for word in "".join(
            character if character.isalnum() else " "
            for character in getattr(box, "text", "").upper()
        ).split()
    }

    for convention, keywords in CONVENTION_KEYWORDS.items():
        hits = len(words & set(keywords))
        if hits >= MIN_CONVENTION_HITS:
            logger.info("drawing reads as %s (%d matching name(s))", convention, hits)
            return convention
    return None


def element_sizes(text_boxes: list) -> tuple[float, float]:
    """The (door_ft, wall_ft) to measure this drawing with."""
    convention = detect_convention(text_boxes)
    if convention is None:
        return TYPICAL_DOOR_FT, TYPICAL_WALL_FT
    return CONVENTIONS[convention]


# How wide a door is, in multiples of the drawing's own wall thickness.
# A 2'6" door against a 9" wall is three and a third; a 2'0" door against
# a 12" wall is two; a double door against a thin partition is six or
# seven. The bounds are set outside all of that, because their job is to
# throw out fragments rather than to judge doors.
#
# The lower bound is the one that matters. A segmenter predicting doors
# eagerly returns slivers a fifth of a door wide, and enough of them drag
# the median below any real door -- which calibrates the whole building
# at a fraction of its size.
MIN_DOOR_GAUGES = 1.5
MAX_DOOR_GAUGES = 8.0
MIN_WALLS_FOR_SCALE = 8
# Words separated by more than this multiple of their height belong to
# different labels. Tesseract assigns one "line" to text at the same
# vertical position however far apart it sits, which on a floor plan merges
# unrelated room labels -- losing one dimension and putting the merged box
# between the two rooms instead of over either.
WORD_GAP_RATIO = 2.0
# Greyscale value below which a pixel counts as lettering. Plan text is drawn
# near-black; hatching, fills and furniture are mid-grey.
INK_CUTOFF = 60


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
    # Inches are optional on either side; absent means exactly that many feet.
    feet_a, inches_a, feet_b, inches_b = (
        int(group) if group else 0 for group in match.groups()
    )
    return (
        feet_a + inches_a / INCHES_PER_FOOT,
        feet_b + inches_b / INCHES_PER_FOOT,
    )


def parse_area_text(text: str) -> float | None:
    """Extract a printed area in square feet from a label, or None.

    Metric areas are converted, so the caller never has to know which unit
    the drawing used.
    """
    match = AREA_PATTERN.search(text)
    if not match:
        return None

    try:
        amount = float(match.group(1).replace(",", "").replace(" ", ""))
    except ValueError:
        return None

    unit = match.group(2).lower().replace(" ", "").replace(".", "")
    metric = unit.endswith(("m", "m2", "m²", "sqm", "metres", "meters"))
    area = amount * SQUARE_FEET_PER_SQUARE_METRE if metric else amount

    if not MIN_PRINTED_AREA_SQFT <= area <= MAX_PRINTED_AREA_SQFT:
        return None
    return area


def _polygon_area_px(room: Room) -> float:
    """The room's own area by the shoelace formula.

    Its bounding box would do for a rectangle and overstate an L-shaped
    room badly, and a terrace or a lounge is very often L-shaped.
    """
    points = room.polygon
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def scale_from_areas(rooms: list[Room], text_boxes: list[TextBox]) -> float | None:
    """Estimate pixels per foot from rooms carrying a printed area.

    An area gives the scale directly: a region of A square feet drawn as
    P square pixels was drawn at sqrt(P/A) pixels per foot. This is the
    relation 3DPlanNet calibrates on (Park & Kim, *Electronics* 2021, 10,
    2729, equation 1), where it reaches 97% scale accuracy on drawings
    that print their areas.

    Returns None when no room can be matched, leaving the caller to fall
    back to something measured off the geometry.
    """
    samples: list[float] = []

    # Matched label to room rather than room to label, because a label
    # belongs to exactly one region while a point on a plan sits inside
    # several: the segmenter emits a polygon per class, so a space read as
    # part bedroom and part generic room yields two overlapping outlines,
    # and a plan can carry ninety of them. Asking each room whether it
    # holds a label took whichever polygon came first in class order,
    # which is arbitrary -- on the reference sheet it charged "2130 SQ.FT."
    # to a region a third of the terrace's size and put the whole building
    # out by a third.
    #
    # The largest containing polygon is the right one. Where outlines
    # nest, the label names the space, not a fragment of it.
    for box in text_boxes:
        area_ft2 = parse_area_text(box.text)
        if area_ft2 is None or area_ft2 <= 0:
            continue

        containing = [room for room in rooms if room.contains(box.centre)]
        if not containing:
            continue

        area_px = max(_polygon_area_px(room) for room in containing)
        if area_px <= 0:
            continue
        samples.append(math.sqrt(area_px / area_ft2))

    if not samples:
        return None

    scale = median(samples)
    logger.info(
        "scale estimated at %.2f px/ft from %d printed area(s)", scale, len(samples)
    )
    return scale


# How far a scale read off the drawing's own text may sit from one measured
# off its geometry before the text is disbelieved.
#
# The geometric estimate is itself only good to about a fifth, so the gate
# has to be looser than that or it would reject correct readings. What it is
# for is the gross failure: a dropped foot mark, a transposed digit, a room
# number read as an area. Those are wrong by multiples, not by a third.
MAX_PRINTED_DISAGREEMENT = 0.4


def corroborated(printed: float, reference: float | None) -> bool:
    """Whether a scale read from text is close enough to a measured one.

    With nothing to check against the text is taken on trust -- it is
    still the best evidence available, and refusing it would leave the
    drawing with no scale at all.
    """
    if reference is None or reference <= 0 or printed <= 0:
        return True

    disagreement = abs(printed - reference) / reference
    if disagreement > MAX_PRINTED_DISAGREEMENT:
        logger.warning(
            "printed scale %.2f px/ft disagrees with %.2f measured off the "
            "drawing by %.0f%%; keeping the measured one",
            printed,
            reference,
            disagreement * 100,
        )
        return False
    return True


def isolate_ink(image: np.ndarray, cutoff: int = INK_CUTOFF) -> np.ndarray:
    """Keep only the near-black lettering, dropping everything lighter.

    Labels printed over hatching are the ones OCR misses, because the hatch
    breaks up the letterforms. Plan text is drawn near-black while hatching,
    fills and furniture are mid-grey, so a hard cutoff erases the background
    and leaves clean glyphs.

    Measured on the reference ground floor: reading the sheet as-is finds 24
    words and three of the seven room names printed there, missing LANDSCAPE,
    PARKING, CHEF and WASH -- every one of them over hatch. After this the
    same sheet yields 71 words and all seven names.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.where(grey < cutoff, 0, 255).astype(np.uint8)


def read_text_boxes(image: np.ndarray, min_confidence: float = MIN_CONFIDENCE) -> list[TextBox]:
    """OCR an image, returning one box per line of text.

    Grouping by line matters: Tesseract emits `15'0"` and `X18'0"` as
    separate words, and neither parses as a dimension alone.

    A small sheet is enlarged first. OCR has a resolution floor and a plan
    that fits a whole house into 600 pixels sits under it: the letters are
    there and unreadable. Measured over 30 sheets averaging 600 px wide,
    enlarging to clear the floor takes the dimension pairs recovered from
    1 to 8, and the sheets yielding any from 1 to 6. Going further buys
    nothing -- at three times it reads the same 8 -- because interpolation
    cannot invent strokes that were never sampled.

    Boxes are reported in the original image's coordinates whatever
    happened here, since callers match them against room polygons drawn in
    those.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    factor = 1.0
    if longest and longest < MIN_OCR_LONG_EDGE:
        factor = min(MIN_OCR_LONG_EDGE / longest, MAX_OCR_UPSCALE)

    read_from = image
    if factor > 1.0:
        read_from = cv2.resize(
            image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
        )
        logger.info("sheet is %d px across; reading text at %.1fx", longest, factor)

    data = pytesseract.image_to_data(
        isolate_ink(read_from), output_type=pytesseract.Output.DICT
    )

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

    if factor > 1.0:
        boxes = [
            replace(
                box,
                bbox=tuple(int(round(v / factor)) for v in box.bbox),
            )
            for box in boxes
        ]

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


def scale_from_doors(
    openings: list,
    typical_width_ft: float = TYPICAL_DOOR_FT,
    gauge: float | None = None,
) -> float | None:
    """Estimate scale from door widths.

    Doors are the most standardised element in a building: a house is mostly
    interior doors at about 2'6", whatever the drawing conventions or the
    language on the sheet. That makes them a scale reference on plans that
    carry no dimensions at all.

    Measured against the reference sheet, whose printed dimensions give
    28.15 px/ft: 23 detected doors have a median width of 68 px, which at
    2'6" gives 27.2 px/ft -- within 4%.

    The median resists the wide main door and any misdetected opening --
    but only up to a point, which is what ``gauge`` is for. Given the
    drawing's wall thickness, anything far narrower than a wall is not a
    door, and slivers like that come through in numbers: on one plan the
    detected widths were 14, 21, 21 and 64 pixels, where a door measures
    about 76. The median landed on 21 and calibrated the building at a
    quarter of its size.
    """
    widths = [o.width for o in openings if o.type == "door" and o.width > 0]
    if gauge:
        plausible = [
            width
            for width in widths
            if gauge * MIN_DOOR_GAUGES <= width <= gauge * MAX_DOOR_GAUGES
        ]
        if len(plausible) < len(widths):
            logger.info(
                "ignoring %d opening(s) too narrow or too wide to be doors",
                len(widths) - len(plausible),
            )
        # Filtered unconditionally. Falling back to the unfiltered widths
        # when too few survive keeps precisely the measurements already
        # judged impossible, and a drawing calibrated from those comes out
        # at a fraction of its size. Better to have no door estimate and
        # fall through to wall thickness.
        widths = plausible

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


def scale_from_gauge(gauge: float, typical_thickness_ft: float = TYPICAL_WALL_FT) -> float:
    """Estimate scale from the drawing's measured wall thickness.

    Preferred over ``scale_from_walls`` wherever the gauge is available.
    Both ask the same question, but the gauge measures the drawn wall
    directly from the mask, while the extracted walls have been through
    orientation filtering and merging first -- and those steps erode. On
    one CubiCasa plan the drawn wall gauges at 20 pixels while the median
    extracted wall reports 10, so the same drawing calibrated at half its
    real size.

    Still the weaker reference, because a plan mixes thin partitions with
    thick external walls and no convention fixes either. Doors are better
    and are tried first.
    """
    scale = gauge / typical_thickness_ft
    logger.info("scale %.2f px/ft from a wall gauge of %.1f px", scale, gauge)
    return scale


def scale_from_known_room(room, width_ft: float, height_ft: float) -> float | None:
    """Pixels per foot, from a room whose real size the user states.

    The only route to scale with no assumption in it. Every other route
    rests on a standard element -- a 2'6" door, a 9" wall -- and the
    residual error is largely those standards not holding: measured
    against ground truth over 30 sheets, real wall thickness runs 0.478
    to 1.176 ft around a median of 0.648, so no constant fits every
    building. A stated size has no such spread.

    Taken from area rather than an edge because a room is rarely drawn
    as the clean rectangle its printed size implies -- a bay, a wardrobe
    recess or a chamfered corner all make one edge disagree with the
    stated width while the area stays close.
    """
    if width_ft <= 0 or height_ft <= 0:
        raise ValueError(
            f"a room's stated size must be positive, got {width_ft} x {height_ft} ft"
        )

    area_px = abs(_polygon_area_px(room))
    if area_px <= 0:
        return None
    return math.sqrt(area_px / (width_ft * height_ft))


def assumed_scale(dpi: int, ratio: float = ASSUMED_DRAWING_RATIO) -> float:
    """Pixels per foot implied by a drafting ratio at a known resolution.

    Used only when the drawing carries no readable dimensions -- a scanned
    sheet, or one whose labels OCR cannot resolve. The model is then correctly
    proportioned but its absolute size is an assumption, which callers must
    say out loud rather than present as measured.
    """
    inches_per_foot_on_paper = INCHES_PER_FOOT / ratio
    return dpi * inches_per_foot_on_paper
