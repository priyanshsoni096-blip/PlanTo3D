"""What a room label means for the 3D model.

Floor plan abbreviations are only loosely standardised, and published guides
cover Anglo-American conventions while saying nothing about the Indian ones
these drawings use -- pooja rooms, OTS, wash areas. Matching is therefore by
keyword and deliberately forgiving: OCR truncates and mangles labels, so
"BALCONY" arrives as "BAL" and "LIFT" as "UFT".

Each category decides how a region is built, not merely how it is coloured:

- ``water``  recessed and blue -- a pool is a hole in the ground, not a mat
- ``lawn``   planted ground cover
- ``paving`` hard landscaping
- ``void``   no floor slab above it, so a double-height space stays open
- ``wet``    tiled interior floor
- ``open``   open to the air, so its edge needs a railing

An unrecognised label falls through to a plain interior room. Guessing from
noise is worse than doing nothing: a misread label would sink a swimming pool
into the middle of a bedroom.
"""

import logging

logger = logging.getLogger(__name__)

# Longer, more specific phrases are tested first so "TERRACE GARDEN" is read
# as planting rather than caught by the bare "TERRACE" rule.
FEATURE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "water",
        (
            "SWIMMING",
            "POOL",
            "JACUZZI",
            "WATER BODY",
            "WATERBODY",
            "FOUNTAIN",
            "PLUNGE",
            "KOI",
        ),
    ),
    (
        "lawn",
        (
            "TERRACE GARDEN",
            "ROOF GARDEN",
            "LANDSCAPE",
            "LANDSCAPING",
            "GARDEN",
            "LAWN",
            "PLANTER",
            "GREEN AREA",
            "SHRUB",
        ),
    ),
    (
        "void",
        (
            "DOUBLE HEIGHT",
            "DOUBLE-HEIGHT",
            "OPEN TO SKY",
            "OPEN TO BELOW",
            "OTS",
            "VOID",
            "SHAFT",
            "DUCT",
            "ATRIUM",
            "SKYLIGHT",
            "CUT OUT",
            "CUTOUT",
        ),
    ),
    (
        "paving",
        (
            "PARKING",
            "DRIVEWAY",
            "DRIVE WAY",
            "CAR PORCH",
            "GARAGE",
            "COURTYARD",
            "PATIO",
            "PATHWAY",
            "WALKWAY",
            "PAVING",
            "SIT OUT",
            "SITOUT",
        ),
    ),
    (
        "open",
        (
            "BALCONY",
            "BAL",
            "TERRACE",
            "DECK",
            "VERANDA",
            "VERANDAH",
            "PORCH",
            "CHAJJA",
        ),
    ),
    (
        "stairs",
        (
            "STAIRCASE",
            "STAIR",
            "STEPS",
            "STEP",
            "LANDING",
        ),
    ),
    (
        "wet",
        (
            "BATHROOM",
            "BATH",
            "TOILET",
            "SHOWER",
            "POWDER",
            "WASH",
            "UTILITY",
            "LAUNDRY",
            "WC",
        ),
    ),
]

# Short marks that are only meaningful as whole words. "UP" and "DN" are the
# standard annotations on a flight of stairs and are often all that is
# printed there, but as substrings they would match GROUP, CUP and DINING.
EXACT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("stairs", ("UP", "DN", "DOWN")),
    ("void", ("OTS",)),
]

# Categories that describe ground rather than an interior floor.
GROUND_COVERS = {"water", "lawn", "paving"}

# How far a size may sit from the name it belongs to, in multiples of the
# name's own text height. A room's name and its dimensions are printed on
# consecutive lines, so a few line heights covers the pairing while keeping
# the next room's label out of reach.
LABEL_PAIRING_LINES = 3.5


def _normalize(label: str) -> str:
    """Upper-case a label and strip the punctuation drafters sprinkle through it.

    Labels arrive as "W.C.", "DRESS/TOILET", "SIT-OUT" and "CHEF'S KITCHEN".
    Matching the raw text misses all of those, so separators collapse to
    single spaces and everything else is dropped.
    """
    cleaned = "".join(
        character if character.isalnum() else " " for character in label.upper()
    )
    return " ".join(cleaned.split())


def classify(label: str) -> str | None:
    """The feature category a room label implies, or None for a plain room."""
    if not label:
        return None

    normalized = _normalize(label)
    squashed = normalized.replace(" ", "")

    for category, keywords in FEATURE_KEYWORDS:
        for keyword in keywords:
            # Checked with and without spaces so "W C" matches "WC" and
            # "SIT OUT" matches "SITOUT", whichever way it was written.
            if keyword in normalized or keyword.replace(" ", "") in squashed:
                return category

    # Whole-word marks are tested last, so a longer phrase always wins: a
    # "DOUBLE HEIGHT" containing no stair mark must not be caught here.
    words = set(normalized.split())
    for category, keywords in EXACT_KEYWORDS:
        if words & set(keywords):
            return category

    return None


def _nearest_dimensions(box, dimension_boxes: list) -> tuple[float, float] | None:
    """The size printed nearest a name, if it is close enough to belong to it.

    Distance is measured in multiples of the name's own text height, so the
    rule holds whatever resolution the sheet was rendered at.
    """
    if not dimension_boxes:
        return None

    line_height = max(box.bbox[3], 1)
    limit = LABEL_PAIRING_LINES * line_height

    best, best_distance = None, limit
    for candidate, dimensions in dimension_boxes:
        dx = candidate.centre[0] - box.centre[0]
        dy = candidate.centre[1] - box.centre[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < best_distance:
            best, best_distance = dimensions, distance

    return best


def regions_from_labels(text_boxes: list, scale: float) -> dict[str, list[list[tuple[float, float]]]]:
    """Build feature regions from dimension labels alone.

    Outdoor areas are the ones the segmentation model is least able to help
    with -- a lawn or a driveway is not a room, so no polygon is produced for
    it -- and colour only covers the part of a bed that was actually hatched.
    But the label states both the name and the size: "LANDSCAPE 49'0\"X13'2\""
    gives 645 sq ft, and the text sits inside the area it names.

    Rectangles are centred on their label. Drafters centre a label in the
    space it describes, so this places the region within a few feet, which is
    far better than omitting it or drawing only the hatched fraction.
    """
    from planto3d.calibrate import parse_dimension_text

    regions: dict[str, list[list[tuple[float, float]]]] = {}
    if scale <= 0:
        return regions

    # A room's name and its size are usually printed on separate lines, and
    # OCR returns them as separate boxes, so neither carries both facts. The
    # size is paired back to the name by proximity.
    dimension_boxes = [
        (box, parsed)
        for box in text_boxes
        if (parsed := parse_dimension_text(box.text)) is not None
    ]

    for box in text_boxes:
        category = classify(box.text)
        if category is None:
            continue

        dimensions = parse_dimension_text(box.text)
        if dimensions is None:
            dimensions = _nearest_dimensions(box, dimension_boxes)
        if dimensions is None:
            continue

        half_width = dimensions[0] * scale / 2
        half_height = dimensions[1] * scale / 2
        centre_x, centre_y = box.centre

        regions.setdefault(category, []).append(
            [
                (centre_x - half_width, centre_y - half_height),
                (centre_x + half_width, centre_y - half_height),
                (centre_x + half_width, centre_y + half_height),
                (centre_x - half_width, centre_y + half_height),
            ]
        )

    if regions:
        logger.info(
            "regions from dimension labels: %s",
            {name: len(items) for name, items in regions.items()},
        )
    return regions


def group_by_feature(rooms: list) -> dict[str, list]:
    """Group rooms by the feature each label implies."""
    grouped: dict[str, list] = {}
    for room in rooms:
        category = classify(room.label)
        if category:
            grouped.setdefault(category, []).append(room)

    if grouped:
        logger.info(
            "features from labels: %s",
            {name: len(rooms) for name, rooms in grouped.items()},
        )
    return grouped
