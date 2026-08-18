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

# Categories that describe ground rather than an interior floor.
GROUND_COVERS = {"water", "lawn", "paving"}


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
    return None


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
