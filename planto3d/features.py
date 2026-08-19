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
            "SWIMMING POOL",
            "INFINITY POOL",
            "PLUNGE POOL",
            "LAP POOL",
            "SPLASH POOL",
            "WADING POOL",
            "REFLECTING POOL",
            "REFLECTION POOL",
            "NATURAL POOL",
            "THERAPY POOL",
            "SPA POOL",
            "HYDROTHERAPY",
            "WATER COURT",
            "WATER GARDEN",
            "FOUNTAIN COURT",
            "HOT TUB",
            "SWIMMING",
            "POOL",
            "JACUZZI",
            "WATER BODY",
            "WATERBODY",
            "WATER FEATURE",
            "FOUNTAIN",
            "POND",
            "PLUNGE",
            "KOI",
        ),
    ),
    (
        "lawn",
        (
            # Landscaped terraces go by several names and all of them are
            # planting rather than bare decking.
            "TERRACE GARDEN",
            "GARDEN TERRACE",
            "ROOF GARDEN",
            "ROOFTOP GARDEN",
            "SKY GARDEN",
            "COURTYARD GARDEN",
            "GREEN COURT",
            "FRONT GARDEN",
            "BACK GARDEN",
            "BACKYARD",
            "PRIVATE GARDEN",
            "COMMON GARDEN",
            "LANDSCAPE",
            "LANDSCAPING",
            "SOFTSCAPE",
            "GARDEN",
            "LAWN",
            "FLOWER BED",
            "RAISED PLANTER",
            "PLANTER",
            "GREEN WALL",
            "VERTICAL GARDEN",
            "LIVING WALL",
            "HANGING GARDEN",
            "TERRACED GARDEN",
            "GREEN TERRACE",
            "GREEN BALCONY",
            "GREEN ROOF",
            "GREEN SPINE",
            "GREEN BUFFER",
            "LANDSCAPE BUFFER",
            "ZEN GARDEN",
            "MEDITATION GARDEN",
            "SENSORY GARDEN",
            "BUTTERFLY GARDEN",
            "SECRET GARDEN",
            "SUNKEN GARDEN",
            "ROCK GARDEN",
            "RAIN GARDEN",
            "TROPICAL GARDEN",
            "NATIVE GARDEN",
            "FORMAL GARDEN",
            "POCKET GARDEN",
            "KITCHEN GARDEN",
            "HERB GARDEN",
            "VEGETABLE GARDEN",
            "SCULPTURE GARDEN",
            "INNER GARDEN",
            "ENTRANCE GARDEN",
            "ARRIVAL GARDEN",
            "SHARED GARDEN",
            "GARDEN COURT",
            "GARDEN PATH",
            "PALM COURT",
            "PALM GARDEN",
            "TREE COURT",
            "TREE GROVE",
            "TREE PIT",
            "PLANTING BED",
            "PLANTING STRIP",
            "PLANTING ZONE",
            "PLANTER BOX",
            "FRONT YARD",
            "REAR YARD",
            "SIDE YARD",
            "ORCHARD",
            "GROVE",
            "MEADOW",
            "BIOSWALE",
            "HEDGE",
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
            "TRIPLE HEIGHT",
            "TRIPLE-HEIGHT",
            "OPEN TO BELOW",
            "LIGHT WELL",
            "LIGHTWELL",
            "LIGHT COURT",
            "AIRWELL",
            "AIR WELL",
            "AIR COURT",
            "VENTILATION COURT",
            "VENTILATION SHAFT",
            "WIND COURT",
            "SUNKEN COURT",
            "SERVICE SHAFT",
            "PLUMBING SHAFT",
            "ELECTRICAL SHAFT",
            "MEP SHAFT",
            "UTILITY SHAFT",
            "LIFT SHAFT",
            "OTB",
            "OTS",
            "VOID",
            "SHAFT",
            "DUCT",
            "ATRIUM",
            "OCULUS",
            "CUT OUT",
            "CUTOUT",
        ),
    ),
    (
        "paving",
        (
            # Parking in all its forms.
            "BASEMENT PARKING",
            "STILT PARKING",
            "COVERED PARKING",
            "OPEN PARKING",
            "VISITOR PARKING",
            "PARKING BAY",
            "EV PARKING",
            "PARKING",
            "CAR PORCH",
            "CARPORT",
            "CAR PORT",
            "GARAGE",
            "DRIVEWAY",
            "DRIVE WAY",
            "DROP OFF",
            "TURNING",
            "ARRIVAL COURT",
            "LOADING",
            "SERVICE BAY",
            # Paved ground. A patio is at ground level and paved; a deck is a
            # raised platform and is treated as open instead.
            "COURTYARD",
            "PATIO",
            "PATHWAY",
            "WALKWAY",
            "JOGGING",
            "PAVING",
            "HARDSCAPE",
            "MOTOR COURT",
            "FORECOURT",
            "ENTRY DRIVE",
            "ARRIVAL DRIVE",
            "TURNING COURT",
            "TURNING CIRCLE",
            "PARKING COURT",
            "GARAGE COURT",
            "PODIUM PARKING",
            "SURFACE PARKING",
            "GUEST PARKING",
            "PRIVATE PARKING",
            "TANDEM PARKING",
            "VALET",
            "PARKING RAMP",
            "BICYCLE PARKING",
            "MOTORCYCLE PARKING",
            "SCOOTER PARKING",
            "ENTRANCE COURT",
            "ENTRY COURT",
            "ARRIVAL PLAZA",
            "GATE COURT",
            "PROMENADE",
            "STEPPING STONES",
            "GARDEN WALK",
            "COVERED WALKWAY",
            "SERVICE COURT",
            "UTILITY YARD",
            "UTILITY COURT",
            "DRYING YARD",
            "WASH YARD",
            "LAUNDRY YARD",
            "SIT OUT",
            "SITOUT",
            "SERVICE YARD",
            # A verandah is a covered space at ground level, not an elevated
            # balcony. Railing one puts a balustrade across the front door.
            "VERANDA",
            "VERANDAH",
            "PORCH",
        ),
    ),
    (
        "open",
        (
            "SERVICE BALCONY",
            "PRIVATE TERRACE",
            "COMMON TERRACE",
            "PENTHOUSE TERRACE",
            "SKY TERRACE",
            "PRIVATE DECK",
            "POOL DECK",
            # Every balcony variant, since a projecting platform always needs
            # a railing whatever the brochure calls it.
            "JULIET BALCONY",
            "FRENCH BALCONY",
            "WRAPAROUND BALCONY",
            "CANTILEVERED BALCONY",
            "CORNER BALCONY",
            "UTILITY BALCONY",
            "PRIVATE BALCONY",
            "SHARED BALCONY",
            "INTERNAL BALCONY",
            # Decks and terraces by any name.
            "ROOF DECK",
            "SUN DECK",
            "GARDEN DECK",
            "VIEWING DECK",
            "OBSERVATION DECK",
            "SCENIC DECK",
            "LOUNGE DECK",
            "BEACH DECK",
            "WATERFRONT DECK",
            "AMENITY DECK",
            "LIFESTYLE DECK",
            "SUN TERRACE",
            "SUNSET TERRACE",
            "VIEWING TERRACE",
            "PANORAMIC TERRACE",
            "ROOFTOP TERRACE",
            "ROOF TERRACE",
            "OPEN TERRACE",
            "COVERED TERRACE",
            "POCKET TERRACE",
            "SHARED TERRACE",
            "AMENITY TERRACE",
            "PODIUM TERRACE",
            "TERRACE COURT",
            "TERRACE LOUNGE",
            "SKY COURT",
            "SKY DECK",
            # Covered outdoor rooms. All of these are open on at least one
            # side, so their edge needs guarding.
            "LOGGIA",
            "LANAI",
            "BREEZEWAY",
            "COLONNADE",
            "ARCADE",
            "CLOISTER",
            "SCREENED PORCH",
            "WRAPAROUND PORCH",
            "BALCONY",
            "BAL",
            "TERRACE",
            "DECK",
            "CABANA",
            "PAVILION",
            "GAZEBO",
            "PERGOLA",
            "TRELLIS",
            "ARBOR",
            # South Asian terms for the same kinds of space.
            "OTLA",
            "OSARI",
            "BARAMDA",
            "CHABUTRA",
            "JHAROKHA",
            "CHHAT",
        ),
    ),
    (
        "tank",
        (
            # Nearly universal on South Asian roofs and almost never absent
            # from the drawing, usually as a bare abbreviation.
            "OVERHEAD TANK",
            "OVER HEAD TANK",
            "OVERHEAD WATER TANK",
            "WATER TANK",
            "STORAGE TANK",
            "ROOF TANK",
            "TERRACE TANK",
            "SINTEX TANK",
            "WATER STORAGE",
        ),
    ),
    (
        "chimney",
        (
            "CHIMNEY STACK",
            "CHIMNEY BREAST",
            "CHIMNEY",
            "FLUE STACK",
            "SMOKE STACK",
            "SMOKESTACK",
        ),
    ),
    (
        "tower",
        (
            "BELL TOWER",
            "CLOCK TOWER",
            "WATER TOWER",
            "TOWER",
            "TURRET",
            "MINARET",
            "BELVEDERE",
            "STEEPLE",
            "SPIRE",
            "PINNACLE",
            "LOOKOUT",
        ),
    ),
    (
        "canopy",
        (
            # A projecting cover on posts or brackets: the thing over a front
            # door or a parked car. Distinct from a balcony, which is walked
            # on, and from a pergola, which is open to the sky.
            "PORTE COCHERE",
            "PORTICO",
            "CAR CANOPY",
            "ENTRANCE CANOPY",
            "ENTRY CANOPY",
            "DOOR CANOPY",
            "CANOPY",
            "AWNING",
            "MARQUISE",
            "OVERHANG",
            "CANTILEVER SLAB",
            "PROJECTION SLAB",
            "WEATHER SHED",
            "SUNSHADE",
            "CHAJJA",
            "CHHAJJA",
        ),
    ),
    (
        "ramp",
        (
            "WHEELCHAIR RAMP",
            "ACCESS RAMP",
            "CAR RAMP",
            "VEHICLE RAMP",
            "ENTRY RAMP",
            "LOADING RAMP",
            "SLOPED APPROACH",
            "SLOPING APPROACH",
            "RAMP DOWN",
            "RAMP UP",
            "RAMP",
        ),
    ),
    (
        "dome",
        (
            "ONION DOME",
            "GEODESIC DOME",
            "SEGMENTAL DOME",
            "HEMISPHERICAL DOME",
            "CORBELLED DOME",
            "ROOF DOME",
            "GLASS DOME",
            "DOMED CEILING",
            "DOMED ROOF",
            "DOME",
            "CUPOLA",
            "ROTUNDA",
            # South Asian and Middle Eastern terms for the same form. A
            # shikhara is a temple spire and a gumbad a tomb dome; both read
            # as a raised cap over a single room, which is what gets built.
            "SHIKHARA",
            "SHIKHAR",
            "GUMBAD",
            "GUMBAZ",
            "VIMANA",
            "QUBBA",
        ),
    ),
    (
        "glazed",
        (
            "GLASS ROOF",
            "GLAZED ROOF",
            "GLASS CEILING",
            "GLAZED CEILING",
            "SLANTING GLASS",
            "SLOPING GLAZING",
            "SLOPED GLAZING",
            "GLASS CANOPY",
            "GLAZED CANOPY",
            "LANTERN LIGHT",
            "ROOF LIGHT",
            "ROOFLIGHT",
            "SKY LIGHT",
            "SKYLIGHT",
            "CLERESTORY",
            "CONSERVATORY",
            "SUN ROOM",
            "SUNROOM",
            "GARDEN ROOM",
            "ORANGERY",
            "GREENHOUSE",
            "GLASS HOUSE",
            "SOLARIUM",
        ),
    ),
    (
        "pitched",
        (
            "PITCHED ROOF",
            "SLOPING ROOF",
            "SLOPED ROOF",
            "SLANT ROOF",
            "SLANTING ROOF",
            "GABLE ROOF",
            "GABLED ROOF",
            "GABLE END",
            "HIP ROOF",
            "HIPPED ROOF",
            "SHED ROOF",
            "LEAN TO ROOF",
            "MANSARD ROOF",
            "MANSARD",
            "GAMBREL",
            "BUTTERFLY ROOF",
            "SKILLION",
            "MONO PITCH",
            "MONOPITCH",
            "TILED ROOF",
            "SLATE ROOF",
            "THATCHED ROOF",
            "THATCH",
            "TRUSS ROOF",
            "GABLE",
        ),
    ),
    (
        "stairs",
        (
            "GRAND STAIRCASE",
            "FEATURE STAIRCASE",
            "FLOATING STAIRCASE",
            "SPIRAL STAIRCASE",
            "HELICAL STAIRCASE",
            "DOG LEG STAIRCASE",
            "SCISSOR STAIR",
            "SERVICE STAIR",
            "ESCAPE STAIR",
            "EMERGENCY STAIR",
            "FIRE STAIR",
            "STAIR HEADROOM",
            "STAIR ENCLOSURE",
            "INTERMEDIATE LANDING",
            "HALF LANDING",
            "STAIRCASE",
            "STAIRWELL",
            "STAIR LOBBY",
            "STAIR HALL",
            "STAIR CORE",
            "STAIR TOWER",
            "FIRE ESCAPE",
            # The rooftop staircase enclosure, in South Asian usage.
            "MUMTY",
            "STAIR",
            "STEPS",
            "STEP",
            "LANDING",
        ),
    ),
    (
        "wet",
        (
            "BUTLER'S PANTRY",
            "SERVICE KITCHEN",
            "POWDER ROOM",
            "DRYING AREA",
            "WASH AREA",
            "STEAM ROOM",
            "JACK AND JILL",
            "MASTER BATHROOM",
            "GUEST BATHROOM",
            "POOL BATHROOM",
            "SPA BATHROOM",
            "OUTDOOR SHOWER",
            "POOL SHOWER",
            "SHOWER ROOM",
            "WET ROOM",
            "HALF BATH",
            "FULL BATH",
            "TREATMENT ROOM",
            "MASSAGE ROOM",
            "THERAPY ROOM",
            "CHANGING ROOM",
            "LOCKER ROOM",
            "HAMMAM",
            "SCULLERY",
            "LARDER",
            "COLD STORE",
            "ENSUITE",
            "EN SUITE",
            "BATHROOM",
            "BATH",
            "TOILET",
            "SHOWER",
            "POWDER",
            "WASH",
            "UTILITY",
            "LAUNDRY",
            "SAUNA",
            "SPA",
            "WC",
        ),
    ),
]

# Flattened and sorted so the longest keyword wins whichever rule it belongs
# to. Ordering by rule alone is fragile as the vocabulary grows: "TERRACE
# GARDEN" contains "TERRACE", "POOL DECK" contains "DECK", and a shorter
# match in an earlier rule would quietly take either.
_FEATURE_LOOKUP: list[tuple[str, str]] = sorted(
    (
        (keyword, category)
        for category, keywords in FEATURE_KEYWORDS
        for keyword in keywords
    ),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

# Short marks that are only meaningful as whole words. "UP" and "DN" are the
# standard annotations on a flight of stairs and are often all that is
# printed there, but as substrings they would match GROUP, CUP and DINING.
EXACT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("stairs", ("UP", "DN", "DOWN")),
    ("void", ("OTS",)),
]

# What an interior room's floor is finished in. Separate from the feature
# categories above because these change only appearance, never geometry --
# and because a room can be both, a kitchen being wet-serviced and tiled.
#
# Checked longest first, so "CHEF'S KITCHEN" reads as a kitchen rather than
# falling through to the generic living rule.
ROOM_FINISHES: list[tuple[str, tuple[str, ...]]] = [
    (
        "timber",
        (
            # Where a house is lived in and slept in.
            "MASTER SUITE",
            "PRIMARY SUITE",
            "OWNER'S SUITE",
            "PRINCIPAL SUITE",
            "JUNIOR SUITE",
            "GUEST SUITE",
            "VIP SUITE",
            "FAMILY SUITE",
            "MASTER RETREAT",
            "BEDROOM",
            "BED ROOM",
            "NURSERY",
            "TEEN ROOM",
            "MASTER",
            "PRIMARY",
            "GUEST",
            # Quiet rooms.
            "STUDY",
            "OFFICE",
            "LIBRARY",
            "READING",
            "DEN",
            "SNUG",
            "PARLOUR",
            "PARLOR",
            "SALON",
            "RETREAT",
            "MUSIC ROOM",
            "SEWING",
            # Wardrobe and dressing.
            "DRESSING",
            "WALK IN CLOSET",
            "WALK IN WARDROBE",
            "WARDROBE",
            "CLOSET",
            "SHOE ROOM",
            "LINEN",
            "DRESS",
            "MUDROOM",
            "MUD ROOM",
            "BOOT ROOM",
            "CLOAKROOM",
            # Prayer rooms, warm underfoot by tradition.
            "TEMPLE",
            "POOJA",
            "PUJA",
            "PRAYER",
            "MANDIR",
            "MEDITATION ROOM",
        ),
    ),
    (
        "tile",
        (
            # Where a house is worked in.
            "BUTLER'S KITCHEN",
            "CHEF'S KITCHEN",
            "PREP KITCHEN",
            "SERVICE KITCHEN",
            "DIRTY KITCHEN",
            "WET KITCHEN",
            "DRY KITCHEN",
            "SHOW KITCHEN",
            "OPEN KITCHEN",
            "BACK KITCHEN",
            "STAFF KITCHEN",
            "KITCHEN",
            "PANTRY",
            "SERVANT",
            "MAID",
            "DRIVER",
            "CARETAKER",
            "STAFF ROOM",
            "HOUSEKEEPING",
            "JANITOR",
            "STORE",
            "STORAGE",
            "BOX ROOM",
            "GARBAGE",
            "REFUSE",
            "BIN STORE",
            "PLANT ROOM",
            "PUMP ROOM",
            "GENERATOR",
            "ELECTRICAL ROOM",
            "METER ROOM",
            "MECHANICAL ROOM",
            "MAINTENANCE",
            "WORKSHOP",
            # Active rooms, hard-wearing underfoot.
            "GYM",
            "GYMNASIUM",
            "FITNESS",
            "YOGA",
            "PILATES",
            "DANCE STUDIO",
            "THEATRE",
            "THEATER",
            "CINEMA",
            "SCREENING",
            "BILLIARDS",
            "SNOOKER",
            "PLAY",
            "GAME",
            "GAMING",
            "MULTI PURPOSE",
            "MULTIPURPOSE",
            "ACTIVITY",
            "RECREATION",
            "STUDIO",
        ),
    ),
    (
        "stone",
        (
            # Reception, circulation and the formal rooms.
            "GRAND FOYER",
            "ENTRANCE HALL",
            "RECEPTION HALL",
            "GREAT ROOM",
            "GREAT HALL",
            "GRAND HALL",
            "CENTRAL HALL",
            "INNER HALL",
            "LIVING",
            "DRAWING",
            "SITTING",
            "FAMILY ROOM",
            "DINING",
            "BREAKFAST",
            "HALL",
            "LOUNGE",
            "FOYER",
            "LOBBY",
            "VESTIBULE",
            "ANTEROOM",
            "ANTE ROOM",
            "GALLERY",
            "AISLE",
            "ASILE",  # as the reference sheets spell it
            "PASSAGE",
            "CORRIDOR",
            "ENTRANCE",
            "RECEPTION",
            "CONSERVATORY",
            "ORANGERY",
            "SUNROOM",
            "SOLARIUM",
            "GARDEN ROOM",
            "ROTUNDA",
            # South Asian and Middle Eastern reception rooms.
            "BAITHAK",
            "MAJLIS",
            "DIWAN",
            "DEORHI",
            "DALAN",
        ),
    ),
]

DEFAULT_FINISH = "stone"


# Flattened and sorted so the longest keyword always wins, whichever rule it
# belongs to. Ordering by rule alone is not enough: "MULTI-PURPOSE HALL"
# contains "HALL", and a shorter match in an earlier rule would take it.
_FINISH_KEYWORDS: list[tuple[str, str]] = sorted(
    ((keyword, finish) for finish, keywords in ROOM_FINISHES for keyword in keywords),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def finish_for(label: str) -> str:
    """The floor finish a room's name implies."""
    if not label:
        return DEFAULT_FINISH

    normalized = _normalize(label)
    squashed = normalized.replace(" ", "")

    for keyword, finish in _FINISH_KEYWORDS:
        if keyword in normalized or keyword.replace(" ", "") in squashed:
            return finish
    return DEFAULT_FINISH


# Categories that describe ground rather than an interior floor.
GROUND_COVERS = {"water", "lawn", "paving"}

# How far a size may sit from the name it belongs to, in multiples of the
# name's own text height. A room's name and its dimensions are printed on
# consecutive lines, so a few line heights covers the pairing while keeping
# the next room's label out of reach.
LABEL_PAIRING_LINES = 3.5
# And how far it may sit sideways, as a multiple of the name's own width. A
# dimension is centred under its name, not beside a different room's.
LABEL_PAIRING_WIDTHS = 1.2


def normalize(label: str) -> str:
    """Public alias, since finish matching needs the same normalisation."""
    return _normalize(label)


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

    for keyword, category in _FEATURE_LOOKUP:
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
    """The size belonging to a room's name, if one is printed with it.

    A room's dimensions are set immediately beneath its name and roughly
    centred on it, so only boxes below and horizontally aligned are
    considered. Taking the nearest text in any direction lets a label with no
    size of its own -- "TERRACE GARDEN 2130 SQ.FT." states an area, not a
    width -- steal the dimensions of whichever room happens to sit closest,
    and be built at that room's size in the wrong place.

    Distances are measured in multiples of the name's own text height, so the
    rule holds whatever resolution the sheet was rendered at.
    """
    if not dimension_boxes:
        return None

    line_height = max(box.bbox[3], 1)
    vertical_limit = LABEL_PAIRING_LINES * line_height
    horizontal_limit = max(box.bbox[2], line_height) * LABEL_PAIRING_WIDTHS

    best, best_distance = None, None
    for candidate, dimensions in dimension_boxes:
        drop = candidate.centre[1] - box.centre[1]
        offset = abs(candidate.centre[0] - box.centre[0])

        # Below the name, not above it, and not off to one side.
        if not 0 < drop <= vertical_limit or offset > horizontal_limit:
            continue
        if best_distance is None or drop < best_distance:
            best, best_distance = dimensions, drop

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
        category = feature_for(room)
        if category:
            grouped.setdefault(category, []).append(room)

    if grouped:
        logger.info(
            "features from names and predicted types: %s",
            {name: len(rooms) for name, rooms in grouped.items()},
        )
    return grouped


# --- Room function without a printed name ------------------------------------
#
# Everything above reads the drawing's text. Most drawings have none: across
# a batch of sixty CubiCasa plans, OCR found a room name on three. The rest
# print a disclaimer and a watermark, and their rooms are identifiable only
# from what is drawn inside them -- a hob, a toilet, a sauna bench.
#
# The segmenter is trained to name that function, so a room arrives carrying
# a predicted ``category`` even when nothing was written on it. These tables
# say what a prediction implies, in the same vocabulary the label keywords
# produce, so the rest of the pipeline cannot tell the two apart.

# Predicted room type -> feature category. Only types that change the model
# appear: a predicted bedroom builds nothing a plain room would not.
CATEGORY_FEATURES = {
    "kitchen": "wet",
    "bath": "wet",
    # Balconies, terraces and porches. They earn a railing, which is the
    # single most visible thing a room type buys on an unlabelled plan.
    "outdoor": "open",
}

# Predicted room type -> floor finish.
CATEGORY_FINISHES = {
    "kitchen": "tile",
    "bath": "tile",
    "outdoor": "stone",
    "storage": "stone",
    "circulation": "stone",
    "bedroom": "timber",
}


def feature_for(room) -> str | None:
    """The feature category a room implies, from its name or its type.

    The printed name wins where there is one: it is what the architect
    actually wrote, and it distinguishes a verandah from a balcony in a way
    no segmenter trained on Finnish apartments can. The predicted type is
    the fallback, and on most drawings it is the only thing available.
    """
    from_label = classify(getattr(room, "label", ""))
    if from_label:
        return from_label
    return CATEGORY_FEATURES.get(getattr(room, "category", ""))


def finish_for_room(room) -> str:
    """The floor finish a room implies, from its name or its type."""
    label = getattr(room, "label", "")
    if label:
        finish = finish_for(label)
        if finish != DEFAULT_FINISH:
            return finish
    return CATEGORY_FINISHES.get(getattr(room, "category", ""), DEFAULT_FINISH)


# Categories that are open to the sky. One named concept rather than a list
# repeated wherever it is needed, because the rule is general and the cost
# of it drifting apart between two call sites is a roof over a swimming
# pool at one and not the other.
#
# Anything here means the same three things: nothing is roofed over it, the
# walls bounding it are its edge rather than its enclosure and are built as
# parapets, and it sits on the floor of the storey it was drawn on.
#
# A "void" is deliberately not here. It is a hole through a floor, which is
# a different thing from a space with sky above it, and a double-height
# living room is very much roofed.
OPEN_TO_SKY = GROUND_COVERS | {"open"}


def is_open_to_sky(room) -> bool:
    """Whether a room has sky above it rather than a roof.

    Reads a printed name where there is one and the segmenter's predicted
    type otherwise, so this works on the great majority of plans that name
    nothing at all.
    """
    return feature_for(room) in OPEN_TO_SKY
