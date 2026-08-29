"""Turn a user's room correction into a Room.label change.

planto3d.features.feature_for() and planto3d.site.classify_cover()/
has_open_edge() already prefer a room's printed label over its predicted
category -- that precedence exists because a printed label is what the
architect actually wrote, and OCR is only one way that text gets there.
A user typing "this is a balcony" is the same kind of evidence, so a
correction is just a label that did not come from OCR: no new downstream
logic, only a new way to set Room.label.

CATEGORY_LABELS gives one canonical, verified keyword per
planto3d.features category, so the correction UI can offer clean names
("open" rather than making a user guess a magic word) while guaranteeing
each one round-trips correctly through classify(), classify_cover() and
has_open_edge().
"""

from dataclasses import replace

from planto3d.pipeline import PipelineResult

NO_CHANGE = "(no change)"

# One representative keyword per planto3d.features category, each verified
# by tests/test_corrections.py to classify() back to its own category and,
# where relevant, to also satisfy site.py's separate railing/ground-cover
# keyword sets. "plain room" clears a label back to empty, which drops the
# room back to whatever its predicted category says (or nothing, if it has
# none) -- the escape hatch for "the model was right, leave it alone" and
# for undoing a previous correction.
CATEGORY_LABELS: dict[str, str] = {
    "plain room": "",
    "water": "POOL",
    "lawn": "GARDEN",
    "void": "VOID",
    "paving": "PARKING",
    "open": "BALCONY",
    "tank": "WATER TANK",
    "chimney": "CHIMNEY",
    "tower": "TOWER",
    "canopy": "CANOPY",
    "ramp": "RAMP",
    "dome": "DOME",
    "glazed": "SKYLIGHT",
    "pitched": "PITCHED ROOF",
    "stairs": "STAIRS",
    "wet": "WC",
}


def apply_room_corrections(
    result: PipelineResult, corrections: dict[tuple[int, int], str]
) -> PipelineResult:
    """Apply a user's per-room overrides in place and return ``result``.

    ``corrections`` keys are ``(floor_index, room_index)`` into
    ``result.floors[floor_index].plan.rooms[room_index]``; values are
    category names from ``CATEGORY_LABELS`` (or ``NO_CHANGE`` to skip a
    room, the default state before a user touches it).
    """
    for (floor_index, room_index), category in corrections.items():
        if category == NO_CHANGE:
            continue
        floor = result.floors[floor_index]
        label = CATEGORY_LABELS[category]
        floor.plan.rooms[room_index] = replace(
            floor.plan.rooms[room_index], label=label
        )
    return result
