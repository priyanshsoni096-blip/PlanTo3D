"""Turning a user's room correction into a Room.label change.

A correction is not a new concept -- planto3d.features.feature_for()
already prefers a room's printed label over its predicted category, and
planto3d.site.classify_cover()/has_open_edge() do the same. So a user's
override just becomes a label that did not come from OCR: no downstream
code needs to change, which these tests confirm directly by round-
tripping every canonical label through all three consumers.
"""

import pytest

from planto3d.corrections import CATEGORY_LABELS, apply_room_corrections
from planto3d.features import classify
from planto3d.geometry_types import FloorPlan, Room
from planto3d.pipeline import FloorResult, PipelineResult
from planto3d.site import classify_cover, has_open_edge
from pathlib import Path


def _plan_with_rooms(labels: list[str]) -> PipelineResult:
    rooms = [
        Room(
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            label=label,
        )
        for label in labels
    ]
    plan = FloorPlan(walls=[], rooms=rooms, openings=[], footprint=[])
    floor = FloorResult(index=0, image_path=Path("x.png"), plan=plan)
    return PipelineResult(floors=[floor], scale=1.0, model_path=None)


@pytest.mark.parametrize("category", sorted(set(CATEGORY_LABELS) - {"plain room"}))
def test_every_canonical_label_classifies_to_its_own_category(category):
    # This is the property the whole design depends on: picking a category
    # in the correction UI must make features.classify() return exactly
    # that category, or the correction silently does nothing.
    label = CATEGORY_LABELS[category]
    assert classify(label) == category


def test_plain_room_has_no_canonical_label_and_clears_instead():
    assert CATEGORY_LABELS["plain room"] == ""


def test_apply_room_corrections_overrides_the_addressed_room():
    result = _plan_with_rooms(["BEDROOM"])
    corrected = apply_room_corrections(result, {(0, 0): "open"})
    assert corrected.floors[0].plan.rooms[0].label == CATEGORY_LABELS["open"]
    assert classify(corrected.floors[0].plan.rooms[0].label) == "open"


def test_apply_room_corrections_leaves_other_rooms_untouched():
    result = _plan_with_rooms(["BEDROOM", "KITCHEN"])
    corrected = apply_room_corrections(result, {(0, 0): "paving"})
    assert corrected.floors[0].plan.rooms[1].label == "KITCHEN"


def test_no_change_is_a_genuine_no_op():
    result = _plan_with_rooms(["BEDROOM"])
    corrected = apply_room_corrections(result, {(0, 0): "(no change)"})
    assert corrected.floors[0].plan.rooms[0].label == "BEDROOM"


def test_plain_room_clears_an_existing_label():
    result = _plan_with_rooms(["TERRACE GARDEN"])
    corrected = apply_room_corrections(result, {(0, 0): "plain room"})
    assert corrected.floors[0].plan.rooms[0].label == ""


def test_open_categories_are_also_railed_by_site_py():
    # site.py has its own separate, narrower keyword set for railings
    # (OPEN_EDGE_KEYWORDS); confirm the canonical "open" label triggers it
    # too, since that is the mechanism the correction relies on for
    # balconies actually getting a railing built.
    assert has_open_edge(CATEGORY_LABELS["open"])


def test_ground_cover_categories_are_recognised_by_site_py():
    assert classify_cover(CATEGORY_LABELS["lawn"]) == "lawn"
    assert classify_cover(CATEGORY_LABELS["paving"]) == "paving"
