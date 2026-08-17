import json

import pytest

from planto3d.geometry_types import FloorPlan, Opening, Room, Wall


def _plan() -> FloorPlan:
    return FloorPlan(
        walls=[Wall(start=(0.0, 0.0), end=(10.0, 0.0), thickness=0.5)],
        rooms=[Room(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], label="BEDROOM")],
        openings=[Opening(wall_id=0, position=5.0, width=3.0, type="door")],
    )


def test_floorplan_roundtrips_through_dict():
    plan = _plan()

    restored = FloorPlan.from_dict(plan.to_dict())

    assert restored == plan


def test_to_dict_matches_the_documented_json_shape():
    d = _plan().to_dict()

    assert d == {
        "walls": [{"start": [0.0, 0.0], "end": [10.0, 0.0], "thickness": 0.5}],
        "rooms": [
            {
                "polygon": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "label": "BEDROOM",
            }
        ],
        "openings": [{"wall_id": 0, "position": 5.0, "width": 3.0, "type": "door"}],
    }


def test_to_dict_is_json_serializable():
    # Coordinates often arrive from numpy; the dict form must be plain Python
    # so it can be written straight to disk between pipeline stages.
    encoded = json.dumps(_plan().to_dict())

    assert FloorPlan.from_dict(json.loads(encoded)) == _plan()


def test_empty_floorplan_roundtrips():
    plan = FloorPlan()

    assert FloorPlan.from_dict(plan.to_dict()) == plan
    assert plan.to_dict() == {"walls": [], "rooms": [], "openings": []}


def test_wall_length_measures_endpoint_distance():
    assert Wall(start=(0.0, 0.0), end=(3.0, 4.0), thickness=0.5).length() == 5.0


@pytest.mark.parametrize("bad_type", ["hatch", "", "DOOR"])
def test_opening_rejects_an_unknown_type(bad_type):
    with pytest.raises(ValueError):
        Opening(wall_id=0, position=1.0, width=2.0, type=bad_type)


def test_room_rejects_a_polygon_that_cannot_close():
    with pytest.raises(ValueError):
        Room(polygon=[(0.0, 0.0), (1.0, 1.0)], label="TEMPLE")
