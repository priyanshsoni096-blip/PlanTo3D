import logging

import pytest

from planto3d.calibrate import TextBox
from planto3d.geometry_types import Room
from planto3d.label_rooms import assign_labels


def _room(size: float = 200.0, offset: float = 0.0) -> Room:
    return Room(
        polygon=[
            (offset, offset),
            (offset + size, offset),
            (offset + size, offset + size),
            (offset, offset + size),
        ]
    )


def _box(text: str, x: float, y: float, confidence: float = 90.0) -> TextBox:
    return TextBox(text=text, bbox=(int(x), int(y), 40, 12), confidence=confidence)


def test_labels_a_room_from_text_inside_it():
    labelled = assign_labels([_room()], [_box("BEDROOM", 80, 90)])

    assert labelled[0].label == "BEDROOM"


def test_prefers_the_name_over_the_dimension_label():
    # Rooms carry both a name and a size; only the name is the label.
    boxes = [_box("15'0\"X18'0\"", 80, 120), _box("BEDROOM", 80, 90)]

    labelled = assign_labels([_room()], boxes)

    assert labelled[0].label == "BEDROOM"


def test_keeps_multi_word_and_punctuated_room_names():
    # Real sheets carry names like these; they must survive intact.
    for name in ("DRESS/TOILET", "MULTI-PURPOSE HALL", "PLAY AREA"):
        labelled = assign_labels([_room()], [_box(name, 80, 90)])
        assert labelled[0].label == name


def test_ignores_text_belonging_to_another_room():
    rooms = [_room(offset=0), _room(offset=400)]
    boxes = [_box("KITCHEN", 80, 90), _box("TEMPLE", 480, 490)]

    labelled = assign_labels(rooms, boxes)

    assert [r.label for r in labelled] == ["KITCHEN", "TEMPLE"]


def test_picks_the_candidate_nearest_the_room_centre():
    # A neighbouring room's name can bleed inside this room's polygon; the
    # room's own label sits nearer its middle.
    room = _room(size=200)
    boxes = [_box("STRAY", 5, 5), _box("KITCHEN", 95, 95)]

    labelled = assign_labels([room], boxes)

    assert labelled[0].label == "KITCHEN"


@pytest.mark.parametrize("noise", ["5 GE", "|", "~ a", "4", ""])
def test_rejects_ocr_noise_as_a_label(noise):
    labelled = assign_labels([_room()], [_box(noise, 80, 90)])

    assert labelled[0].label == ""


def test_warns_and_leaves_the_label_blank_when_nothing_matches(caplog):
    with caplog.at_level(logging.WARNING):
        labelled = assign_labels([_room()], [])

    assert labelled[0].label == ""
    assert "no label" in caplog.text.lower()


def test_does_not_mutate_the_rooms_passed_in():
    room = _room()

    labelled = assign_labels([room], [_box("FOYER", 80, 90)])

    assert room.label == ""
    assert labelled[0] is not room


def test_preserves_room_order_and_geometry():
    rooms = [_room(offset=0), _room(offset=400)]

    labelled = assign_labels(rooms, [_box("KITCHEN", 80, 90)])

    assert [r.polygon for r in labelled] == [r.polygon for r in rooms]
    assert labelled[1].label == ""
