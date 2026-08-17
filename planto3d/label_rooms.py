"""Name each room from the text printed inside it.

Room names come from OCR rather than the segmentation model. CubiCasa5K's
categories are Finnish residential ones and have no equivalent for the room
types on these sheets -- Temple, Verandah, Dress/Toilet -- so the model is
only asked where a room is, never what it is called.

A room's own name sits near its middle, while a neighbour's label can spill
across the boundary, so the nearest candidate to the centre wins.
"""

import logging
import math
from dataclasses import replace

from planto3d.calibrate import TextBox, parse_dimension_text
from planto3d.geometry_types import Room

logger = logging.getLogger(__name__)

# A label needs this many letters before it is believed. OCR litters a dense
# drawing with fragments like "5 GE", "~ a" and stray rule marks.
MIN_LABEL_LETTERS = 3


def _is_plausible_label(text: str) -> bool:
    if parse_dimension_text(text):
        return False
    return sum(character.isalpha() for character in text) >= MIN_LABEL_LETTERS


def _centre(room: Room) -> tuple[float, float]:
    left, top, right, bottom = room.bounds()
    return ((left + right) / 2, (top + bottom) / 2)


def assign_labels(rooms: list[Room], text_boxes: list[TextBox]) -> list[Room]:
    """Return copies of ``rooms`` with labels taken from the text inside them.

    Rooms with no plausible label keep an empty one and are logged, rather
    than raising -- a single unnamed room should not abort a floor.
    """
    candidates = [box for box in text_boxes if _is_plausible_label(box.text)]

    labelled = []
    for room in rooms:
        centre = _centre(room)
        inside = [box for box in candidates if room.contains(box.centre)]

        if inside:
            best = min(inside, key=lambda box: math.dist(box.centre, centre))
            labelled.append(replace(room, label=best.text))
        else:
            logger.warning("no label found for room centred at (%.0f, %.0f)", *centre)
            labelled.append(replace(room, label=""))

    named = sum(1 for room in labelled if room.label)
    logger.info("labelled %d of %d room(s)", named, len(labelled))
    return labelled
