"""Tests for correcting window annotations without any interface.

The model's own window predictions are exported as LabelMe rectangles, a
person corrects them in LabelMe and ticks "reviewed", and training uses the
corrected boxes in place of the annotation's windows. What is pinned here is
the round trip, and above all the two ways it could silently go wrong: an
unreviewed export leaking into training, and held-out test plans being
corrected and then trained on.
"""

import json

import numpy as np
import pytest

from planto3d.classes import BACKGROUND, ROOM, WALL, WINDOW


def _mask():
    mask = np.full((40, 60), BACKGROUND, dtype=np.int64)
    mask[10:30, 10:50] = ROOM
    mask[10:12, 10:50] = WALL
    mask[10:12, 15:25] = WINDOW      # a window in the top wall
    mask[28:30, 10:50] = WALL
    mask[28:30, 30:44] = WINDOW      # and one in the bottom wall
    return mask


def test_export_turns_each_predicted_window_into_one_rectangle():
    from planto3d.window_labels import mask_to_labelme

    labels = mask_to_labelme(_mask(), image_name="F1_scaled.png")

    assert labels["imagePath"] == "F1_scaled.png"
    assert (labels["imageHeight"], labels["imageWidth"]) == (40, 60)
    assert labels["flags"] == {"reviewed": False}
    boxes = sorted(
        (tuple(shape["points"][0]), tuple(shape["points"][1]))
        for shape in labels["shapes"]
    )
    # Rectangles as LabelMe stores them: top-left, bottom-right, inclusive.
    assert boxes == [((15, 10), (24, 11)), ((30, 28), (43, 29))]
    assert all(s["label"] == "window" and s["shape_type"] == "rectangle"
               for s in labels["shapes"])


def test_applying_labels_replaces_the_annotations_windows():
    from planto3d.window_labels import apply_window_labels

    labels = {"shapes": [
        {"label": "window", "shape_type": "rectangle", "points": [[40, 10], [47, 11]]},
    ]}
    mask = apply_window_labels(_mask(), labels)

    # The corrected box is window...
    assert (mask[10:12, 40:48] == WINDOW).all()
    # ...and the windows the person removed go back to the wall they sat in,
    # because CubiCasa paints windows over walls.
    assert (mask[10:12, 15:25] == WALL).all()
    assert (mask[28:30, 30:44] == WALL).all()
    # Nothing else moves.
    assert (mask[15:25, 15:45] == ROOM).all()


def test_points_may_be_given_in_either_corner_order():
    # LabelMe keeps the order the rectangle was dragged in.
    from planto3d.window_labels import apply_window_labels

    labels = {"shapes": [
        {"label": "window", "shape_type": "rectangle", "points": [[47, 11], [40, 10]]},
    ]}
    assert (apply_window_labels(_mask(), labels)[10:12, 40:48] == WINDOW).all()


def test_an_unreviewed_export_is_never_applied(tmp_path):
    # The whole point of the tick box: an untouched export is the model's own
    # guess, and training on it would teach the model its own mistakes.
    from planto3d.window_labels import adjust_mask, mask_to_labelme

    image = tmp_path / "F1_scaled.png"
    labels = mask_to_labelme(np.full((40, 60), WALL, dtype=np.int64), image_name=image.name)
    labels["shapes"] = [{"label": "window", "shape_type": "rectangle", "points": [[0, 0], [5, 5]]}]
    image.with_suffix(".json").write_text(json.dumps(labels))

    original = _mask()
    assert (adjust_mask(original.copy(), image) == original).all()

    labels["flags"]["reviewed"] = True
    image.with_suffix(".json").write_text(json.dumps(labels))
    assert (adjust_mask(original.copy(), image)[0:6, 0:6] == WINDOW).all()


def test_a_plan_with_no_label_file_is_untouched(tmp_path):
    from planto3d.window_labels import adjust_mask

    original = _mask()
    assert (adjust_mask(original.copy(), tmp_path / "F1_scaled.png") == original).all()


def test_the_held_out_test_split_cannot_be_exported():
    # Correcting test plans and then training on them would contaminate the
    # benchmark every result in this project is now quoted from.
    from scripts.export_windows import refuse_held_out

    with pytest.raises(SystemExit):
        refuse_held_out("data/cubicasa5k/test.txt")
    with pytest.raises(SystemExit):
        refuse_held_out("data/cubicasa_test60.txt")
    refuse_held_out("data/cubicasa5k/train.txt")  # does not raise
