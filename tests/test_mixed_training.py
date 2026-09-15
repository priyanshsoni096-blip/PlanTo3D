"""Tests for training on CubiCasa and CVC-FP together.

CVC-FP is a second drafting convention, but its labels mean different things
from CubiCasa's, and trained on as they are they would teach the model
wrong things. Pinned here:

- its doors are the swing arc, CubiCasa's the leaf, so its door pixels are
  not scored at all rather than teaching arcs;
- its rooms carry no type, so a room pixel is scored as "some room type"
  rather than as the untyped ROOM class, which would teach the model to stop
  typing rooms;
- the plans held out to judge the result never reach training;
- a run cut off by Colab picks up where it stopped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from planto3d.classes import BACKGROUND, BEDROOM, DOOR, KITCHEN, NUM_CLASSES, ROOM, WALL, WINDOW

NAMES = (
    [f"Ia_A{i:02d}_sommaire" for i in range(6)]
    + [f"IIa_B{i:02d}" for i in range(9)]
    + [f"image{i:03d}" for i in range(3)]
    + [f"p{i}" for i in range(1, 3)]
)


# --- the split ---------------------------------------------------------------


def test_the_split_is_fixed_by_its_seed():
    from planto3d.cvc_fp import split_names

    first = split_names(NAMES, test_share=1 / 3, seed=20260915)

    assert first == split_names(list(reversed(NAMES)), test_share=1 / 3, seed=20260915)
    assert first != split_names(NAMES, test_share=1 / 3, seed=1)


def test_no_plan_is_both_trained_on_and_tested_on():
    from planto3d.cvc_fp import split_names

    train, test = split_names(NAMES, test_share=1 / 3, seed=20260915)

    assert not set(train) & set(test)
    assert sorted(train + test) == sorted(NAMES)


def test_every_drawing_subset_is_tested_on():
    # The subsets differ in style on purpose. A test set drawn from the big
    # subset alone would say nothing about the others.
    from planto3d.cvc_fp import split_names, subset

    _, test = split_names(NAMES, test_share=1 / 3, seed=20260915)

    assert {subset(name) for name in test} == {subset(name) for name in NAMES}


def test_subsets_are_read_from_the_file_names():
    from planto3d.cvc_fp import subset

    assert subset("Ia_AP2022_sommaire") == "Ia"
    assert subset("IIa_BK0701") == "IIa"
    assert subset("image001a") == "image"
    assert subset("p3") == "p"
    assert subset("10") == "numbered"


def test_the_committed_test_list_is_the_seeded_draw(tmp_path):
    from pathlib import Path

    from planto3d.cvc_fp import TEST_LIST, read_names, sample_paths, split_names

    root = Path("data/cvc_fp")
    if not root.is_dir():
        pytest.skip("CVC-FP is not on this machine")
    names = [image.stem for image, _ in sample_paths(root)]
    _, test = split_names(names)

    assert read_names(TEST_LIST) == sorted(test)


# --- the labels ----------------------------------------------------------------


def _plan(root):
    import cv2

    folder = root / "ImagesGT"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "p1.png"), np.full((60, 60, 3), 255, np.uint8))
    (folder / "p1_gt_1.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<polygon class="Room" points="0,0 60,0 60,30 0,30"/>'
        '<polygon class="Wall" points="0,30 60,30 60,40 0,40"/>'
        '<polygon class="Door" points="10,40 20,40 20,50 10,50"/>'
        '<polygon class="Window" points="40,30 50,30 50,40 40,40"/>'
        "</svg>"
    )
    return root


def test_cvc_fp_labels_score_rooms_as_any_type_and_doors_not_at_all(tmp_path):
    from training.dataset import ANY_ROOM, IGNORE_INDEX, CvcFpDataset

    dataset = CvcFpDataset(_plan(tmp_path), ["p1"], size=60)
    _, mask = dataset[0]

    assert mask[10, 10] == ANY_ROOM
    assert mask[45, 15] == IGNORE_INDEX
    assert mask[35, 5] == WALL
    assert mask[35, 45] == WINDOW
    assert mask[55, 55] == BACKGROUND
    assert ROOM not in mask.unique().tolist()
    assert DOOR not in mask.unique().tolist()


def test_cvc_fp_training_uses_only_the_names_it_is_given(tmp_path):
    from training.dataset import CvcFpDataset

    with pytest.raises(ValueError):
        CvcFpDataset(_plan(tmp_path), ["not_there"], size=60)


# --- the loss --------------------------------------------------------------------


def _logits_for(label, shape=(1, 4, 4)):
    logits = torch.full((shape[0], NUM_CLASSES, *shape[1:]), -8.0)
    logits[:, label] = 8.0
    return logits


def test_an_any_room_pixel_is_right_whatever_room_type_is_predicted():
    from training.dataset import ANY_ROOM
    from training.train import build_loss

    loss = build_loss()
    target = torch.full((1, 4, 4), ANY_ROOM, dtype=torch.int64)

    as_bedroom = loss(_logits_for(BEDROOM), target)
    as_kitchen = loss(_logits_for(KITCHEN), target)
    as_wall = loss(_logits_for(WALL), target)

    assert float(as_bedroom) == pytest.approx(float(as_kitchen), abs=1e-4)
    assert float(as_bedroom) < 0.01
    assert float(as_wall) > 1.0


def test_ignored_pixels_do_not_change_the_loss():
    from training.dataset import IGNORE_INDEX
    from training.train import build_loss

    loss = build_loss()
    target = torch.full((1, 4, 4), WALL, dtype=torch.int64)
    target[0, :, :2] = IGNORE_INDEX

    right = _logits_for(WALL)
    wrong_where_ignored = right.clone()
    wrong_where_ignored[0, :, :, :2] = -8.0
    wrong_where_ignored[0, DOOR, :, :2] = 8.0

    assert float(loss(right, target)) == pytest.approx(
        float(loss(wrong_where_ignored, target)), abs=1e-4
    )


def test_plain_cubicasa_targets_score_as_before():
    # No marker values: the loss must be exactly the weighted cross-entropy
    # plus Dice it always was, so a CubiCasa-only run is unchanged.
    import segmentation_models_pytorch as smp

    from training.train import build_loss, class_weights

    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 6, 6)
    target = torch.randint(0, NUM_CLASSES, (2, 6, 6))
    expected = torch.nn.CrossEntropyLoss(weight=class_weights())(logits, target) + smp.losses.DiceLoss(
        mode="multiclass"
    )(logits, target)

    assert float(build_loss()(logits, target)) == pytest.approx(float(expected), abs=1e-4)


# --- resuming ----------------------------------------------------------------


def test_a_cut_off_run_resumes_from_the_last_finished_epoch(tmp_path):
    from training.train import load_progress, save_progress

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
    for _ in range(5):
        scheduler.step()
    path = tmp_path / "run.progress.pt"

    save_progress(path, model, optimizer, scheduler, epoch=5, best_dice=0.71)

    fresh = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=24)
    start, best = load_progress(path, fresh, fresh_optimizer, fresh_scheduler)

    assert (start, best) == (6, 0.71)
    assert torch.equal(fresh.weight, model.weight)
    assert fresh_scheduler.last_epoch == 5


def test_with_no_progress_file_a_run_starts_at_the_beginning(tmp_path):
    from training.train import load_progress

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)

    assert load_progress(tmp_path / "missing.pt", model, optimizer, scheduler) == (1, 0.0)


# --- the command line ----------------------------------------------------------


def test_the_command_accepts_the_second_corpus():
    from training.train import build_parser

    arguments = build_parser().parse_args(
        ["data", "out.pt", "--cvc-root", "cvc", "--cvc-repeat", "3"]
    )
    assert (str(arguments.cvc_root), arguments.cvc_repeat) == ("cvc", 3)
    defaults = build_parser().parse_args(["data", "out.pt"])
    assert defaults.cvc_root is None
