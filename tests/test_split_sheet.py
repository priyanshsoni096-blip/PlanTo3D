"""Sheets carrying several plans must be split, not read as one storey."""

import numpy as np

from planto3d.ingest import split_sheet

WHITE, INK = 255, 40


def _plan(width=300, height=400):
    """A dense block standing in for a drawing."""
    piece = np.full((height, width, 3), WHITE, dtype=np.uint8)
    piece[40 : height - 40, 30 : width - 30] = INK
    piece[height // 2, :] = INK
    return piece


def _sheet(count, gutter=90, margin=60):
    plans = [_plan() for _ in range(count)]
    height = plans[0].shape[0]
    width = margin * 2 + sum(p.shape[1] for p in plans) + gutter * (count - 1)

    sheet = np.full((height, width, 3), WHITE, dtype=np.uint8)
    x = margin
    for plan in plans:
        sheet[:, x : x + plan.shape[1]] = plan
        x += plan.shape[1] + gutter
    return sheet


def test_three_plans_in_a_row_become_three_images():
    # The case that reconstructed three buildings as one flat floor.
    assert len(split_sheet(_sheet(3))) == 3


def test_two_plans_become_two():
    assert len(split_sheet(_sheet(2))) == 2


def test_a_single_plan_is_left_alone():
    assert len(split_sheet(_sheet(1))) == 1


def test_gaps_inside_one_drawing_do_not_split_it():
    # A plan has plenty of internal white space; only a gutter wide relative
    # to the whole sheet separates two drawings.
    sheet = _sheet(1)
    width = sheet.shape[1]
    sheet[:, width // 2 - 8 : width // 2 + 8] = WHITE

    assert len(split_sheet(sheet)) == 1


def test_margins_are_not_mistaken_for_gutters():
    # Wide white borders sit outside the plans, not between them.
    sheet = _sheet(1, margin=200)

    assert len(split_sheet(sheet)) == 1


def test_the_pieces_keep_the_full_sheet_height():
    pieces = split_sheet(_sheet(3))

    assert all(p.shape[0] == _sheet(3).shape[0] for p in pieces)


def test_every_piece_carries_a_drawing():
    from planto3d.ingest import _ink_mask

    for piece in split_sheet(_sheet(3)):
        assert _ink_mask(piece).mean() > 0.01


def test_pieces_stay_in_sheet_order():
    # Left to right is basement upward on the drawings that do this, which
    # is the order the storeys stack in.
    sheet = _sheet(3)
    pieces = split_sheet(sheet)

    widths = [p.shape[1] for p in pieces]
    assert sum(widths) <= sheet.shape[1]


def test_a_blank_sheet_is_not_split():
    assert len(split_sheet(np.full((400, 900, 3), WHITE, dtype=np.uint8))) == 1


def test_a_narrow_image_is_never_split():
    assert len(split_sheet(_plan(width=200))) == 1
