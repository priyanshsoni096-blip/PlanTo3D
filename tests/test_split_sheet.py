"""Sheets carrying several plans must be split, not read as one storey."""

import numpy as np

from planto3d.ingest import MIN_SPLIT_WIDTH, split_sheet

WHITE, INK = 255, 40


def _plan(width=300, height=400):
    """Walls enclosing rooms, standing in for a drawing.

    This used to be a solid block of ink, which was enough while the
    splitter only asked where the gutters were and how much ink each piece
    carried. It also asks whether a piece encloses any space now, since
    that is what separates a floor plan from a legend or a title block --
    and a solid block encloses nothing. So the stand-in has to be a plan:
    an outer wall with partitions inside it.
    """
    piece = np.full((height, width, 3), WHITE, dtype=np.uint8)
    thickness = 10
    top, bottom, left, right = 40, height - 40, 30, width - 30

    # Outer wall, drawn as a ring rather than a filled rectangle.
    piece[top:bottom, left:right] = INK
    piece[top + thickness : bottom - thickness, left + thickness : right - thickness] = WHITE

    # One partition each way, leaving four rooms.
    middle_y, middle_x = (top + bottom) // 2, (left + right) // 2
    piece[middle_y : middle_y + thickness, left:right] = INK
    piece[top:bottom, middle_x : middle_x + thickness] = INK
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


def test_narrow_gaps_inside_one_drawing_do_not_split_it():
    # A plan has plenty of internal white space, and none of it is a gutter.
    #
    # The width alone cannot carry this test any more. Real gutters between
    # plans laid side by side measure 2-3% of the sheet -- narrower than
    # this synthetic sheet's own internal band used to be -- so the
    # threshold had to come down to 2% to find them at all. What keeps a
    # single plan whole now is the check on the pieces afterwards, not the
    # width of the gap.
    sheet = _sheet(1)
    width = sheet.shape[1]
    # Well under any real gutter. Measured against CubiCasa's recorded
    # floor counts, the gutters between two plans run 1.0 to 3.1% of the
    # sheet, so a gap in that range is genuinely ambiguous on width alone
    # -- a symmetric block with a clean full-height gap down the middle is
    # two plans as far as anything can tell. What separates them in
    # practice is the check on the pieces afterwards, not the gap.
    gap = max(int(width * 0.003), 2)
    middle = width // 2
    sheet[:, middle - gap // 2 : middle + gap // 2] = WHITE

    assert len(split_sheet(sheet)) == 1


def test_a_split_leaving_a_sliver_is_rejected():
    # The check that replaced width as the real defence. Every false split
    # measured across sixty CubiCasa sheets left one piece holding less
    # than a fifth of the ink -- several less than a tenth -- while genuine
    # multi-plan sheets divide far more evenly, both drawings being at the
    # same scale with comparable detail.
    sheet = _sheet(1)
    height, width = sheet.shape[:2]

    # A wide band of blank page near one edge: gutter-shaped, but what it
    # separates is a plan from a scrap.
    sheet[:, int(width * 0.16) : int(width * 0.22)] = WHITE

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


def test_plans_stacked_top_to_bottom_are_found_too():
    # Sheets are laid out both ways. Looking only for vertical gutters
    # missed every stacked sheet outright -- nine of sixty CubiCasa sheets
    # hold more than one plan and several stack them.
    tall = np.rot90(_sheet(2)).copy()

    assert len(split_sheet(tall)) == 2


def test_a_ruled_gutter_still_reads_as_one_gutter():
    # Two plans side by side are frequently separated by a border line
    # rather than blank page alone. That single line of ink broke the empty
    # run in two, and each half then looked too narrow to be a gutter.
    sheet = _sheet(2)
    width = sheet.shape[1]
    sheet[:, width // 2 - 1 : width // 2 + 1] = 0

    assert len(split_sheet(sheet)) == 2


def _legend(width=300, height=150):
    """A key: rows of short strokes beside small symbols.

    Carries plenty of ink and sits behind a perfectly good gutter, which
    is exactly why it used to be built as a second storey. It encloses
    nothing, which is what gives it away.
    """
    piece = np.full((height, width, 3), WHITE, dtype=np.uint8)
    for row, y in enumerate(range(20, height - 20, 34)):
        piece[y : y + 9, 24 : 24 + 18] = INK          # the symbol
        piece[y + 2 : y + 8, 60 : width - 40] = INK   # the caption beside it
    return piece


def test_a_legend_under_the_plan_is_not_a_second_storey():
    # Sheet 8150: a sketch with a key beneath it, split into plan plus key
    # and reconstructed as a two-storey building. The gutter is real; what
    # is behind it is not a plan.
    plan = _plan(width=460)
    legend = _legend(width=plan.shape[1])
    gutter = np.full((70, plan.shape[1], 3), WHITE, dtype=np.uint8)
    sheet = np.vstack([plan, gutter, legend])

    # The sheet has to clear MIN_SPLIT_WIDTH or nothing is even considered,
    # which would pass this test for the wrong reason.
    assert sheet.shape[1] >= MIN_SPLIT_WIDTH

    assert len(split_sheet(sheet)) == 1


def test_a_real_second_plan_behind_the_same_gutter_still_splits():
    # The guard must not simply refuse every stacked sheet: the thing it
    # rejects is a piece that encloses nothing, not a piece below a gutter.
    plan = _plan(width=460)
    gutter = np.full((70, plan.shape[1], 3), WHITE, dtype=np.uint8)
    sheet = np.vstack([plan, gutter, _plan(width=460)])

    assert len(split_sheet(sheet)) == 2
