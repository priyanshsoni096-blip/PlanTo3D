"""Tests for the open-to-sky sweep harness.

The harness is the instrument every later open-air decision is read off,
so what is pinned here is the arithmetic rather than the corpus: pooling
over pixels rather than averaging over plans, and never dividing by a
count a corpus is allowed to leave at zero.
"""


def test_summarise_reports_iou_recall_and_precision():
    from scripts.open_air_accuracy import PlanScore, summarise

    scores = [
        PlanScore(plan="a", true_open_px=100, predicted_open_px=50,
                  intersection_px=40, rooms=10, undecidable=2),
        PlanScore(plan="b", true_open_px=100, predicted_open_px=200,
                  intersection_px=80, rooms=10, undecidable=3),
    ]
    result = summarise(scores)
    # Pooled over pixels, not averaged over plans: a large terrace and a
    # small one are not equally important to the roof.
    assert result["recall"] == 120 / 200
    assert result["precision"] == 120 / 250
    assert result["iou"] == 120 / (200 + 250 - 120)
    assert result["plans"] == 2
    assert result["undecidable_share"] == 5 / 20


def test_a_plan_with_no_outdoor_truth_does_not_divide_by_zero():
    from scripts.open_air_accuracy import PlanScore, summarise

    result = summarise([
        PlanScore(plan="a", true_open_px=0, predicted_open_px=0,
                  intersection_px=0, rooms=4, undecidable=1),
    ])
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0
    assert result["iou"] == 0.0


def test_a_terrace_found_at_half_its_extent_does_not_score_as_found():
    # The reason this is scored in pixels at all. Room-level scoring calls
    # this a hit; half the terrace still has a roof over it.
    from scripts.open_air_accuracy import PlanScore, summarise

    result = summarise([
        PlanScore(plan="a", true_open_px=1000, predicted_open_px=500,
                  intersection_px=500, rooms=8, undecidable=0),
    ])
    assert result["recall"] == 0.5
    assert result["precision"] == 1.0
    assert result["iou"] == 0.5


def _sheet_with(plan_count: int):
    """A synthetic sheet carrying `plan_count` plans, gutters between them."""
    import numpy as np

    sheet = np.full((900, 900 * plan_count, 3), 255, dtype=np.uint8)
    for index in range(plan_count):
        left = index * 900 + 200
        # A drawn rectangle with a ruled interior, so it reads as a plan
        # rather than as a stray mark.
        sheet[200:700, left:left + 500] = 0
        sheet[210:690, left + 10:left + 490] = 255
        for y in range(240, 660, 40):
            sheet[y:y + 3, left + 10:left + 490] = 0
    return sheet


def test_a_sheet_holding_one_plan_is_scored():
    from scripts.open_air_accuracy import carries_one_plan

    assert carries_one_plan(_sheet_with(1)) is True


def test_a_sheet_holding_several_plans_is_not_scored():
    # The harness runs the pipeline at split=1 so that the prediction and
    # the annotation share one frame. On a sheet that really holds three
    # plans that is a configuration the build never uses: the segmenter
    # sees the gutters between the plans and calls them outdoor, and the
    # harness counts page margin as a terrace the build got wrong. Scoring
    # such a sheet measures the harness, not the roof.
    #
    # output_scorecard.py already declines to judge walls on a split sheet
    # for the same reason.
    from scripts.open_air_accuracy import carries_one_plan

    assert carries_one_plan(_sheet_with(3)) is False


def test_a_sheet_the_splitter_cannot_read_is_scored_rather_than_dropped():
    # split_sheet raises on a sheet too narrow to look for a split in.
    # A sheet that cannot be shown to hold several plans is scored, so a
    # failure in the splitter silently shrinks nothing.
    import numpy as np

    from scripts.open_air_accuracy import carries_one_plan

    assert carries_one_plan(np.full((40, 40, 3), 255, dtype=np.uint8)) is True
