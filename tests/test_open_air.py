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
