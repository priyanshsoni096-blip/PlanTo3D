"""Tests for comparing two checkpoints on the held-out measurements.

The comparison reads each measurement script's printed summary, so what is
pinned here is the reading: every figure comes out of real script output
exactly, and a script that failed or changed its wording stops the comparison
rather than reading as "no change". The excerpts below are copied verbatim
from runs on the 60 held-out plans.
"""

import pytest

CLASS_OUTPUT = """60 sheets
class          sheets   share  IoU pooled   recall  IoU per sheet
-----------------------------------------------------------------
background         60  40.40%       0.953    97.0%          0.960
wall               60   6.44%       0.660    88.6%          0.700
door               60   0.71%       0.545    75.9%          0.558
window             59   1.30%       0.223    27.3%          0.232
median across classes, pooled: 0.741
"""

SCALE_OUTPUT = """scored          60
median error    12.9%
worst           55.6%
within 20%      44/60

source        plans  median err  within 20%     bias
doors            38       8.7%      30/38    -1.9%
walls            22      16.7%      14/22   -14.3%
"""

SCORECARD_OUTPUT = """correct on every count: 11/60  (18%)

what fails, worst first
   walls         40 of 60
   size          16 of 60
   openings      11 of 60
   rooms          8 of 60
   storeys        3 of 60
   built          0 of 60
"""

WALL_OUTPUT = """                                               median     mean
coverage  (annotated wall that was built)       97.9%    94.9%
agreement (built wall that is really wall)      79.9%    79.2%

plans below 70% coverage:  1 of 60
plans below 70% agreement: 6 of 60
"""

WINDOW_OUTPUT = """59 plans with annotated windows
annotated windows:  422
predicted windows:  727
recall:    87.7%  (of annotated windows, how many were found)
precision: 84.5%  (of predicted windows, how many are real)
"""

OPEN_AIR_OUTPUT = """plans scored        52
sheets set aside    8  (several plans on one sheet)
rooms               483

IoU                 75.4%   <-- the headline
recall              89.3%
precision           82.9%

undecidable rooms   23.2%  (neither label nor predicted type)
"""


def test_class_accuracy_reads_the_classes_that_decide_the_model():
    from scripts.compare_checkpoints import parse_class_accuracy

    figures = parse_class_accuracy(CLASS_OUTPUT)
    assert figures["wall IoU"] == 0.660
    assert figures["door IoU"] == 0.545
    assert figures["window IoU"] == 0.223
    assert figures["window pixel recall %"] == 27.3


def test_scale_reads_median_error_and_plans_within_a_fifth():
    from scripts.compare_checkpoints import parse_scale_accuracy

    figures = parse_scale_accuracy(SCALE_OUTPUT)
    assert figures["scale median error %"] == 12.9
    # A count of plans, named so the table never prints it as a percentage.
    assert figures["plans scaled within a fifth"] == 44


def test_scorecard_reads_the_headline_and_every_failure_count():
    from scripts.compare_checkpoints import parse_scorecard

    figures = parse_scorecard(SCORECARD_OUTPUT)
    assert figures["right on every check"] == 11
    assert figures["fails on walls"] == 40
    assert figures["fails on size"] == 16
    assert figures["fails on built"] == 0


def test_wall_accuracy_reads_the_medians_not_the_means():
    from scripts.compare_checkpoints import parse_wall_accuracy

    figures = parse_wall_accuracy(WALL_OUTPUT)
    assert figures["wall coverage %"] == 97.9
    assert figures["wall agreement %"] == 79.9


def test_window_detection_reads_recall_and_precision():
    from scripts.compare_checkpoints import parse_window_detection

    figures = parse_window_detection(WINDOW_OUTPUT)
    assert figures["window detection recall %"] == 87.7
    assert figures["window detection precision %"] == 84.5


def test_open_air_reads_the_headline_iou():
    from scripts.compare_checkpoints import parse_open_air

    assert parse_open_air(OPEN_AIR_OUTPUT)["open-air IoU %"] == 75.4


def test_output_without_its_summary_stops_the_comparison():
    # A script that crashed prints a traceback and no summary. Read as
    # nothing, it would make a candidate look unchanged on that measure.
    from scripts.compare_checkpoints import parse_scale_accuracy, parse_scorecard

    with pytest.raises(ValueError, match="scale"):
        parse_scale_accuracy("Traceback (most recent call last):\n  ...\n")
    with pytest.raises(ValueError, match="scorecard"):
        parse_scorecard("")


def test_direction_is_per_measure():
    # Scale error and failure counts are better lower; everything else is
    # better higher. Getting one backwards would call a regression a win.
    from scripts.compare_checkpoints import compare

    rows = {row.name: row for row in compare(
        {"scale median error %": 12.9, "fails on walls": 40, "wall IoU": 0.660, "door IoU": 0.545},
        {"scale median error %": 13.7, "fails on walls": 12, "wall IoU": 0.700, "door IoU": 0.545},
    )}
    assert rows["scale median error %"].better is False
    assert rows["fails on walls"].better is True
    assert rows["wall IoU"].better is True
    assert rows["door IoU"].better is None


def test_the_command_accepts_both_checkpoints():
    # Two flags on this project reached a function and never the command
    # line. This asks the real parser.
    from scripts.compare_checkpoints import build_parser

    arguments = build_parser().parse_args(
        ["data", "--baseline", "a.pt", "--candidate", "b.pt", "--limit", "20"]
    )
    assert (arguments.baseline, arguments.candidate, arguments.limit) == ("a.pt", "b.pt", 20)
    assert build_parser().parse_args(["data", "--baseline", "a.pt", "--candidate", "b.pt"]).limit == 60
