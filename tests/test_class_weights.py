"""The loss weights that keep rare classes from being abandoned.

A window is 0.11% of a drawing. Unweighted, the cheapest way for the model
to reduce its loss is to stop predicting windows entirely -- the error it
adds is smaller than the error it removes elsewhere. These weights are what
stop that, so their shape matters more than their exact values.
"""

import pytest
import torch

from planto3d.classes import (
    BACKGROUND,
    DOOR,
    NUM_CLASSES,
    ROOM,
    WINDOW,
)
from training.train import CLASS_FREQUENCY, WEIGHT_CEILING, class_weights


def test_there_is_a_weight_for_every_class():
    assert class_weights().shape == (NUM_CLASSES,)


def test_the_weights_average_one():
    # Keeps the loss on the same scale as an unweighted run, so learning
    # rates and logged numbers stay comparable across the change.
    assert class_weights().mean().item() == pytest.approx(1.0, abs=1e-5)


def test_a_rarer_class_is_weighted_more_heavily():
    weights = class_weights()

    assert weights[WINDOW] > weights[DOOR] > weights[ROOM] > weights[BACKGROUND]


def test_the_rarest_class_keeps_its_lead_over_the_next():
    # The ceiling once sat below both, flattening a window and a door to the
    # same weight when a window is five times the rarer.
    weights = class_weights()

    assert weights[WINDOW] > weights[DOOR] * 1.5


def test_the_spread_stays_trainable():
    # Plain inverse frequency would be nearly 400:1, and the gradient from a
    # few thin strips then drowns out the walls.
    weights = class_weights()

    assert weights.max() / weights.min() < 25


def test_a_vanishing_class_cannot_run_away():
    weights = class_weights({**CLASS_FREQUENCY, WINDOW: 1e-9})

    assert torch.isfinite(weights).all()
    assert weights.max() / weights.min() < 200


def test_the_ceiling_clears_every_real_class_but_the_rarest():
    # If the ceiling caught several classes it would flatten them together,
    # which is the failure it was raised to avoid.
    raw = {index: freq ** -0.5 for index, freq in CLASS_FREQUENCY.items()}
    clamped = [index for index, value in raw.items() if value > WEIGHT_CEILING]

    assert clamped == [WINDOW]
