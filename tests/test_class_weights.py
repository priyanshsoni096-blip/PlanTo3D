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
from training.train import (
    CLASS_FREQUENCY,
    WEIGHT_CEILING,
    _match_device,
    build_loss,
    class_weights,
)


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


class TestTheWeightsFollowTheModel:
    """Cross-entropy refuses to mix devices, and its weight vector is the
    one tensor in a training step that nothing else moves.

    The model, the images and the masks are all sent to the GPU explicitly.
    A weight vector built beside them on the CPU is easy to miss, and it is
    missed on the first batch rather than at setup -- so a run that looked
    fine through every CPU test died the moment it touched a T4:

        Expected all tensors to be on the same device, but got weight is on
        cpu, different from other tensors on cuda:0

    These tests use a stand-in for a second device, because the machine
    this is developed on has one. That is the whole reason the bug got out.
    """

    class Elsewhere:
        """A tensor that reports living on another device."""

        def __init__(self, name="cuda:0"):
            self.device = name
            self.moved_to = None

        def to(self, target):
            self.moved_to = target
            return self

    def test_a_weight_already_in_the_right_place_is_left_alone(self):
        weight = torch.ones(3)

        assert _match_device(weight, torch.zeros(2)) is weight

    def test_a_weight_on_another_device_is_moved_to_match(self):
        weight = self.Elsewhere("cpu")
        logits = self.Elsewhere("cuda:0")

        _match_device(weight, logits)

        assert weight.moved_to == "cuda:0"

    def test_no_weights_is_not_an_error(self):
        # An unweighted loss is a legitimate configuration.
        assert _match_device(None, torch.zeros(2)) is None

    def test_the_loss_survives_weights_left_on_the_wrong_device(self):
        # The end-to-end version: build the loss without saying where the
        # model is, then use it. It has to work rather than raise.
        loss = build_loss()
        logits = torch.randn(2, NUM_CLASSES, 8, 8)
        target = torch.randint(0, NUM_CLASSES, (2, 8, 8))

        assert torch.isfinite(loss(logits, target))

    def test_asking_for_a_device_puts_them_there(self):
        loss = build_loss(device="cpu")
        logits = torch.randn(2, NUM_CLASSES, 8, 8)
        target = torch.randint(0, NUM_CLASSES, (2, 8, 8))

        assert torch.isfinite(loss(logits, target))
