"""The loss weights that keep rare classes from being abandoned.

A door is 0.68% of a drawing. Unweighted, the cheapest way for the model
to reduce its loss is to stop predicting doors entirely -- the error it
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
    # Doors are the rarest class once windows are rasterised whole: 0.68% of
    # a page against windows' 1.48%. Before the fill fix the masks had
    # windows at 0.11%, and this test pinned that wrong order.
    weights = class_weights()

    assert weights[DOOR] > weights[WINDOW] > weights[ROOM] > weights[BACKGROUND]


def test_weights_follow_frequency_for_every_class():
    # Stated for all classes rather than for a named pair, so the next
    # correction to a measured share cannot leave a test asserting the old
    # ranking -- which is what the window lead over doors turned out to be.
    weights = class_weights()
    by_rarity = sorted(CLASS_FREQUENCY, key=CLASS_FREQUENCY.get)

    ranked = [weights[index].item() for index in by_rarity]
    assert ranked == sorted(ranked, reverse=True)


def test_the_spread_stays_trainable():
    # Plain inverse frequency would be nearly 400:1, and the gradient from a
    # few thin strips then drowns out the walls.
    weights = class_weights()

    assert weights.max() / weights.min() < 25


def test_a_vanishing_class_cannot_run_away():
    weights = class_weights({**CLASS_FREQUENCY, WINDOW: 1e-9})

    assert torch.isfinite(weights).all()
    assert weights.max() / weights.min() < 200


def test_no_boost_is_exactly_the_shipped_weights():
    # The boost is an experiment, opt-in. Everything above describes the
    # weights the installed checkpoint was trained with, and a run that asks
    # for nothing must reproduce them to the digit.
    assert torch.equal(class_weights(boost=None), class_weights())
    assert torch.equal(class_weights(boost={DOOR: 1.0}), class_weights())


def test_a_door_boost_raises_doors_against_every_other_class():
    # Doors decide scale, the largest end-to-end failure, and the segmenter
    # finds about 2 of the 6.5 doors CubiCasa draws on the plans that fail.
    plain = class_weights()
    boosted = class_weights(boost={DOOR: 2.0})

    assert boosted[DOOR] / boosted[ROOM] > plain[DOOR] / plain[ROOM]
    assert boosted[DOOR] / boosted[BACKGROUND] > plain[DOOR] / plain[BACKGROUND]
    assert boosted.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_a_boost_cannot_lift_a_class_past_the_ceiling():
    # Otherwise a large boost reopens the runaway gradient the ceiling exists
    # to prevent.
    import math

    boosted = class_weights(boost={DOOR: 1000.0})

    # Clamped to the ceiling, so its lead over background is exactly the
    # ceiling against background's unboosted weight.
    expected = WEIGHT_CEILING * math.sqrt(CLASS_FREQUENCY[BACKGROUND])
    assert boosted[DOOR] / boosted[BACKGROUND] == pytest.approx(expected, rel=1e-5)
    assert boosted.max() == boosted[DOOR]


def test_the_training_command_accepts_door_boost():
    # Two flags on this project were added to a function and never reached
    # the command line. This asks the real parser.
    from training.train import build_parser

    arguments = build_parser().parse_args(["data", "out.pt", "--door-boost", "2"])
    assert arguments.door_boost == 2.0
    assert build_parser().parse_args(["data", "out.pt"]).door_boost == 1.0


def test_the_ceiling_catches_no_real_class():
    # The ceiling exists for a vanishing class (see the test above that
    # drives a share to 1e-9). With windows measured whole, no real class
    # comes near it: the rarest, door, sits at about half.
    raw = {index: freq ** -0.5 for index, freq in CLASS_FREQUENCY.items()}
    clamped = [index for index, value in raw.items() if value > WEIGHT_CEILING]

    assert clamped == []


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
