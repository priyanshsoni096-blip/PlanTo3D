"""Dice and IoU over class-index predictions.

Both are computed per class and averaged, not pooled over all pixels. A floor
plan is mostly background and room fill, so a pooled score is dominated by
the easy classes and would read as high while the model misses every door.
Classes absent from both prediction and target are excluded rather than
scored as perfect, which would inflate the average the same way.
"""

import torch

from planto3d.classes import NUM_CLASSES

EPSILON = 1e-7


def _per_class(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersection, prediction size and target size for each class."""
    intersections, predicted, actual = [], [], []
    for index in range(num_classes):
        p = prediction == index
        t = target == index
        intersections.append((p & t).sum())
        predicted.append(p.sum())
        actual.append(t.sum())
    return (
        torch.stack(intersections).float(),
        torch.stack(predicted).float(),
        torch.stack(actual).float(),
    )


def dice_score(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES
) -> float:
    intersection, predicted, actual = _per_class(prediction, target, num_classes)
    present = (predicted + actual) > 0
    if not present.any():
        return 1.0
    scores = 2 * intersection / (predicted + actual + EPSILON)
    return float(scores[present].mean())


def iou_score(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES
) -> float:
    intersection, predicted, actual = _per_class(prediction, target, num_classes)
    union = predicted + actual - intersection
    present = union > 0
    if not present.any():
        return 1.0
    return float((intersection[present] / (union[present] + EPSILON)).mean())


def per_class_iou(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES
) -> dict[int, float | None]:
    """IoU for each class, or None where the class appears nowhere."""
    intersection, predicted, actual = _per_class(prediction, target, num_classes)
    union = predicted + actual - intersection
    return {
        index: (float(intersection[index] / (union[index] + EPSILON)) if union[index] > 0 else None)
        for index in range(num_classes)
    }
