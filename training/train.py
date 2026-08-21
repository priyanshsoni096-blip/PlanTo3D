"""Train the U-Net segmenter on CubiCasa5K.

Intended for a Colab GPU. The encoder starts from ImageNet weights, so this
is a fine-tune rather than training from scratch -- the published CubiCasa
checkpoint is for a stacked hourglass network and cannot be loaded here.

Loss combines cross-entropy with Dice. Cross-entropy alone follows pixel
counts, and a floor plan is overwhelmingly background and room fill, so doors
and windows -- a fraction of a percent of pixels each -- contribute almost
nothing to the gradient and get predicted away entirely.
"""

import argparse
import logging
import math
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from planto3d.classes import (
    BACKGROUND,
    BATH,
    BEDROOM,
    CIRCULATION,
    CLASS_NAMES,
    DOOR,
    KITCHEN,
    NUM_CLASSES,
    OUTDOOR,
    ROOM,
    STORAGE,
    WALL,
    WINDOW,
)
from training.dataset import DEFAULT_SIZE, CubiCasaDataset
from training.metrics import dice_score, iou_score, per_class_iou

logger = logging.getLogger(__name__)

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"


def build_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=num_classes,
    )


# Share of pixels each class occupies, measured over sixty CubiCasa samples
# with ``scripts/class_balance.py``. Re-measure rather than adjust by feel if
# the class scheme changes.
#
# The spread is the problem: a window is 0.11% of a drawing and the
# background 42%, four hundred times more. Unweighted, the cheapest way for
# the model to cut its loss is to stop predicting windows at all -- it costs
# almost nothing and the average barely moves.
CLASS_FREQUENCY = {
    BACKGROUND: 0.4244,
    WALL: 0.0830,
    ROOM: 0.1992,
    DOOR: 0.0060,
    WINDOW: 0.0011,
    BEDROOM: 0.0774,
    KITCHEN: 0.0583,
    BATH: 0.0279,
    STORAGE: 0.0310,
    CIRCULATION: 0.0388,
    OUTDOOR: 0.0530,
}

# Inverse square root rather than plain inverse frequency. Plain inverse
# would weight a window 386 times a background pixel, and the gradient from
# a handful of thin strips then swamps everything else -- the model chases
# windows and loses the walls. The square root keeps the ordering while
# compressing the range to something trainable, around twenty to one.
#
# The ceiling is a guard against a class that is nearly absent rather than
# merely rare, where the reciprocal runs away. It is set clear of the
# rarest real class: at ten it caught doors and windows together and gave
# them equal weight, when a window is five times the rarer of the two and
# needs the larger share of the attention.
WEIGHT_CEILING = 25.0


def class_weights(
    frequency: dict[int, float] | None = None, ceiling: float = WEIGHT_CEILING
) -> torch.Tensor:
    """Loss weights per class, normalised to average one.

    Averaging to one keeps the loss on the same scale as an unweighted run,
    so learning rates carry over and the numbers stay comparable to earlier
    training logs.
    """
    frequency = frequency or CLASS_FREQUENCY
    raw = torch.tensor(
        [1.0 / math.sqrt(max(frequency.get(i, 1.0), 1e-6)) for i in range(NUM_CLASSES)]
    )
    return (raw.clamp(max=ceiling) / raw.clamp(max=ceiling).mean()).float()


def _match_device(
    weight: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor | None:
    """Put ``weight`` on whatever device ``reference`` is on.

    Cross-entropy refuses to mix devices, and its weight vector is the one
    tensor in a training step that nothing else moves: the model, the
    images and the masks are all sent to the GPU explicitly, and a weight
    built beside them on the CPU is easy to miss until the first batch.
    """
    if weight is None or weight.device == reference.device:
        return weight
    return weight.to(reference.device)


def build_loss(
    weights: torch.Tensor | None = None, device: torch.device | str | None = None
) -> callable:
    """Cross-entropy weighted against class imbalance, plus Dice.

    Dice is already insensitive to class size, which is why it is here; the
    weighting fixes the cross-entropy term beside it.

    ``device`` has to be given wherever the model is not on the CPU. The
    weights are a tensor like any other and cross-entropy refuses to mix
    devices, so a CPU weight vector against logits on a GPU stops the run
    on the first batch -- which is exactly where it stopped, having passed
    every CPU test beforehand.
    """
    weights = class_weights() if weights is None else weights
    if device is not None:
        weights = weights.to(device)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    dice = smp.losses.DiceLoss(mode="multiclass")

    def combined(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Belt and braces: the weights follow the logits wherever they are,
        # so passing the wrong device -- or forgetting to pass one -- costs
        # a single tensor copy rather than the whole run. The first version
        # relied on the caller alone and died on the first batch of a GPU
        # run, having passed every test on a machine that has no GPU.
        cross_entropy.weight = _match_device(cross_entropy.weight, logits)
        return cross_entropy(logits, target) + dice(logits, target)

    return combined


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    dice_total, iou_total, batches = 0.0, 0.0, 0
    class_totals = {index: [] for index in range(NUM_CLASSES)}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        predictions = model(images).argmax(dim=1)

        dice_total += dice_score(predictions, masks)
        iou_total += iou_score(predictions, masks)
        for index, value in per_class_iou(predictions, masks).items():
            if value is not None:
                class_totals[index].append(value)
        batches += 1

    return {
        "dice": dice_total / max(batches, 1),
        "iou": iou_total / max(batches, 1),
        "per_class_iou": {
            CLASS_NAMES[index]: (sum(values) / len(values) if values else None)
            for index, values in class_totals.items()
        },
    }


def train(
    data_root: Path,
    output_path: Path,
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    size: int = DEFAULT_SIZE,
    limit: int | None = None,
    num_workers: int = 2,
    augment: bool = True,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("no GPU detected; training on CPU will be impractically slow")

    data_root = Path(data_root)
    # Augmented for training, plain for validation: a score measured on
    # randomly rotated and re-compressed inputs is not comparable between
    # epochs, and being comparable is the whole job of validation.
    train_set = CubiCasaDataset(
        data_root, data_root / "train.txt", size, limit, augment=augment
    )
    val_set = CubiCasaDataset(data_root, data_root / "val.txt", size, limit)
    logger.info("train %d, val %d", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    model = build_model().to(device)
    loss_fn = build_loss(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        # So the same drawing is transformed differently each time round.
        # Left unset, augmentation is fixed per sample and buys the variety
        # of a slightly larger dataset rather than of a much larger one.
        train_set.set_epoch(epoch)
        model.train()
        running, seen = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            seen += 1

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        logger.info(
            "epoch %d/%d  loss %.4f  val dice %.4f  val iou %.4f",
            epoch,
            epochs,
            running / max(seen, 1),
            metrics["dice"],
            metrics["iou"],
        )
        logger.info("  per-class IoU: %s", metrics["per_class_iou"])

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder": ENCODER,
                    "num_classes": NUM_CLASSES,
                    "size": size,
                    "val_dice": best_dice,
                    "epoch": epoch,
                },
                output_path,
            )
            logger.info("  saved checkpoint (dice %.4f)", best_dice)

    logger.info("best validation dice %.4f -> %s", best_dice, output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="folder holding train.txt and the samples")
    parser.add_argument("output", type=Path, help="where to write the checkpoint")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="cap samples, for a smoke test")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    train(
        args.data_root,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        size=args.size,
        limit=args.limit,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
