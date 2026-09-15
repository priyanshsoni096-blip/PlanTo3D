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
    ROOM_CLASSES,
    STORAGE,
    WALL,
    WINDOW,
)
from planto3d import cvc_fp
from training.dataset import ANY_ROOM, DEFAULT_SIZE, IGNORE_INDEX, CubiCasaDataset, CvcFpDataset
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


# Share of pixels each class occupies, measured over sixty CubiCasa training
# samples with ``scripts/class_balance.py``. Re-measure rather than adjust by
# feel if the class scheme or the rasteriser changes.
#
# Re-measured 2026-09-15, after fixing ``cubicasa.svg_to_mask``: it had filled
# every polygon of a class in one call, and CubiCasa repeats outlines, so
# every window came out as its border alone and its glass was labelled wall.
# The table this replaced had windows at 0.11% and walls at 8.3% for that
# reason. Measured whole, windows are 1.48% and walls 7.1%, and doors -- not
# windows -- are the rarest class. Sixty training plans drawn with
# random.Random(20260915) from train.txt.
#
# The spread is still the problem: a door is 0.68% of a drawing and the
# background 42%, sixty times more. Unweighted, the cheapest way for the
# model to cut its loss is to stop predicting the thin classes at all.
CLASS_FREQUENCY = {
    BACKGROUND: 0.4172,
    WALL: 0.0710,
    ROOM: 0.1972,
    DOOR: 0.0068,
    WINDOW: 0.0148,
    BEDROOM: 0.0837,
    KITCHEN: 0.0497,
    BATH: 0.0280,
    STORAGE: 0.0310,
    CIRCULATION: 0.0439,
    OUTDOOR: 0.0567,
}

# Inverse square root rather than plain inverse frequency. Plain inverse
# would weight a door 61 times a background pixel, and the gradient from
# a handful of thin strips then swamps everything else -- the model chases
# windows and loses the walls. The square root keeps the ordering while
# compressing the range to something trainable -- 7.8 to one across the
# shares above.
#
# The ceiling is a guard against a class that is nearly absent rather than
# merely rare, where the reciprocal runs away. With the corrected shares no
# real class reaches it -- the rarest, door, sits at about half -- so it now
# only bounds a vanishing class and an experimental boost.
WEIGHT_CEILING = 25.0


def class_weights(
    frequency: dict[int, float] | None = None,
    ceiling: float = WEIGHT_CEILING,
    boost: dict[int, float] | None = None,
) -> torch.Tensor:
    """Loss weights per class, normalised to average one.

    Averaging to one keeps the loss on the same scale as an unweighted run,
    so learning rates carry over and the numbers stay comparable to earlier
    training logs.
    """
    frequency = frequency or CLASS_FREQUENCY
    # An optional multiplier per class, for experiments. Applied before the
    # ceiling, so no boost can reopen the runaway gradient the ceiling stops.
    boost = boost or {}
    raw = torch.tensor(
        [
            boost.get(i, 1.0) / math.sqrt(max(frequency.get(i, 1.0), 1e-6))
            for i in range(NUM_CLASSES)
        ]
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
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
    room_types = sorted(ROOM_CLASSES)

    def combined(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Belt and braces: the weights follow the logits wherever they are,
        # so passing the wrong device -- or forgetting to pass one -- costs
        # a single tensor copy rather than the whole run. The first version
        # relied on the caller alone and died on the first batch of a GPU
        # run, having passed every test on a machine that has no GPU.
        cross_entropy.weight = _match_device(cross_entropy.weight, logits)

        # A pixel known only to be "some room" is not scored by either term
        # below; it gets its own, further down.
        any_room = target == ANY_ROOM
        exact = target.masked_fill(any_room, IGNORE_INDEX)

        loss = logits.new_zeros(())
        if (exact != IGNORE_INDEX).any():
            loss = loss + cross_entropy(logits, exact) + dice(logits, exact)

        if any_room.any():
            # The probability of every room type together, so the model is
            # rewarded for calling it a room and indifferent to which kind.
            # Weighted like an average room class, to sit on the same scale
            # as the cross-entropy beside it.
            log_room = torch.logsumexp(torch.log_softmax(logits, dim=1)[:, room_types], dim=1)
            weight = cross_entropy.weight[room_types].mean()
            loss = loss - weight * log_room[any_room].mean()

        return loss

    return combined


def save_progress(path: Path, model, optimizer, scheduler, epoch: int, best_dice: float) -> None:
    """Everything needed to carry on after ``epoch``, if the run is cut off.

    Colab stops a notebook after a few hours; the corrected-window run was
    stopped at epoch 21 of 24. Written after every epoch, beside the
    checkpoint, and a restart picks up from it instead of from nothing.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
        },
        path,
    )


def load_progress(path: Path, model, optimizer, scheduler) -> tuple[int, float]:
    """(first epoch to run, best validation Dice so far), restoring state if saved."""
    path = Path(path)
    if not path.is_file():
        return 1, 0.0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    logger.info("resuming after epoch %d (best dice %.4f)", state["epoch"], state["best_dice"])
    return state["epoch"] + 1, float(state["best_dice"])


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
    door_boost: float = 1.0,
    cvc_root: Path | None = None,
    cvc_repeat: int = 4,
    resume: bool = True,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("no GPU detected; training on CPU will be impractically slow")

    data_root = Path(data_root)
    # Augmented for training, plain for validation: a score measured on
    # randomly rotated and re-compressed inputs is not comparable between
    # epochs, and being comparable is the whole job of validation.
    cubicasa_train = CubiCasaDataset(
        data_root, data_root / "train.txt", size, limit, augment=augment
    )
    parts = [cubicasa_train]
    cvc_names: list[str] = []
    if cvc_root is not None:
        cvc_names = _cvc_training_names(Path(cvc_root), limit)
        parts.append(
            CvcFpDataset(cvc_root, cvc_names, size, augment=augment, repeat=cvc_repeat)
        )
        logger.info("CVC-FP: %d plan(s), each shown %d time(s) an epoch", len(cvc_names), cvc_repeat)
    train_set = torch.utils.data.ConcatDataset(parts)
    val_set = CubiCasaDataset(data_root, data_root / "val.txt", size, limit)
    logger.info("train %d, val %d", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    model = build_model().to(device)
    # door_boost is an experiment: the shipped checkpoint was trained at 1.0.
    # Doors decide scale, the largest end-to-end failure, and on the held-out
    # plans that fail it the segmenter finds about 2 of the 6.5 doors drawn.
    loss_fn = build_loss(class_weights(boost={DOOR: door_boost}), device=device)
    logger.info("door boost %.2f", door_boost)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.pt")
    start, best_dice = (
        load_progress(progress_path, model, optimizer, scheduler) if resume else (1, 0.0)
    )

    for epoch in range(start, epochs + 1):
        # So the same drawing is transformed differently each time round.
        # Left unset, augmentation is fixed per sample and buys the variety
        # of a slightly larger dataset rather than of a much larger one.
        for part in parts:
            part.set_epoch(epoch)
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
                    "door_boost": door_boost,
                    "cvc_fp_plans": len(cvc_names),
                    "cvc_repeat": cvc_repeat if cvc_names else 0,
                },
                output_path,
            )
            logger.info("  saved checkpoint (dice %.4f)", best_dice)

        save_progress(progress_path, model, optimizer, scheduler, epoch, best_dice)

    logger.info("best validation dice %.4f -> %s", best_dice, output_path)
    return output_path


def _cvc_training_names(root: Path, limit: int | None) -> list[str]:
    """Every CVC-FP plan not on the committed test list.

    The committed list is the authority, not a fresh draw, so a plan added
    to the folder later can never slip into training while being tested on.
    """
    names = [image.stem for image, _ in cvc_fp.sample_paths(root)]
    held_back = set(cvc_fp.read_names(cvc_fp.TEST_LIST))
    if not held_back:
        raise ValueError(f"{cvc_fp.TEST_LIST} is empty; refusing to train on every CVC-FP plan")
    training = sorted(name for name in names if name not in held_back)
    return training[:limit] if limit is not None else training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="folder holding train.txt and the samples")
    parser.add_argument("output", type=Path, help="where to write the checkpoint")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="cap samples, for a smoke test")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--door-boost", type=float, default=1.0,
        help="multiply the door loss weight; 1.0 reproduces the shipped checkpoint",
    )
    parser.add_argument(
        "--cvc-root", type=Path, default=None,
        help="also train on CVC-FP here, minus data/cvc_fp_test.txt",
    )
    parser.add_argument(
        "--cvc-repeat", type=int, default=4, help="times each CVC-FP plan is shown per epoch"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args()

    train(
        args.data_root,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        size=args.size,
        limit=args.limit,
        num_workers=args.num_workers,
        door_boost=args.door_boost,
        cvc_root=args.cvc_root,
        cvc_repeat=args.cvc_repeat,
    )


if __name__ == "__main__":
    main()
