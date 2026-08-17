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
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from planto3d.classes import CLASS_NAMES, NUM_CLASSES
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


def build_loss() -> callable:
    cross_entropy = torch.nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")

    def combined(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("no GPU detected; training on CPU will be impractically slow")

    data_root = Path(data_root)
    train_set = CubiCasaDataset(data_root, data_root / "train.txt", size, limit)
    val_set = CubiCasaDataset(data_root, data_root / "val.txt", size, limit)
    logger.info("train %d, val %d", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    model = build_model().to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
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
