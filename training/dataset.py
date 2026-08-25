"""CubiCasa5K as a PyTorch dataset of segmentation masks."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from planto3d.cubicasa import sample_paths, svg_to_mask
from training.augment import augment

logger = logging.getLogger(__name__)

# Input size for the network. Floor plans are large and mostly empty, so a
# square resize keeps batches affordable without losing wall structure.
#
# It costs the thin classes, though, and how much is measurable. A window
# is drawn about 4 pixels wide on a CubiCasa sheet, and by the time the
# sheet is squeezed into a square input it is this thin:
#
#     class      drawn    at 512    at 768
#     wall        22px     8.4px    12.5px
#     door        16px     6.5px     9.8px
#     window       4px     1.5px     2.2px
#
# One and a half pixels. The model is not bad at windows so much as barely
# shown them -- it finds 37% of them at an IoU of 0.096, against 0.52 to
# 0.77 for every other class. Its loss weight is already the highest at
# 3.72, and weighting cannot recover what resampling threw away.
#
# 768 raises every thin class by half again, at roughly twice the memory
# and time. Whether that is worth it is a decision about GPU hours rather
# than about code, so it is a parameter.
DEFAULT_SIZE = 512

# What fits alongside each size on a 16 GB card, for a ResNet34 U-Net.
# Memory goes as the square of the side, so doubling the pixels halves the
# batch.
SUGGESTED_BATCH = {512: 8, 640: 6, 768: 4, 1024: 2}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CubiCasaDataset(Dataset):
    """Pairs a floor plan image with its rasterized annotation mask.

    Masks are rendered at the annotation's own resolution and then resized
    with nearest-neighbour interpolation -- anything smoother invents class
    indices that were never labelled, blending a wall and a room into a door.

    ``augment`` turns on the training transforms. Leave it off for
    validation: a score measured on randomly rotated and re-compressed
    inputs is not comparable between epochs, and the point of validation is
    to be comparable.
    """

    def __init__(
        self,
        root: Path,
        split_file: Path,
        size: int = DEFAULT_SIZE,
        limit: int | None = None,
        augment: bool = False,
        seed: int = 0,
    ):
        self.samples = sample_paths(Path(root), Path(split_file))
        if limit is not None:
            self.samples = self.samples[:limit]
        self.size = size
        self.augment = augment
        self.seed = seed

        if not self.samples:
            raise ValueError(f"no usable samples found under {root}")

    # Bumped between epochs so the same sample is drawn differently each
    # time. Left at zero the augmentation is fixed per sample, which gives
    # the variety of a slightly larger dataset rather than of a much larger
    # one.
    epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, svg_path = self.samples[index]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"could not read {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = svg_to_mask(svg_path, image.shape[:2])

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            mask.astype(np.uint8),
            (self.size, self.size),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.augment:
            # Seeded on the sample and the epoch's worth of draws so far,
            # so a run is reproducible while every epoch still sees the
            # drawing differently.
            rng = np.random.default_rng((self.seed, index, self.epoch))
            image, mask = augment(image, mask, rng)

        normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(normalized).permute(2, 0, 1),
            torch.from_numpy(mask.astype(np.int64)),
        )
