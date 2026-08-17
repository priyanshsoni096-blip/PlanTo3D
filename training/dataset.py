"""CubiCasa5K as a PyTorch dataset of five-class segmentation masks."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from planto3d.cubicasa import sample_paths, svg_to_mask

logger = logging.getLogger(__name__)

# Input size for the network. Floor plans are large and mostly empty, so a
# square resize keeps batches affordable without losing wall structure.
DEFAULT_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CubiCasaDataset(Dataset):
    """Pairs a floor plan image with its rasterized annotation mask.

    Masks are rendered at the annotation's own resolution and then resized
    with nearest-neighbour interpolation -- anything smoother invents class
    indices that were never labelled, blending a wall and a room into a door.
    """

    def __init__(
        self,
        root: Path,
        split_file: Path,
        size: int = DEFAULT_SIZE,
        limit: int | None = None,
    ):
        self.samples = sample_paths(Path(root), Path(split_file))
        if limit is not None:
            self.samples = self.samples[:limit]
        self.size = size

        if not self.samples:
            raise ValueError(f"no usable samples found under {root}")

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

        normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(normalized).permute(2, 0, 1),
            torch.from_numpy(mask.astype(np.int64)),
        )
