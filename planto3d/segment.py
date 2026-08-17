"""Run the trained U-Net over a floor plan.

Drops into the pipeline wherever the classical baseline goes, so both can be
compared on identical downstream code.

The network takes a fixed square input, but the pipeline works in the page's
own pixel frame -- OCR boxes, wall coordinates and room polygons all share
it. The mask is therefore resized back to the page before being returned,
using nearest-neighbour: anything smoother invents class indices the model
never predicted, blending a wall and a room into a door.

torch is imported lazily so the geometry stages, the classical baseline and
the tests around them stay usable without a multi-gigabyte install.
"""

import logging
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import NUM_CLASSES

logger = logging.getLogger(__name__)

ENCODER = "resnet34"
DEFAULT_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Segmenter:
    """Loads a checkpoint and turns page images into class masks."""

    def __init__(self, checkpoint_path: Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"no checkpoint at {self.checkpoint_path}")
        self._requested_device = device

    @cached_property
    def _loaded(self):
        import torch

        device = torch.device(
            self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(self.checkpoint_path, map_location=device, weights_only=False)

        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=state.get("encoder", ENCODER),
            encoder_weights=None,  # weights come from the checkpoint
            in_channels=3,
            classes=state.get("num_classes", NUM_CLASSES),
        )
        model.load_state_dict(state["model_state"])
        model.eval().to(device)

        size = int(state.get("size", DEFAULT_SIZE))
        logger.info(
            "loaded %s on %s (input %dpx, val dice %s)",
            self.checkpoint_path.name,
            device,
            size,
            state.get("val_dice"),
        )
        return model, device, size

    @property
    def input_size(self) -> int:
        return self._loaded[2]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Class-index mask at the input image's own resolution."""
        import torch

        model, device, size = self._loaded

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else (
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        )
        height, width = rgb.shape[:2]

        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        batch = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted = model(batch).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Back to the page's frame, which the rest of the pipeline works in.
        return cv2.resize(
            predicted, (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)


def load_segmenter(checkpoint_path: Path | None):
    """Return the trained segmenter, or the classical baseline when absent."""
    if checkpoint_path is None:
        from planto3d.classical import classical_mask

        logger.info("no checkpoint given; using the classical baseline")
        return classical_mask

    return Segmenter(checkpoint_path)
