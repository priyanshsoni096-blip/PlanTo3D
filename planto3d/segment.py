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

from planto3d.classes import NUM_CLASSES, WINDOW

logger = logging.getLogger(__name__)

ENCODER = "resnet34"
DEFAULT_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A window only has to be this likely to be called one, even when another
# class scores higher.
#
# Taking the most likely class at every pixel is the right rule when the
# classes are comparable in size. Windows are not: they are 0.10% of a
# drawing, drawn about four pixels wide, and arrive at the network barely
# more than one pixel wide after the square resize. A pixel of window sits
# surrounded by wall, and wall wins the average.
#
# Measured on 190 annotated windows across 28 plans, scored as detection
# rather than per-pixel overlap -- whether an opening lands on the window
# the drawing shows:
#
#     rule                    recall   precision      F1
#     argmax                   58.4%       39.6%   0.472
#     P(window) >= 0.30        62.6%       46.7%   0.535
#   * P(window) >= 0.25        63.2%       45.3%   0.527
#     P(window) >= 0.20        60.5%       47.3%   0.531
#     P(window) >= 0.35        57.9%       40.3%   0.475
#
# Better on both counts at once, which is not the usual shape of such a
# trade: forcing the confident pixels through also cleans up the fragments
# either side of them, so fewer spurious openings survive as well as more
# real ones. Anything from 0.20 to 0.30 gives much the same answer and
# 0.35 upwards is indistinguishable from argmax, so this sits in the
# middle of the band rather than on its best single point.
#
# Applies to windows alone. No other class is thin enough to need it, and
# a floor on a large class would take pixels from its neighbours.
WINDOW_PROBABILITY_FLOOR = 0.25


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
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
            window = probabilities[WINDOW].cpu().numpy()

        # Windows are too thin to win an argmax against the wall they sit
        # in, so they are given a lower bar. See WINDOW_PROBABILITY_FLOOR.
        predicted = np.where(
            window >= WINDOW_PROBABILITY_FLOOR, WINDOW, predicted
        ).astype(np.uint8)

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
