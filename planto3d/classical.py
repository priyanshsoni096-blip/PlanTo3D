"""A segmentation baseline built from thresholds rather than a trained model.

CAD sheets fill walls and rooms with flat, distinct greys, so a classical
pass recovers them without any learning. Measured on the reference sheet:
55% of pixels are pure white background, 25% are exactly 200 (room fill),
and a 5% spike at 99 is the wall fill. Hatching, furniture outlines and text
are also dark but hairline-thin, and an erosion test separates them cleanly
-- 60% of the wall band survives a 5x5 erosion against 0% of the thin bands.

This exists for three reasons: it unblocks the rest of the pipeline before a
model is trained, it gives the trained model a baseline to be measured
against, and it is a useful fallback for clean CAD input.

Its limitation is the point of the learned model. The thresholds here are
tuned to one drawing template, and rooms drawn with hatched tile fills
rather than flat grey are missed entirely. A model trained across CubiCasa5K
should generalise where this cannot.
"""

import logging

import cv2
import numpy as np

from planto3d.classes import BACKGROUND, ROOM, WALL

logger = logging.getLogger(__name__)

# Flat fills measured from the reference sheet.
WALL_FILL = 99
ROOM_FILL = 200
# Half-width of the accepted band around each fill, covering anti-aliasing
# and mild shading differences between sheets.
FILL_TOLERANCE = 8
# Walls are solid bodies; anything that vanishes under this erosion is a
# hairline (furniture, hatching, text) and is not a wall.
SOLIDITY_KERNEL = 5
# Gaps smaller than this inside a wall body are closed -- door swing arcs and
# dimension leaders nick the fill in places.
CLOSE_KERNEL = 3
# Components smaller than this many pixels are speckle.
MIN_COMPONENT_AREA = 60


def _greyscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _near(grey: np.ndarray, level: int, tolerance: int = FILL_TOLERANCE) -> np.ndarray:
    return (np.abs(grey.astype(np.int16) - level) <= tolerance).astype(np.uint8)


def _drop_small_components(mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def wall_mask(image: np.ndarray, fill: int = WALL_FILL) -> np.ndarray:
    """Binary mask of wall bodies."""
    grey = _greyscale(image)
    candidate = _near(grey, fill)

    # Keep only pixels belonging to a solid body, then restore the body's
    # full width by dilating the surviving core back out.
    kernel = np.ones((SOLIDITY_KERNEL, SOLIDITY_KERNEL), np.uint8)
    core = cv2.erode(candidate, kernel)
    solid = cv2.dilate(core, kernel) & candidate

    closed = cv2.morphologyEx(
        solid, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    )
    return _drop_small_components(closed)


def room_mask(image: np.ndarray, fill: int = ROOM_FILL) -> np.ndarray:
    """Binary mask of room interiors, one connected region per room.

    Deliberately not morphologically closed. Room fill flows through every
    doorway, and what separates one room from the next is the hairline door
    leaf and swing arc drawn across the opening. Closing bridges exactly
    those lines: on the reference ground floor it collapsed 15 rooms into a
    single blob holding 96% of the fill.
    """
    grey = _greyscale(image)
    return _drop_small_components(_near(grey, fill))


def classical_mask(image: np.ndarray) -> np.ndarray:
    """Produce a class mask in the same format the segmentation model emits.

    Doors and windows are left as background; separating them from the wall
    fill needs symbol recognition, which is the model's job.
    """
    grey = _greyscale(image)
    mask = np.full(grey.shape, BACKGROUND, dtype=np.int64)

    mask[room_mask(grey) == 1] = ROOM
    mask[wall_mask(grey) == 1] = WALL  # walls win where fills touch

    logger.info(
        "classical mask: %.1f%% wall, %.1f%% room",
        (mask == WALL).mean() * 100,
        (mask == ROOM).mean() * 100,
    )
    return mask
