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

from planto3d.classes import BACKGROUND, ROOM, WALL, WINDOW

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
# Coloured ink on an otherwise greyscale sheet. Green marks planting; the
# only other saturated family is the cyan used for glazing.
COLOUR_SATURATION = 60
COLOUR_VALUE = 60
GREEN_HUE = (35, 85)
CYAN_HUE = (85, 115)
# A window is a long thin band; planting symbols are round. Shape alone
# separates the two coloured families.
MIN_WINDOW_ELONGATION = 3.0
MIN_WINDOW_AREA = 60
# Bridges the mullions dividing one window run. Sized for a 5-inch mullion at
# the working resolution; separate windows sit feet apart, so this cannot
# merge two of them.
WINDOW_CLOSE = 13
# Planting is drawn as scattered symbols, closed into continuous beds.
PLANTING_CLOSE = 35
PLANTING_SIMPLIFY = 6.0
MIN_PLANTING_AREA = 2500


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


def vegetation_regions(image: np.ndarray, min_area: int = MIN_PLANTING_AREA) -> list[list[tuple[float, float]]]:
    """Outlines of the planted areas, from the green ink on the drawing.

    Architectural sheets are otherwise greyscale, so saturated colour is a
    reliable signal. Measured on the reference sheet, the only two coloured
    families are a green one -- lawn hatching and tree symbols, appearing as
    compact blobs -- and a cyan one marking glazing, which is far too
    elongated to be confused with planting.

    This complements the segmentation model rather than competing with it:
    the model is trained on interiors and does not label a garden as anything
    at all, since a garden is not a room.
    """
    if image.ndim != 3:
        return []

    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    green = (
        (saturation > COLOUR_SATURATION)
        & (value > COLOUR_VALUE)
        & (hue >= GREEN_HUE[0])
        & (hue <= GREEN_HUE[1])
    ).astype(np.uint8)

    if not green.any():
        return []

    # Planting is drawn as scattered symbols; close them into continuous beds.
    merged = cv2.morphologyEx(
        green, cv2.MORPH_CLOSE, np.ones((PLANTING_CLOSE, PLANTING_CLOSE), np.uint8)
    )
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        simplified = cv2.approxPolyDP(contour, PLANTING_SIMPLIFY, closed=True)
        if len(simplified) >= 3:
            regions.append([(float(p[0][0]), float(p[0][1])) for p in simplified])

    logger.info("found %d planted region(s)", len(regions))
    return regions


def window_mask(image: np.ndarray, min_area: int = MIN_WINDOW_AREA) -> np.ndarray:
    """Binary mask of window openings, from the cyan strips on the drawing.

    Windows are drawn as long thin bands across a wall's thickness, which
    makes them separable from planting by shape alone: measured on the
    reference sheet, the cyan runs average 70 times longer than they are
    wide, while planting symbols are round.

    Far more reliable than the model on these sheets. Windows are a tenth of
    a percent of CubiCasa's pixels, and the trained model scores 0.12 IoU on
    them -- it finds roughly twice as many blobs as there are windows, most
    of them spurious.
    """
    empty = np.zeros(image.shape[:2], np.uint8)
    if image.ndim != 3:
        return empty

    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    cyan = (
        (saturation > COLOUR_SATURATION)
        & (value > COLOUR_VALUE)
        & (hue >= CYAN_HUE[0])
        & (hue <= CYAN_HUE[1])
    ).astype(np.uint8)
    if not cyan.any():
        return empty

    # A window run is broken by its mullions; close them into one opening.
    joined = cv2.morphologyEx(
        cyan, cv2.MORPH_CLOSE, np.ones((WINDOW_CLOSE, WINDOW_CLOSE), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    kept = empty
    found = 0
    for index in range(1, count):
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        if max(width, height) / max(min(width, height), 1) < MIN_WINDOW_ELONGATION:
            continue
        kept[labels == index] = 1
        found += 1

    logger.info("found %d window strip(s)", found)
    return kept


def refine_windows(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Replace the model's windows with ones read from the drawing's colour.

    Only when colour finds any. On a sheet that does not mark windows in
    colour there is nothing to substitute, and the model's guess -- however
    weak -- is better than none.
    """
    strips = window_mask(image)
    if not strips.any():
        return mask

    refined = mask.copy()
    refined[refined == WINDOW] = BACKGROUND
    refined[strips == 1] = WINDOW
    return refined


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
