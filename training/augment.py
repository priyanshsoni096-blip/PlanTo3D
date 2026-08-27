"""Ways of showing the model a drawing it has not seen before.

There was no augmentation at all, which is the largest thing missing from
the training. A segmentation model sees 5,000 plans and learns those 5,000
plans; what makes it read a sixth thousand is having been shown each of
them drawn differently.

The transforms here are not a general-purpose kit. Each one exists because
it corresponds to something real that arrives at the pipeline and is
currently read badly:

``rotate``      Sheets arrive in any orientation. Over 30 CubiCasa plans,
                four read better turned 90 degrees than upright.
``flip``        A mirrored plan is a plan. Free variety, no downside.
``rescale``     The same building is drawn at every resolution, and the
                model had only ever seen each one at one size.
``exposure``    Scans and photographs are lighter, darker and flatter than
                the exports the dataset is made of.
``compress``    Plans arrive as messaging-app JPEGs at 40 KB, where every
                wall carries ringing artefacts along its edge.
``blur``        Photographs of drawings, and scans of photocopies.
``unfill``      Some offices draw a wall as two lines with nothing between
                them. Measured with ``scripts/convention_stress.py``, that
                convention costs 0.214 of wall IoU and takes wall recall
                from 0.899 to 0.689 -- much the worst of the eight tested,
                and the only one left that needs the model rather than the
                reader. CubiCasa fills most walls, so the model has to be
                shown the empty ones.

Rotation is by quarter turns only. A floor plan is rectilinear and the
geometry stage downstream can only recover axis-aligned walls, so teaching
the model to segment a plan rotated 30 degrees would produce masks nothing
downstream can use.

Geometry is applied to the image and the mask together; everything else
touches the image alone. Masks are resampled with nearest-neighbour
throughout -- anything smoother invents class indices that were never
labelled, blending a wall and a room into a door.
"""

import cv2
import numpy as np

from planto3d.classes import WALL

# How much of the page a random crop may take before being resized back.
# Never the whole page, or the model never sees a plan larger than it fits,
# and never so little that a crop is one room.
CROP_RANGE = (0.65, 1.0)

# Brightness and contrast, as multipliers on the image.
EXPOSURE_RANGE = (0.75, 1.25)
CONTRAST_RANGE = (0.8, 1.2)

# JPEG quality. 30 is worse than anything a phone produces, which is the
# point: the drawings that arrive worst are the ones worth training on.
JPEG_QUALITY_RANGE = (30, 85)

# Gaussian blur radius in pixels, as a share of the image's shorter side.
BLUR_RANGE = (0.0015, 0.005)

# How often each is applied.
PROBABILITIES = {
    "rotate": 0.75,
    "flip": 0.5,
    "rescale": 0.5,
    "exposure": 0.5,
    # Roughly a quarter of CubiCasa's walls are already drawn thin, so this
    # is set to bring the share of hollow-walled plans the model sees to
    # about half, without making the filled convention the unusual one.
    "unfill": 0.3,
    "compress": 0.3,
    "blur": 0.2,
}


def rotate(image: np.ndarray, mask: np.ndarray, turns: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn both by a quarter turn at a time.

    Quarter turns only. A plan is rectilinear, and the geometry stage can
    only recover axis-aligned walls, so a model taught to read a plan at 30
    degrees produces masks nothing downstream can use.
    """
    turns %= 4
    if not turns:
        return image, mask
    return np.rot90(image, turns).copy(), np.rot90(mask, turns).copy()


def flip(image: np.ndarray, mask: np.ndarray, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mirror both. A mirrored plan is a plan."""
    axis = 1 if horizontal else 0
    return np.flip(image, axis).copy(), np.flip(mask, axis).copy()


def rescale(
    image: np.ndarray, mask: np.ndarray, fraction: float, offset: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a window of the page and stretch it back to full size.

    This is what teaches scale: the same walls arrive sometimes thick and
    sometimes thin, so the model stops relying on either.
    """
    height, width = mask.shape
    crop_h, crop_w = int(height * fraction), int(width * fraction)
    if crop_h < 8 or crop_w < 8:
        return image, mask

    top = int((height - crop_h) * offset[0])
    left = int((width - crop_w) * offset[1])

    image = image[top : top + crop_h, left : left + crop_w]
    mask = mask[top : top + crop_h, left : left + crop_w]

    return (
        cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR),
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST),
    )


def exposure(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Lighten, darken and flatten, the way a scan does."""
    middle = image.mean()
    adjusted = (image.astype(np.float32) - middle) * contrast + middle * brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def compress(image: np.ndarray, quality: int) -> np.ndarray:
    """Put the image through JPEG, artefacts and all.

    Plans arrive as messaging-app JPEGs at 40 KB, where every wall carries
    ringing along its edge and the thin classes -- doors, windows -- are
    the first thing the encoder throws away.
    """
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Soften, the way a photograph of a drawing is soft."""
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def unfill_walls(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Redraw filled walls as an outline with empty space inside.

    Uses the mask to find the walls, so the outline lands exactly where the
    drawing put it and the labels stay true -- the wall is still a wall,
    it is just drawn hollow. Only the image changes.

    Where a wall is already hollow this does almost nothing, which is the
    right behaviour: CubiCasa's walls run from 0.05 to 0.95 dark-ink share
    and roughly a quarter are drawn thin already.
    """
    walls = (mask == WALL).astype(np.uint8)
    if not walls.any():
        return image

    # Leave a rim so the wall keeps its edges; empty what is inside it.
    inside = cv2.erode(walls, np.ones((3, 3), np.uint8)).astype(bool)
    if not inside.any():
        return image

    out = image.copy()
    paper = int(np.median(image[walls == 0])) if (walls == 0).any() else 255
    out[inside] = paper if out.ndim == 2 else (paper, paper, paper)
    return out


def augment(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    probabilities: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the whole set, each with its own chance.

    Order matters a little: geometry first, then how the drawing is inked,
    then the photometric steps, so those act on what the network will
    actually see -- and compression last, because that is the last thing to
    happen to a drawing before it reaches anyone.
    """
    chance = {**PROBABILITIES, **(probabilities or {})}

    if rng.random() < chance["rotate"]:
        image, mask = rotate(image, mask, int(rng.integers(1, 4)))

    if rng.random() < chance["flip"]:
        image, mask = flip(image, mask, bool(rng.integers(0, 2)))

    if rng.random() < chance["rescale"]:
        image, mask = rescale(
            image,
            mask,
            float(rng.uniform(*CROP_RANGE)),
            (float(rng.random()), float(rng.random())),
        )

    # Before the photometric steps, because how a wall is inked is part of
    # the drawing rather than something that happened to it afterwards. Left
    # until last it would come back crisp on a sheet that was otherwise
    # blurred and compressed, which is a giveaway rather than a convention.
    if rng.random() < chance["unfill"]:
        image = unfill_walls(image, mask)

    if rng.random() < chance["exposure"]:
        image = exposure(
            image, float(rng.uniform(*EXPOSURE_RANGE)), float(rng.uniform(*CONTRAST_RANGE))
        )

    if rng.random() < chance["blur"]:
        shorter = min(image.shape[:2])
        image = blur(image, float(rng.uniform(*BLUR_RANGE)) * shorter)

    if rng.random() < chance["compress"]:
        image = compress(image, int(rng.integers(*JPEG_QUALITY_RANGE)))

    return image, mask
