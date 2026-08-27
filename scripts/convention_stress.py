"""How much does a change of drafting convention cost the segmenter?

Every figure in `docs/AUDIT.md` rests on CubiCasa5K, which is one drafting
tradition, so the honest question -- what happens on a drawing office that
works differently -- has no answer. Sourcing a second corpus is the real
answer and is expensive. This is the cheap one: take the sheets we have
and redraw them the way other conventions draw them, holding the
annotation fixed so the ground truth stays valid.

Only the *rendering* is changed, never the geometry. The walls are in the
same places, the rooms are the same rooms; what differs is how they are
inked. So any drop is caused by the convention alone, and the size of the
drop says how much a second corpus would be worth.

This is not a substitute for real data. A convention differs in more ways
than can be simulated -- symbol vocabulary, annotation habits, what is
drawn at all -- and a transform written by the same person who reads the
result is not an independent test. Treat these as a lower bound on the
damage, not an estimate of it.

    python scripts/convention_stress.py <corpus> --checkpoint models/unet_cubicasa.pt
"""

import argparse
import logging
import statistics
import warnings
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import ROOM_CLASSES, WALL
from planto3d.cubicasa import svg_to_mask
from planto3d.extract import extract_rooms, extract_walls
from planto3d.segment import load_segmenter

# A sheet still counts as reconstructable with at least this much geometry.
MIN_WALLS = 8
MIN_ROOMS = 3


def _wall_region(truth: np.ndarray) -> np.ndarray:
    return (truth == WALL).astype(np.uint8)


def as_drawn(image, truth):
    return image


def hatched_walls(image, truth, spacing=9):
    """Outline kept, interior emptied and filled with 45-degree hatching.

    How a structural wall is drawn in conventions that distinguish it from
    a partition. CubiCasa fills walls solid, so the model has not seen it.
    """
    out = image.copy()
    inner = cv2.erode(_wall_region(truth), np.ones((3, 3), np.uint8)).astype(bool)
    out[inner] = (255, 255, 255)
    height, width = truth.shape
    lines = np.zeros((height, width), np.uint8)
    for offset in range(-height, width, spacing):
        cv2.line(lines, (offset, 0), (offset + height, height), 255, 1)
    out[(lines > 0) & inner] = (30, 30, 30)
    return out


def outline_walls(image, truth):
    """Walls as two parallel lines with nothing between them."""
    out = image.copy()
    inner = cv2.erode(_wall_region(truth), np.ones((3, 3), np.uint8)).astype(bool)
    out[inner] = (255, 255, 255)
    return out


def poche_walls(image, truth):
    """Walls filled solid black, the heaviest convention."""
    out = image.copy()
    out[_wall_region(truth).astype(bool)] = (0, 0, 0)
    return out


def blueprint(image, truth):
    """White on a dark ground, as a reversed print."""
    return 255 - image


def toned_paper(image, truth, level=200):
    """Drawn on tinted stock rather than white, as a scan of an old print."""
    out = image.astype(np.float32)
    out = out * (level / 255.0)
    return out.astype(np.uint8)


def thin_lines(image, truth):
    """A finer pen: every stroke eroded by a pixel."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    thinned = cv2.dilate(grey, np.ones((3, 3), np.uint8))  # dilating light = thinning ink
    return cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)


def heavy_lines(image, truth):
    """A broader pen: every stroke thickened."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.cvtColor(cv2.erode(grey, np.ones((3, 3), np.uint8)), cv2.COLOR_GRAY2BGR)


def scanned(image, truth):
    """Photocopied and rescanned: speckle, softness, compression."""
    out = cv2.GaussianBlur(image, (3, 3), 0)
    noise = np.random.default_rng(7).normal(0, 9, out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else out


TRANSFORMS = [
    ("as drawn", as_drawn),
    ("hatched walls", hatched_walls),
    ("outline walls", outline_walls),
    ("solid poche walls", poche_walls),
    ("heavier pen", heavy_lines),
    ("finer pen", thin_lines),
    ("toned paper", toned_paper),
    ("photocopied", scanned),
    ("reversed print", blueprint),
]


def score(mask, truth):
    walls_p, walls_t = mask == WALL, truth == WALL
    rooms_p = np.isin(mask, list(ROOM_CLASSES))
    rooms_t = np.isin(truth, list(ROOM_CLASSES))
    if not walls_t.any():
        return None

    def iou(a, b):
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else 0.0

    return {
        "wall_iou": iou(walls_p, walls_t),
        "wall_recall": float(np.logical_and(walls_p, walls_t).sum() / walls_t.sum()),
        "room_iou": iou(rooms_p, rooms_t),
    }


def main(root: str, checkpoint: Path | None, limit: int) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    segmenter = load_segmenter(checkpoint)

    sheets = []
    for path in sorted(Path(root).glob("*/*/F1_scaled.png"))[:limit]:
        image = cv2.imread(str(path))
        svg = path.parent / "model.svg"
        if image is None or not svg.is_file():
            continue
        truth = svg_to_mask(svg, image.shape[:2])
        if (truth == WALL).sum() < 500:
            continue
        sheets.append((image, truth))

    print(f"{len(sheets)} sheets, {len(TRANSFORMS)} conventions\n")
    print(f"{'convention':22}{'wall IoU':>10}{'wall recall':>13}"
          f"{'room IoU':>10}{'reconstructs':>14}")
    print("-" * 69)

    baseline = None
    for name, transform in TRANSFORMS:
        walls, recalls, rooms, built = [], [], [], 0
        for image, truth in sheets:
            mask = segmenter(transform(image, truth))
            result = score(mask, truth)
            if result is None:
                continue
            walls.append(result["wall_iou"])
            recalls.append(result["wall_recall"])
            rooms.append(result["room_iou"])
            if (len(extract_walls(mask)) >= MIN_WALLS
                    and len(extract_rooms(mask)) >= MIN_ROOMS):
                built += 1
        if not walls:
            continue
        row = (statistics.median(walls), statistics.median(recalls),
               statistics.median(rooms), built)
        if baseline is None:
            baseline = row
            mark = ""
        else:
            mark = f"  ({row[0]-baseline[0]:+.3f} wall IoU)"
        print(f"{name:22}{row[0]:>10.3f}{row[1]:>13.3f}{row[2]:>10.3f}"
              f"{row[3]:>9}/{len(sheets):<4}{mark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=15)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.limit)
