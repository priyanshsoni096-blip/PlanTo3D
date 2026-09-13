"""Correcting window annotations without building an interface.

Windows are the weakest thing the segmenter reads, and the evidence is that
the cause is the training data: windows are about 0.1% of CubiCasa's
annotated pixels, and the same weights score 2.7 times better on a corpus
they never saw. The one untried remedy is better window annotations, and
drawing them from nothing is slow. Correcting the model's own predictions is
not.

So there is no tool here to draw with. The model's WINDOW pixels are written
out as rectangles in LabelMe's format, beside each image and under the name
LabelMe opens automatically. A person corrects them in LabelMe -- a free,
existing application -- and ticks "reviewed". Training then uses the
corrected rectangles in place of the annotation's windows, but only for
files that carry that tick: an untouched export is the model's own guess, and
training on it would teach the model its own mistakes.

Corrections are made beside a local copy of the dataset, while training on
Colab downloads a fresh copy that has none of them, so reviewed files travel
in a zip: ``bundle_reviewed`` on the laptop, ``unpack_bundle`` in the
notebook.
"""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from planto3d.classes import WALL, WINDOW

LABEL = "window"

# Components smaller than this are speckle from the argmax, not a window a
# person would draw, and exporting them would bury the real ones in boxes to
# delete.
MIN_WINDOW_PIXELS = 4

# The LabelMe version whose file layout this writes. Later versions read it.
LABELME_VERSION = "5.4.1"


def labels_path(image_path: Path) -> Path:
    """Where LabelMe looks for an image's labels: beside it, same stem."""
    return Path(image_path).with_suffix(".json")


def mask_to_labelme(mask: np.ndarray, image_name: str) -> dict:
    """One rectangle per predicted window, unreviewed."""
    binary = (mask == WINDOW).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    shapes = []
    for index in range(1, count):
        left, top, width, height, area = (int(v) for v in stats[index])
        if area < MIN_WINDOW_PIXELS:
            continue
        shapes.append({
            "label": LABEL,
            # Inclusive corners, as LabelMe stores a rectangle.
            "points": [[left, top], [left + width - 1, top + height - 1]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        })

    return {
        "version": LABELME_VERSION,
        "flags": {"reviewed": False},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(mask.shape[0]),
        "imageWidth": int(mask.shape[1]),
    }


def apply_window_labels(mask: np.ndarray, labels: dict) -> np.ndarray:
    """Replace the mask's windows with the labelled rectangles.

    Every existing window pixel goes back to wall first. CubiCasa paints
    windows last, over the wall they sit in (``cubicasa.PAINT_ORDER``), so
    wall is what lies underneath a window the person deleted.
    """
    mask = mask.copy()
    mask[mask == WINDOW] = WALL
    height, width = mask.shape[:2]

    for shape in labels.get("shapes", []):
        if shape.get("label") != LABEL or shape.get("shape_type") != "rectangle":
            continue
        (x_a, y_a), (x_b, y_b) = shape["points"][:2]
        # LabelMe keeps the order the rectangle was dragged in.
        left, right = sorted((int(round(x_a)), int(round(x_b))))
        top, bottom = sorted((int(round(y_a)), int(round(y_b))))
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, width - 1), min(bottom, height - 1)
        if right < left or bottom < top:
            continue
        mask[top:bottom + 1, left:right + 1] = WINDOW
    return mask


def load_reviewed(image_path: Path) -> dict | None:
    """An image's labels, only if a person has ticked them reviewed."""
    path = labels_path(image_path)
    if not path.is_file():
        return None
    labels = json.loads(path.read_text(encoding="utf-8"))
    if labels.get("flags", {}).get("reviewed") is not True:
        return None
    return labels


def adjust_mask(mask: np.ndarray, image_path: Path) -> np.ndarray:
    """The mask training should use: corrected where a review exists."""
    labels = load_reviewed(image_path)
    return mask if labels is None else apply_window_labels(mask, labels)


def bundle_reviewed(image_paths, root: Path, bundle_path: Path) -> int:
    """Zip every reviewed label file, keyed by its path under the dataset root.

    Unreviewed files stay behind for the same reason training ignores them.
    """
    root = Path(root).resolve()
    count = 0
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for image in image_paths:
            if load_reviewed(image) is None:
                continue
            label = labels_path(image).resolve()
            archive.write(label, label.relative_to(root).as_posix())
            count += 1
    return count


def unpack_bundle(bundle_path: Path, data_root: Path) -> int:
    """Put a bundle's label files back beside their plans under ``data_root``.

    Only ``.json`` entries are written, so a bundle can never replace an
    image or annotation, and every entry is checked before any is written:
    one that would land outside the dataset -- ``../`` in its name -- stops
    the whole unpack rather than writing the others first.
    """
    root = Path(data_root).resolve()
    with zipfile.ZipFile(bundle_path) as archive:
        entries = [name for name in archive.namelist() if name.endswith(".json")]
        targets = []
        for name in entries:
            target = (root / name).resolve()
            if root not in target.parents:
                raise ValueError(f"refusing {name!r}: it would write outside {root}")
            targets.append((name, target))
        for name, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return len(targets)
