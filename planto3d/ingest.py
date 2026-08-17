"""PDF ingestion: rasterize architectural sheets and crop to the drawing area.

The RDA sheet template wraps each floor plan in an outer border, with an
"OPTION / CONSTRUCTION AREA" band and a title block beneath the drawing.
Everything outside the drawing is noise for segmentation, so it is cropped
away before the image reaches the model.

Borders are found from the *consensus* across all pages in a set rather than
per page. A sheet template repeats at identical coordinates on every page,
while plan content does not, so intersecting each page's candidate lines
leaves the frame and discards plan features that happen to run the full
width -- a plot boundary sitting a few pixels inside the drawing edge, for
instance. Consensus also guarantees every page crops to the same box, which
keeps floors registered to a shared origin for later stacking.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Rasterization resolution, chosen for OCR legibility of the dimension
# labels. Measured on the reference sheet, counting correctly parsed room
# dimensions and room names: 150 dpi read none, 300 dpi read 2, 400 dpi read
# 8, 600 dpi dropped back to 7 dimensions and lost most room names -- past a
# point the glyphs outgrow the size Tesseract handles best, so higher is not
# better. This is also the canonical coordinate space: the segmentation model
# runs on a downscaled copy and its mask is resized back to this resolution,
# so OCR boxes and extracted geometry share one frame.
WORKING_DPI = 400
# A pixel counts as ink below this greyscale value.
INK_THRESHOLD = 240
# A row/column counts as a border line when this fraction of it is ink.
BORDER_DENSITY = 0.85
# Border lines are searched for within these fractions of the content box.
TOP_BAND = 0.08
BOTTOM_BAND = 0.70
SIDE_BAND = 0.08
# Pixels trimmed inside a detected border so the border itself is excluded.
BORDER_INSET = 2
# Lines this close on different pages are treated as the same template line.
LINE_TOLERANCE = 2
# Lines within this many pixels belong to the same physical border.
CLUSTER_GAP = 10

Box = tuple[int, int, int, int]


def rasterize_pdf(pdf_path: Path, output_dir: Path, dpi: int = WORKING_DPI) -> list[Path]:
    """Render each PDF page to a PNG. Returns the page image paths in order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    output_paths = []
    for i, page in enumerate(pages, start=1):
        out_path = output_dir / f"page-{i}.png"
        page.save(out_path)
        output_paths.append(out_path)
    return output_paths


def _ink_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return (gray < INK_THRESHOLD).astype(np.uint8)


def _content_bbox(ink: np.ndarray) -> Box:
    """Bounding box (left, top, right, bottom) of all ink pixels."""
    cols = np.where(ink.sum(axis=0) > 0)[0]
    rows = np.where(ink.sum(axis=1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        raise ValueError("image is blank; no content to crop")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _border_lines(ink: np.ndarray, axis: int) -> np.ndarray:
    """Indices of rows (axis=1) or columns (axis=0) that read as border lines."""
    extent = ink.shape[1] if axis == 1 else ink.shape[0]
    return np.where(ink.sum(axis=axis) / extent > BORDER_DENSITY)[0]


def _common_lines(line_sets: list[np.ndarray]) -> np.ndarray:
    """Lines present in every set, within LINE_TOLERANCE."""
    if not line_sets:
        return np.array([], dtype=int)
    return np.array(
        [
            line
            for line in line_sets[0]
            if all(np.any(np.abs(other - line) <= LINE_TOLERANCE) for other in line_sets[1:])
        ],
        dtype=int,
    )


def _cluster(lines: np.ndarray) -> list[list[int]]:
    """Group adjacent lines into one cluster per physical border."""
    clusters: list[list[int]] = []
    for line in lines:
        if clusters and line - clusters[-1][-1] <= CLUSTER_GAP:
            clusters[-1].append(int(line))
        else:
            clusters.append([int(line)])
    return clusters


def _leading_edge(lines: np.ndarray, limit: float, fallback: int) -> int:
    """Innermost line of the last border cluster before ``limit``."""
    candidates = lines[lines <= limit]
    return _cluster(candidates)[-1][-1] if len(candidates) else fallback


def _trailing_edge(lines: np.ndarray, limit: float, fallback: int) -> int:
    """Outermost line of the first border cluster at or after ``limit``."""
    candidates = lines[lines >= limit]
    return _cluster(candidates)[0][-1] if len(candidates) else fallback


def detect_drawing_region(images: list[np.ndarray]) -> Box:
    """Locate the floor plan drawing shared by a set of sheets from one template.

    Returns one (left, top, right, bottom) box in absolute pixel coordinates,
    excluding the sheet border, the option/area band, and the title block.
    The same box applies to every page, so all floors share a coordinate frame.
    """
    if not images:
        raise ValueError("no images supplied")

    masks = [_ink_mask(image) for image in images]
    boxes = [_content_bbox(mask) for mask in masks]

    # Widest content box across pages, so no page loses content to the crop.
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)

    contents = [mask[top : bottom + 1, left : right + 1] for mask in masks]
    height, width = contents[0].shape

    h_lines = _common_lines([_border_lines(c, axis=1) for c in contents])
    v_lines = _common_lines([_border_lines(c, axis=0) for c in contents])

    draw_top = _leading_edge(h_lines, TOP_BAND * height, fallback=0)
    draw_bottom = _trailing_edge(h_lines, BOTTOM_BAND * height, fallback=height - 1)
    draw_left = _leading_edge(v_lines, SIDE_BAND * width, fallback=0)
    draw_right = _trailing_edge(v_lines, (1 - SIDE_BAND) * width, fallback=width - 1)

    return (
        left + draw_left + BORDER_INSET,
        top + draw_top + BORDER_INSET,
        left + draw_right - BORDER_INSET,
        top + draw_bottom - BORDER_INSET,
    )


def crop_pages(image_paths: list[Path]) -> list[Path]:
    """Crop a set of rendered sheets to their shared drawing area.

    Writes each alongside its source as ``<stem>_cropped.png``. Returns the
    written paths in the order given.
    """
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        images.append(image)

    left, top, right, bottom = detect_drawing_region(images)
    logger.info("shared drawing region: x %d-%d, y %d-%d", left, right, top, bottom)

    output_paths = []
    for path, image in zip(image_paths, images):
        cropped = image[top : bottom + 1, left : right + 1]
        out_path = path.with_name(f"{path.stem}_cropped.png")
        cv2.imwrite(str(out_path), cropped)
        logger.info(
            "cropped %s %dx%d -> %dx%d",
            path.name,
            image.shape[1],
            image.shape[0],
            cropped.shape[1],
            cropped.shape[0],
        )
        output_paths.append(out_path)
    return output_paths
