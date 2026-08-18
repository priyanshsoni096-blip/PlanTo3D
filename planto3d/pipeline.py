"""End-to-end: a floor plan PDF in, a 3D model out.

Stages run in order -- rasterize, crop, segment, extract, calibrate, label,
extrude -- with the segmenter swappable so the classical baseline and the
trained model can be compared on identical downstream code.

Scale is estimated once across every floor rather than per floor. Sheets in
one set share a scale, and a sparsely labelled floor cannot measure itself:
the reference terrace yields a single dimension against the ground floor's
eight.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from planto3d.calibrate import (
    ASSUMED_DRAWING_RATIO,
    TextBox,
    assumed_scale,
    estimate_scale,
    read_text_boxes,
    scale_from_doors,
    scale_from_walls,
)
from planto3d.classical import classical_mask, refine_windows, vegetation_regions
from planto3d.extract import (
    extract_footprint,
    extract_openings,
    extract_rooms,
    extract_walls,
)
from planto3d.extrude import DEFAULT_WALL_HEIGHT_FT
from planto3d.materials import build_scene, export_scene
from planto3d.geometry_types import FloorPlan, Wall
from planto3d.ingest import WORKING_DPI, crop_pages, rasterize_pdf
from planto3d.label_rooms import assign_labels

logger = logging.getLogger(__name__)

Segmenter = Callable[[np.ndarray], np.ndarray]

# Extraction thresholds at the working resolution. Rooms are large areas;
# walls shorter than this are extraction noise.
MIN_WALL_LENGTH = 25
MIN_ROOM_AREA = 2000
# Rooms are filtered again once the scale is known, in real units. A pixel
# threshold cannot tell a cupboard from a bathroom without knowing how big a
# pixel is: at the reference scale, 2000 px is 2.5 sq ft, so specks of
# segmentation noise were surviving as "rooms". The smallest genuine room on
# these plans is a WC at around 30 sq ft.
MIN_ROOM_SQFT = 12.0


@dataclass
class FloorResult:
    """One floor's extracted geometry and the image it came from."""

    index: int
    image_path: Path
    plan: FloorPlan
    text_boxes: list[TextBox] = field(default_factory=list)

    @property
    def named_rooms(self) -> list[str]:
        return [room.label for room in self.plan.rooms if room.label]


@dataclass
class PipelineResult:
    floors: list[FloorResult]
    scale: float | None
    model_path: Path | None
    # Where the scale came from, weakest last: "dimensions" is measured off
    # printed room sizes; "doors" and "walls" infer it from standard element
    # sizes; "ratio" assumes a drafting convention. Anything but "dimensions"
    # means the proportions are right but the absolute size is inferred, and
    # callers must not present it as measured.
    scale_source: str = "dimensions"

    @property
    def scale_assumed(self) -> bool:
        return self.scale_source != "dimensions"

    @property
    def wall_count(self) -> int:
        return sum(len(f.plan.walls) for f in self.floors)

    @property
    def room_count(self) -> int:
        return sum(len(f.plan.rooms) for f in self.floors)

    @property
    def opening_count(self) -> int:
        return sum(len(f.plan.openings) for f in self.floors)


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Signed area of a polygon, by the shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _extract_floor(index: int, image_path: Path, segmenter: Segmenter) -> FloorResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read page image: {image_path}")

    mask = refine_windows(segmenter(image), image)
    walls = extract_walls(mask, min_wall_length=MIN_WALL_LENGTH)
    rooms = extract_rooms(mask, min_area=MIN_ROOM_AREA)
    footprint = extract_footprint(mask)
    openings = extract_openings(mask, walls)
    planting = vegetation_regions(image)
    text_boxes = read_text_boxes(image)

    logger.info(
        "floor %d: %d wall(s), %d room(s), %d opening(s), %d text line(s)",
        index,
        len(walls),
        len(rooms),
        len(openings),
        len(text_boxes),
    )
    return FloorResult(
        index=index,
        image_path=image_path,
        plan=FloorPlan(
            walls=walls,
            rooms=rooms,
            openings=openings,
            footprint=footprint,
            planting=planting,
        ),
        text_boxes=text_boxes,
    )


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _load_pages(source: Path, pages_dir: Path) -> list[Path]:
    """Rasterize a PDF, or take image files as the pages directly.

    Plenty of plans arrive as images rather than PDFs -- exports, scans, and
    every sample in CubiCasa5K -- so requiring a PDF would rule out most of
    the world's floor plans for no good reason.
    """
    if source.is_dir():
        images = sorted(
            path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"no images found in {source}")
        logger.info("using %d image(s) from %s", len(images), source)
        return images

    if source.suffix.lower() in IMAGE_SUFFIXES:
        logger.info("using single image %s", source.name)
        return [source]

    return rasterize_pdf(source, pages_dir)


def run(
    source: Path,
    output_dir: Path,
    segmenter: Segmenter = classical_mask,
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    crop: bool = True,
) -> PipelineResult:
    """Convert a floor plan into a stacked 3D model.

    ``source`` may be a PDF, a single image, or a directory of images -- one
    per storey, in filename order.

    ``crop`` trims each sheet to its drawing frame. Turn it off for images
    that are already just the plan, with no title block to remove: the crop
    looks for borders common to every page, and on a single tight image it
    finds none and can only take away.
    """
    source, output_dir = Path(source), Path(output_dir)
    pages_dir = output_dir / "pages"

    pages = _load_pages(source, pages_dir)
    cropped = crop_pages(pages) if crop and len(pages) > 1 else pages

    floors = [
        _extract_floor(index, path, segmenter) for index, path in enumerate(cropped)
    ]

    # Pool every floor's rooms and text so sparsely labelled sheets still
    # benefit from the set's shared scale.
    scale = estimate_scale(
        [room for floor in floors for room in floor.plan.rooms],
        [box for floor in floors for box in floor.text_boxes],
    )

    # Many plans carry no dimensions at all. Rather than give up, fall back
    # through progressively weaker references, each still grounded in the
    # drawing itself before resorting to a drafting convention. This has to
    # settle before rooms can be filtered by real-world size.
    scale_source = "dimensions"
    if scale is None:
        scale = scale_from_doors(
            [opening for floor in floors for opening in floor.plan.openings]
        )
        scale_source = "doors"
    if scale is None:
        scale = scale_from_walls([wall for floor in floors for wall in floor.plan.walls])
        scale_source = "walls"
    if scale is None:
        scale = assumed_scale(WORKING_DPI)
        scale_source = "ratio"
        logger.warning(
            "nothing measurable found; assuming %.1f px/ft from a 1:%.0f ratio",
            scale,
            ASSUMED_DRAWING_RATIO,
        )

    # With a scale in hand, drop regions too small to be rooms. A pixel
    # threshold cannot tell a cupboard from a bathroom without knowing how
    # big a pixel is, so specks of segmentation noise survive extraction as
    # 4 sq ft "rooms" and go on to be labelled, railed and paved.
    minimum_px = MIN_ROOM_SQFT * scale * scale
    for floor in floors:
        kept = [
            room
            for room in floor.plan.rooms
            if abs(_polygon_area(room.polygon)) >= minimum_px
        ]
        dropped = len(floor.plan.rooms) - len(kept)
        if dropped:
            logger.info(
                "floor %d: dropped %d region(s) under %.0f sq ft",
                floor.index,
                dropped,
                MIN_ROOM_SQFT,
            )
        floor.plan.rooms = assign_labels(kept, floor.text_boxes)

    model_path = None
    if scale is None:
        logger.warning("scale could not be determined; skipping 3D export")
    else:
        # The cropped sheet is the plot: its frame encloses the whole site,
        # so the drawing's extent gives the boundary rather than a guess.
        first = cv2.imread(str(floors[0].image_path))
        page_size = (first.shape[1], first.shape[0]) if first is not None else None

        scene = build_scene(
            [floor.plan for floor in floors],
            wall_height_ft=wall_height_ft,
            scale=scale,
            page_size=page_size,
        )
        model_path = export_scene(scene, output_dir / "house.glb")

    return PipelineResult(
        floors=floors,
        scale=scale,
        model_path=model_path,
        scale_source=scale_source,
    )


def draw_overlay(floor: FloorResult) -> np.ndarray:
    """Draw extracted walls and rooms over the source page for inspection."""
    image = cv2.imread(str(floor.image_path))

    for wall in floor.plan.walls:
        cv2.line(
            image,
            (int(wall.start[0]), int(wall.start[1])),
            (int(wall.end[0]), int(wall.end[1])),
            (0, 0, 255),
            max(1, int(wall.thickness / 2)),
        )

    for room in floor.plan.rooms:
        points = np.array([[int(x), int(y)] for x, y in room.polygon], dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 170, 0), 3)
        if room.label:
            left, top, _, _ = room.bounds()
            cv2.putText(
                image,
                room.label,
                (int(left) + 6, int(top) + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 0, 0),
                2,
            )

    return image
