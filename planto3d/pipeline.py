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

from planto3d.calibrate import TextBox, estimate_scale, read_text_boxes
from planto3d.classical import classical_mask
from planto3d.extract import extract_footprint, extract_rooms, extract_walls
from planto3d.extrude import DEFAULT_WALL_HEIGHT_FT, export_glb, floors_to_mesh
from planto3d.geometry_types import FloorPlan, Wall
from planto3d.ingest import crop_pages, rasterize_pdf
from planto3d.label_rooms import assign_labels

logger = logging.getLogger(__name__)

Segmenter = Callable[[np.ndarray], np.ndarray]

# Extraction thresholds at the working resolution. Rooms are large areas;
# walls shorter than this are extraction noise.
MIN_WALL_LENGTH = 25
MIN_ROOM_AREA = 2000


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

    @property
    def wall_count(self) -> int:
        return sum(len(f.plan.walls) for f in self.floors)

    @property
    def room_count(self) -> int:
        return sum(len(f.plan.rooms) for f in self.floors)


def _extract_floor(index: int, image_path: Path, segmenter: Segmenter) -> FloorResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read page image: {image_path}")

    mask = segmenter(image)
    walls = extract_walls(mask, min_wall_length=MIN_WALL_LENGTH)
    rooms = extract_rooms(mask, min_area=MIN_ROOM_AREA)
    footprint = extract_footprint(mask)
    text_boxes = read_text_boxes(image)

    logger.info(
        "floor %d: %d wall(s), %d room(s), %d footprint vertices, %d text line(s)",
        index,
        len(walls),
        len(rooms),
        len(footprint),
        len(text_boxes),
    )
    return FloorResult(
        index=index,
        image_path=image_path,
        plan=FloorPlan(walls=walls, rooms=rooms, footprint=footprint),
        text_boxes=text_boxes,
    )


def run(
    pdf_path: Path,
    output_dir: Path,
    segmenter: Segmenter = classical_mask,
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
) -> PipelineResult:
    """Convert a floor plan PDF into a stacked 3D model."""
    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    pages_dir = output_dir / "pages"

    pages = rasterize_pdf(pdf_path, pages_dir)
    cropped = crop_pages(pages)

    floors = [
        _extract_floor(index, path, segmenter) for index, path in enumerate(cropped)
    ]

    # Pool every floor's rooms and text so sparsely labelled sheets still
    # benefit from the set's shared scale.
    scale = estimate_scale(
        [room for floor in floors for room in floor.plan.rooms],
        [box for floor in floors for box in floor.text_boxes],
    )

    for floor in floors:
        floor.plan.rooms = assign_labels(floor.plan.rooms, floor.text_boxes)

    model_path = None
    if scale is None:
        logger.warning("scale could not be determined; skipping 3D export")
    else:
        mesh = floors_to_mesh(
            [floor.plan for floor in floors], wall_height_ft=wall_height_ft, scale=scale
        )
        model_path = export_glb(mesh, output_dir / "house.glb")

    return PipelineResult(floors=floors, scale=scale, model_path=model_path)


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
