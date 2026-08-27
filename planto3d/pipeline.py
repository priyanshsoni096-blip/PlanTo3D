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
from statistics import median
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from planto3d.calibrate import (
    ASSUMED_DRAWING_RATIO,
    TextBox,
    assumed_scale,
    corroborated,
    estimate_scale,
    read_text_boxes,
    scale_from_areas,
    scale_from_doors,
    scale_from_gauge,
    scale_from_walls,
)
from planto3d.classical import classical_mask, refine_windows, vegetation_regions
from planto3d.extract import (
    close_envelope,
    wall_gauge,
    extract_footprint,
    extract_openings,
    extract_rooms,
    extract_walls,
)
from planto3d.extrude import DEFAULT_WALL_HEIGHT_FT
from planto3d.features import regions_from_labels
from planto3d.materials import build_scene, export_scene
from planto3d.design import Landscaping
from planto3d.style import Palette
from planto3d.geometry_types import FloorPlan, Wall
from planto3d.ingest import (
    WORKING_DPI,
    crop_pages,
    rasterize_pdf,
    read_image,
    split_sheet,
)
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


# Scale sources that are the drawing's own statement of its size rather
# than something inferred from it. Everything else means the proportions
# are right and the absolute size is an estimate.
PRINTED_SCALE_SOURCES = frozenset({"dimensions", "areas"})


@dataclass
class FloorResult:
    """One floor's extracted geometry and the image it came from."""

    index: int
    image_path: Path
    plan: FloorPlan
    text_boxes: list[TextBox] = field(default_factory=list)
    # How many of ``plan.walls`` were read off the drawing. The rest were
    # invented by envelope closing at the assumed scale, so measuring them
    # to find the scale just returns the assumption.
    drawn_wall_count: int | None = None
    # The drawing's own wall thickness, measured off the mask. A truer
    # figure than the extracted walls give: those have been through
    # orientation filtering and merging first, and both erode.
    wall_gauge_px: float | None = None

    @property
    def drawn_walls(self) -> list:
        """The walls the drawing actually shows."""
        if self.drawn_wall_count is None:
            return self.plan.walls
        return self.plan.walls[: self.drawn_wall_count]

    @property
    def named_rooms(self) -> list[str]:
        return [room.label for room in self.plan.rooms if room.label]


@dataclass
class PipelineResult:
    floors: list[FloorResult]
    scale: float | None
    model_path: Path | None
    # Where the scale came from, weakest last: "dimensions" is measured off
    # printed room sizes and "areas" off a printed floor area; "doors" and
    # "walls" infer it from standard element sizes; "ratio" assumes a
    # drafting convention. Anything the drawing did not state means the
    # proportions are right but the absolute size is inferred, and callers
    # must not present it as measured.
    scale_source: str = "dimensions"

    @property
    def scale_assumed(self) -> bool:
        return self.scale_source not in PRINTED_SCALE_SOURCES

    @property
    def wall_count(self) -> int:
        return sum(len(f.plan.walls) for f in self.floors)

    @property
    def room_count(self) -> int:
        return sum(len(f.plan.rooms) for f in self.floors)

    @property
    def opening_count(self) -> int:
        return sum(len(f.plan.openings) for f in self.floors)


def _split_into_storeys(page: Path, output_dir: Path) -> list[Path]:
    """Split a sheet carrying several plans, writing one image per storey.

    Left in sheet order, which reads basement to top floor on the drawings
    that do this -- the same order the storeys stack in.
    """
    image = read_image(page)
    if image is None:
        return [page]

    pieces = split_sheet(image)
    if len(pieces) < 2:
        return [page]

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, piece in enumerate(pieces):
        path = output_dir / f"{page.stem}-storey-{index}.png"
        cv2.imwrite(str(path), piece)
        written.append(path)

    logger.info("split %s into %d storey(s)", page.name, len(written))
    return written


# Thinnest wall the geometry can work with. Below this the orientation
# opening cannot separate one wall from another, a door cannot be told
# from a fragment, and the scale that comes out is a fraction of the truth.
MIN_WORKABLE_GAUGE = 10.0

# Never enlarged past this, where the cost stops buying anything.
MAX_ENLARGEMENT = 4.0


def _enlarge_if_unmeasurable(image, mask, segmenter: Segmenter):
    """Enlarge and re-read a drawing whose walls are too thin to measure.

    Costs a second pass over the segmenter, which is why it is conditional:
    on any drawing of ordinary resolution the gauge is already fine and
    this returns immediately.
    """
    gauge = wall_gauge(mask)
    if gauge >= MIN_WORKABLE_GAUGE:
        return image, mask

    factor = min(MIN_WORKABLE_GAUGE / max(gauge, 1.0), MAX_ENLARGEMENT)
    height, width = image.shape[:2]
    logger.info(
        "walls measure %.0f px, too thin to work with; enlarging %.1fx", gauge, factor
    )

    # Cubic, not nearest: this runs on the drawing rather than on a mask,
    # and the network reads a smooth enlargement better than a blocky one.
    enlarged = cv2.resize(
        image,
        (int(round(width * factor)), int(round(height * factor))),
        interpolation=cv2.INTER_CUBIC,
    )
    return enlarged, refine_windows(segmenter(enlarged), enlarged)


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Signed area of a polygon, by the shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _extract_floor(index: int, image_path: Path, segmenter: Segmenter) -> FloorResult:
    image = read_image(image_path)
    if image is None:
        raise FileNotFoundError(f"could not read page image: {image_path}")

    mask = refine_windows(segmenter(image), image)

    # Whether a drawing is big enough is a question about the wall, not
    # about the page. A CubiCasa sheet 650 pixels across is small but its
    # walls are twenty pixels thick, and everything downstream works; a web
    # image of a house is 500 pixels across with four pixel walls, and
    # every threshold in the geometry sits above them at once.
    #
    # So it is asked after segmenting rather than before, and answered by
    # the gauge. Enlarging invents no detail; it lifts what detail there is
    # back above the floors the geometry cannot go below.
    image, mask = _enlarge_if_unmeasurable(image, mask, segmenter)
    # Measured once and shared, so every stage sizes itself against the same
    # figure -- and so it can be reported later, since the wall thickness is
    # also the weakest of the scale references.
    #
    # Nothing is passed explicitly here. Doing so pinned each stage to sizes
    # measured on one drawing at one resolution, which is what made the same
    # plan reconstruct differently depending only on how large it had been
    # rendered.
    gauge = wall_gauge(mask)
    walls = extract_walls(mask, gauge=gauge)
    rooms = extract_rooms(mask, gauge=gauge)
    footprint = extract_footprint(mask, gauge=gauge)

    # Segmentation loses stretches of exterior wall wherever the drawing is
    # busy, leaving holes in the facade -- and every window that would have
    # bound to the missing wall is lost with it.
    #
    # The real scale is not known until every floor has been read, so the
    # nominal one implied by the rasterization resolution is used here. Its
    # job is only to size tolerances -- how long a gap must be to count, how
    # far to probe inward -- and those survive being a few percent out.
    #
    # How many walls were read off the drawing is recorded before the
    # invented ones are added. They are not evidence of anything: drawn at
    # the assumed scale, measuring them to find the scale simply returns
    # the assumption, and with enough of them the estimate collapsed onto
    # exactly 32.0 px/ft -- the assumed figure -- plan after plan.
    drawn_wall_count = len(walls)
    walls += close_envelope(
        mask, walls, footprint, scale=assumed_scale(WORKING_DPI)
    )

    openings = extract_openings(mask, walls, gauge=gauge)
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
        drawn_wall_count=drawn_wall_count,
        wall_gauge_px=gauge,
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
    palette: Palette | None = None,
    site: Landscaping | None = None,
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

    # A single sheet often carries several plans side by side. Read as one
    # storey it reconstructs several buildings as one flat floor, so each
    # sheet is split before anything else looks at it.
    if len(cropped) == 1:
        cropped = _split_into_storeys(cropped[0], pages_dir)

    floors = [
        _extract_floor(index, path, segmenter) for index, path in enumerate(cropped)
    ]

    # What the drawing measures on its own geometry. Worked out first
    # because it is also what a printed number is checked against, and
    # because it is the fallback when there is no printed number to read.
    gauges = [floor.wall_gauge_px for floor in floors if floor.wall_gauge_px]
    gauge = float(median(gauges)) if gauges else None

    reference = scale_from_doors(
        [opening for floor in floors for opening in floor.plan.openings],
        gauge=gauge,
    )
    reference_source = "doors"
    if reference is None:
        reference = (
            scale_from_gauge(gauge)
            if gauge
            else scale_from_walls(
                [wall for floor in floors for wall in floor.drawn_walls]
            )
        )
        reference_source = "walls"

    # What the drawing says about itself. Better evidence than anything
    # inferred from standard element sizes -- a printed "13'0\" x 10'0\"" or
    # "2130 SQ.FT." is the architect's own statement of size, where a door
    # width is an assumption about doors in general.
    #
    # Pooled across floors, so sparsely labelled sheets still benefit from
    # the set's shared scale: the reference terrace sheet yields one
    # dimension on its own against eight from its ground floor.
    rooms = [room for floor in floors for room in floor.plan.rooms]
    boxes = [box for floor in floors for box in floor.text_boxes]

    printed = estimate_scale(rooms, boxes)
    printed_source = "dimensions"
    if printed is None:
        printed = scale_from_areas(rooms, boxes)
        printed_source = "areas"

    # But only if it survives the check. OCR on a busy drawing misreads a
    # foot mark, drops a digit, or takes a door tag for a room size, and an
    # ungated printed number resizes the entire building on one bad read.
    # Agreement means the text was read correctly and refines the answer;
    # disagreement means it was not, and the geometry stands.
    scale = None
    scale_source = "ratio"
    if printed is not None and corroborated(printed, reference):
        scale, scale_source = printed, printed_source
    elif reference is not None:
        scale, scale_source = reference, reference_source

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
        # Outdoor areas rarely survive as rooms, but their labels state both
        # the name and the size, so they can be placed from the text alone.
        floor.plan.labelled_regions = regions_from_labels(floor.text_boxes, scale)

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
            palette=palette,
            site=site,
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
