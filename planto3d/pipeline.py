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
from planto3d.classical import classical_mask, vegetation_regions
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

# How far the two geometric scale estimates may sit apart before the
# drawing is treated as having given inconsistent evidence about its size.
#
# Swept across 30 plans carrying both estimates, scoring how far apart the
# two groups' errors end up -- which is the only thing a flag like this is
# for. The estimates usually agree closely: the median disagreement is
# 0.062 and the 90th percentile 0.207, so this line sits well out in the
# tail and flags the few plans that argue with themselves.
#
#     threshold   flagged   error if confident   if flagged   gap
#        0.10      11/30           10.8%            17.6%     6.8
#      * 0.14       6/30           11.1%            21.4%    10.3
#        0.18       5/30           11.5%            17.6%     6.1
#        0.26       2/30           12.3%            17.5%     5.2
MAX_SCALE_DISAGREEMENT = 0.14


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
    # How far apart the two geometric estimates were, as a share of the
    # larger: 0 is perfect agreement, None when only one could be worked
    # out. It says nothing about which was chosen -- only whether the
    # drawing gave consistent evidence about its own size.
    scale_agreement: float | None = None
    # The cropped sheet's pixel dimensions, needed by ``build_scene`` to
    # place the site boundary. Computed once in ``extract`` and carried
    # through so ``build`` does not need to re-read the image; defaulted
    # so existing callers that construct a ``PipelineResult`` directly
    # (as the tests do) do not need to supply it.
    page_size: tuple[int, int] | None = None

    @property
    def scale_assumed(self) -> bool:
        return self.scale_source not in PRINTED_SCALE_SOURCES

    @property
    def scale_confident(self) -> bool:
        """Whether the drawing's own evidence about its size was consistent.

        A printed dimension is the architect saying so and is believed
        outright. Otherwise the two geometric estimates have to agree:
        measured over 30 plans, the ones where they agree come out 11.1%
        from true at the median and the ones that do not, 21.4%.

        False does not mean the size is wrong, and True does not mean it is
        right -- most of the residual error is the drawing being drawn to
        standards the code does not know it uses, which no amount of
        internal agreement can detect. It means the evidence held together.
        """
        if self.scale_source in PRINTED_SCALE_SOURCES:
            return True
        if self.scale_agreement is None:
            return False
        return self.scale_agreement <= MAX_SCALE_DISAGREEMENT

    @property
    def wall_count(self) -> int:
        return sum(len(f.plan.walls) for f in self.floors)

    @property
    def room_count(self) -> int:
        return sum(len(f.plan.rooms) for f in self.floors)

    @property
    def opening_count(self) -> int:
        return sum(len(f.plan.openings) for f in self.floors)


def _split_into_storeys(
    page: Path, output_dir: Path, force: int | None = None
) -> list[Path]:
    """Split a sheet carrying several plans, writing one image per storey.

    Left in sheet order, which reads basement to top floor on the drawings
    that do this -- the same order the storeys stack in.
    """
    image = read_image(page)
    if image is None:
        return [page]

    pieces = split_sheet(image, force=force)
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
    return enlarged, segmenter(enlarged)


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

    # The segmenter's own answer, unrefined.
    #
    # `refine_windows` used to run here, replacing the model's windows with
    # ones read from the drawing's colour wherever colour found enough
    # strips to be trusted. Measured against the annotations over 28 plans
    # it was doing severe harm, and worst exactly where it claimed to help:
    #
    #     what it did to the sheet            plans   F1 model   F1 after
    #     colour trusted, model wiped            12      0.509      0.051
    #     colour merged in                        7      0.406      0.397
    #     no colour found                         9      0.600      0.600
    #
    # Overall it took window detection from 62.1% recall at 43.5%
    # precision to 41.6% at 26.7%. It hurt both subsets, the colourful one
    # worst. Its premise -- that colour beats the model on sheets that mark
    # glazing in colour -- was true of a much weaker model and is not true
    # of this one.
    #
    # The function stays in `classical` where the baseline still uses it.
    mask = segmenter(image)

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


# Anything not on this list is handed to the PDF rasterizer, which is the
# right default and a poor failure: a plan in an unlisted format comes back
# as "unable to get page count" rather than as a model. GIF earned its
# place the hard way -- every one of BRIDGE's 2,400 Indian plans is a GIF,
# and all 60 of a sample failed on that line while `read_image` was reading
# them perfectly well.
IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".gif",
    ".ppm",
    ".pgm",
}


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


def extract(
    source: Path,
    output_dir: Path,
    segmenter: Segmenter = classical_mask,
    crop: bool = True,
    split: int | None = None,
) -> PipelineResult:
    """Read a floor plan and extract its geometry, scale and room labels.

    ``source`` may be a PDF, a single image, or a directory of images -- one
    per storey, in filename order.

    ``crop`` trims each sheet to its drawing frame. Turn it off for images
    that are already just the plan, with no title block to remove: the crop
    looks for borders common to every page, and on a single tight image it
    finds none and can only take away.

    ``split`` overrides the sheet splitter: 1 keeps the sheet whole, N forces
    N plans, None reads it automatically.

    Everything through labeling -- rasterize, crop, segment, extract walls/
    rooms/openings, calibrate scale, assign labels -- with no scene built
    yet. This is the pause point: a caller can inspect or edit
    ``result.floors[i].plan.rooms[j].label`` before calling ``build()``,
    which is exactly what a human-correction step needs.
    """
    source, output_dir = Path(source), Path(output_dir)
    pages_dir = output_dir / "pages"

    pages = _load_pages(source, pages_dir)
    cropped = crop_pages(pages) if crop and len(pages) > 1 else pages

    # A single sheet often carries several plans side by side. Read as one
    # storey it reconstructs several buildings as one flat floor, so each
    # sheet is split before anything else looks at it.
    if len(cropped) == 1:
        cropped = _split_into_storeys(cropped[0], pages_dir, force=split)

    floors = [
        _extract_floor(index, path, segmenter) for index, path in enumerate(cropped)
    ]

    # What the drawing measures on its own geometry. Worked out first
    # because it is also what a printed number is checked against, and
    # because it is the fallback when there is no printed number to read.
    gauges = [floor.wall_gauge_px for floor in floors if floor.wall_gauge_px]
    gauge = float(median(gauges)) if gauges else None

    from_doors = scale_from_doors(
        [opening for floor in floors for opening in floor.plan.openings],
        gauge=gauge,
    )
    from_walls = (
        scale_from_gauge(gauge)
        if gauge
        else scale_from_walls([wall for floor in floors for wall in floor.drawn_walls])
    )

    reference, reference_source = (
        (from_doors, "doors") if from_doors is not None else (from_walls, "walls")
    )

    # Both are worked out even though only one is used, because how far
    # apart they are says something the chosen one cannot. Doors and the
    # wall gauge measure different things and are wrong in different ways,
    # so when they agree the answer is usually good and when they do not,
    # something in the drawing is being misread. Over 30 plans carrying
    # both, the door estimate is 10.8% out at the median where they agree
    # and 17.6% out where they do not.
    #
    # Combining them was tried and does not help -- every blend scores
    # 24 of 30 within a fifth against the 25 that doors alone manages, so
    # the choice above is unchanged and only the confidence is new.
    agreement = None
    if from_doors and from_walls:
        agreement = abs(from_doors - from_walls) / max(from_doors, from_walls)

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

    # The cropped sheet is the plot: its frame encloses the whole site, so
    # the drawing's extent gives the boundary rather than a guess. Computed
    # here and carried on the result so ``build`` does not need to re-read
    # the image later.
    first = cv2.imread(str(floors[0].image_path))
    page_size = (first.shape[1], first.shape[0]) if first is not None else None

    return PipelineResult(
        floors=floors,
        scale=scale,
        model_path=None,
        scale_source=scale_source,
        scale_agreement=agreement,
        page_size=page_size,
    )


def build(
    result: PipelineResult,
    output_dir: Path,
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    palette: Palette | None = None,
    site: Landscaping | None = None,
) -> PipelineResult:
    """Build and export the 3D scene from an already-extracted result.

    Mutates and returns ``result`` with ``model_path`` set. Split out of
    what used to be the tail of ``run()`` so a correction step can edit
    ``result.floors[i].plan.rooms[j].label`` in between extraction and
    this call -- a correction is just a label that did not come from OCR,
    and every downstream consumer (``features.feature_for``,
    ``site.classify_cover``) reads only ``room.label``, so no other code
    needs to change.
    """
    output_dir = Path(output_dir)
    if result.scale is None:
        logger.warning("scale could not be determined; skipping 3D export")
        return result

    scene = build_scene(
        [floor.plan for floor in result.floors],
        wall_height_ft=wall_height_ft,
        scale=result.scale,
        page_size=result.page_size,
        palette=palette,
        site=site,
    )
    result.model_path = export_scene(scene, output_dir / "house.glb")
    return result


def run(
    source: Path,
    output_dir: Path,
    segmenter: Segmenter = classical_mask,
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    crop: bool = True,
    palette: Palette | None = None,
    site: Landscaping | None = None,
    split: int | None = None,
) -> PipelineResult:
    """Convert a floor plan into a stacked 3D model. See ``extract`` + ``build``.

    ``split`` overrides the sheet splitter: 1 keeps the sheet whole, N forces
    N plans, None reads it automatically.
    """
    result = extract(source, output_dir, segmenter, crop, split=split)
    return build(result, output_dir, wall_height_ft, palette, site)


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
