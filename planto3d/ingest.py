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

from planto3d.tools import poppler_bin_dir

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
# Splitting a sheet that carries several plans. A gutter is a band of
# near-empty page; both it and the pieces either side are measured as
# fractions of the sheet so the rule holds at any resolution.
# A gutter is nearly empty *for this sheet*, not empty in absolute terms.
# The band between two plans carries dimension text, road labels and the
# odd leader line -- on one sheet it measured 1.5 to 3% ink against 80%
# for a column through a drawing, and an absolute threshold of 0.4% called
# it occupied and refused to split. Fifteen percent of a typical drawn
# column separates the two by a wide margin on every sheet tried.
GUTTER_INK_RATIO = 0.15

# The floor, for a sheet so sparse that a fraction of its own ink means
# nothing -- a single plan on a large white page, where the median drawn
# column is already faint.
GUTTER_INK_FRACTION = 0.004

# Columns below this carry no drawing at all and are margin rather than
# gutter. Excluded when working out what a typical drawn column looks
# like, or a wide white border drags the reference down towards zero.
INKED_COLUMN_FLOOR = 0.01
# A gutter has to be wide relative to the sheet rather than a fixed number
# of pixels, so the rule holds at any resolution.
#
# Swept against CubiCasa's recorded floor counts, this is a plateau rather
# than a peak -- anything from 0.004 to 0.008 gives the same 55 of 60, and
# it falls away on both sides. The middle is taken for the margin, and the
# flatness is the evidence that it is not fitted to these sixty sheets.
#
#     0.020   51/60   precision 73%   recall 57%
#     0.012   53/60             77%             71%
#     0.006   55/60             85%             79%
#     0.002   54/60             83%             71%
#
# Loosening it this far is only safe because the pieces are checked
# afterwards: detection is generous and _pieces_look_like_plans throws out
# what does not stand up.
MIN_GUTTER_FRACTION = 0.006

# A gutter is frequently ruled down the middle, one plan's border box
# against the next. That single line of ink breaks the empty run in two,
# and each half then looks too narrow to be a gutter. Bands of ink up to
# this share of the sheet are bridged before runs are measured, which is
# wide enough for a border and far too narrow for a drawing.
GUTTER_RULE_FRACTION = 0.01
MIN_PIECE_FRACTION = 0.12
MIN_PIECE_INK = 0.01
MIN_SPLIT_WIDTH = 400

# Least share of a sheet's ink a piece must hold to be believed a plan of
# its own. Measured against CubiCasa's recorded floor counts: every false
# split left at least one piece below a fifth of the ink, several below a
# tenth, while genuine multi-plan sheets divide far more evenly -- two
# drawings of the same building are drawn at the same scale and carry
# comparable detail.
#
# This is deliberately a share of ink rather than of width. A wide band of
# blank page either side of a plan says nothing about whether it is a plan;
# the drawing in it does.
MIN_PLAN_INK_SHARE = 0.2

# Boundary lines needed before the fallback may split a sheet. One line is
# ambiguous; two, leaving three pieces, is a claim about the whole sheet.
# See ``split_sheet``.
MIN_BOUNDARY_CUTS = 2

# How much of a piece's empty area must be sealed off from its own border
# before that piece is accepted as a floor plan.
#
# This is the one thing a floor plan always does: walls enclose rooms. A
# legend, a key, a title block or a revision table encloses nothing --
# flood fill from its edge reaches almost every empty pixel in it. Measured
# over 25 pieces of correctly split sheets the share runs 0.21 to 0.96 with
# a median of 0.50; the legend on sheet 8150 scores 0.019. This sits an
# order of magnitude above the legend and half the worst real plan.
MIN_ENCLOSED_SHARE = 0.10

# How many points along each edge to start the flood from. A drawing whose
# outer wall touches the sheet edge would seal the fill out from a single
# corner, so it is started from many points along all four sides.
BORDER_PROBES = 200
# Detecting the page itself. A tone this light covering this much of a sheet
# is paper, not drawing -- no plan's linework covers a tenth of the page in
# one flat shade.
PAPER_MIN_VALUE = 150
PAPER_MIN_SHARE = 0.10
PAPER_TOLERANCE = 6
# Splitting by the rules that bound each plan, for sheets with no blank page
# between them. A rule runs nearly the full height, which nothing inside a
# drawing does; two standing close together mark where plans meet.
RULE_INK_FRACTION = 0.85
RULE_JOIN_GAP = 12
RULE_PAIR_FRACTION = 0.12
SHEET_MARGIN_FRACTION = 0.08

Box = tuple[int, int, int, int]


def rasterize_pdf(pdf_path: Path, output_dir: Path, dpi: int = WORKING_DPI) -> list[Path]:
    """Render each PDF page to a PNG. Returns the page image paths in order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_bin_dir())
    output_paths = []
    for i, page in enumerate(pages, start=1):
        out_path = output_dir / f"page-{i}.png"
        page.save(out_path)
        output_paths.append(out_path)
    return output_paths


def read_image(path: Path) -> np.ndarray | None:
    """Read an image, flattening any transparency onto white.

    Plans are frequently exported with a transparent background, and reading
    one with the alpha channel discarded leaves whatever happened to sit in
    the colour channels there -- often a checkerboard, which registers as ink
    across the entire sheet. On CubiCasa samples that put the minimum column
    ink at 8%, so nothing could ever look like empty page: gutters between
    plans vanished, and every measurement of blankness was wrong.

    Compositing onto white restores what the drawing actually looks like.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 3 and image.shape[2] == 4:
        colour = image[:, :, :3].astype(np.float32)
        alpha = (image[:, :, 3:4].astype(np.float32)) / 255.0
        flattened = colour * alpha + 255.0 * (1.0 - alpha)
        return flattened.astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def paper_tones(grey: np.ndarray) -> set[int]:
    """Light tones that make up the page rather than the drawing.

    A fixed threshold assumes paper is white. Plenty of exports have a
    transparency checkerboard flattened into the image, whose mid-greys sit
    below any sensible ink cutoff and are counted as drawing -- on CubiCasa
    samples that left no column below 8% ink, so no part of the sheet could
    ever look blank and gutters between plans were undetectable.

    Any light tone occupying a large share of the sheet is background: a
    drawing's lines never cover that much of a page.
    """
    counts = np.bincount(grey.ravel(), minlength=256)
    share = counts / max(grey.size, 1)
    return {
        value
        for value in range(PAPER_MIN_VALUE, 256)
        if share[value] >= PAPER_MIN_SHARE
    }


def _ink_mask(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink = grey < INK_THRESHOLD

    # Knock out whatever light tone dominates the sheet, with a little
    # latitude either side for the softening a resize introduces.
    for tone in paper_tones(grey):
        ink &= ~(np.abs(grey.astype(np.int16) - tone) <= PAPER_TOLERANCE)

    return ink.astype(np.uint8)


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


def _boundary_cuts(ink: np.ndarray) -> list[int]:
    """Where a sheet divides, found from the rules that bound each plan.

    Looking for empty gutters fails whenever plans are drawn inside boundary
    boxes: the box edges and the dimension lines running beside them mean
    something crosses every column, and no part of the sheet is ever blank.

    The boxes themselves are the better signal. A rule bounding a plan runs
    almost the full height of the sheet, which nothing inside a drawing does.
    Where two plans meet, two such rules stand close together -- the right
    edge of one box and the left edge of the next -- and the division lies
    between them.
    """
    height, width = ink.shape
    column_ink = ink.sum(axis=0) / height

    # Runs of columns that are nearly solid ink from top to bottom.
    rules: list[tuple[int, int]] = []
    start = previous = None
    for column in np.where(column_ink >= RULE_INK_FRACTION)[0]:
        if start is None:
            start = column
        elif column - previous > RULE_JOIN_GAP:
            rules.append((start, previous))
            start = column
        previous = column
    if start is not None:
        rules.append((start, previous))

    margin = width * SHEET_MARGIN_FRACTION
    interior = [(a, b) for a, b in rules if a > margin and b < width - margin]
    if not interior:
        return []

    # Group rules that stand close together; each group is one division.
    groups: list[list[tuple[int, int]]] = [[interior[0]]]
    for rule in interior[1:]:
        if rule[0] - groups[-1][-1][1] <= width * RULE_PAIR_FRACTION:
            groups[-1].append(rule)
        else:
            groups.append([rule])

    # A group of one is a single rule, and a single rule running the height
    # of a sheet is a long wall as often as it is a plan's border -- it is
    # what this produced every one of its false splits from. What the
    # docstring describes, and what actually marks a division, is a *pair*:
    # the right edge of one box against the left edge of the next, with the
    # division in the gap between them.
    return [
        (group[0][0] + group[-1][1]) // 2 for group in groups if len(group) >= 2
    ]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end of each run of True in a boolean mask."""
    found, start = [], None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            found.append((start, index))
            start = None
    if start is not None:
        found.append((start, len(mask)))
    return found


def _bridge(mask: np.ndarray, span: int) -> np.ndarray:
    """Fill short False gaps, so a ruled line does not break an empty run.

    Interior gaps only: bridging one that reaches an end would swallow the
    sheet's own margin into the first or last gutter.
    """
    bridged = mask.copy()
    for start, end in _runs(~mask):
        if end - start <= span and start > 0 and end < len(mask):
            bridged[start:end] = True
    return bridged


def _gutter_threshold(profile: np.ndarray) -> float:
    """How little ink a column may carry and still count as a gutter.

    Measured against the sheet rather than fixed, because "empty" is
    relative to how heavily the drawing is inked. A band between two plans
    is never truly blank -- it carries dimension text, road labels and
    leader lines -- and judging it against zero declared it occupied.
    """
    drawn = profile[profile > INKED_COLUMN_FLOOR]
    if drawn.size == 0:
        return GUTTER_INK_FRACTION
    return max(float(np.median(drawn)) * GUTTER_INK_RATIO, GUTTER_INK_FRACTION)


def _gutter_cuts(ink: np.ndarray, axis: int) -> list[int]:
    """Where to cut, from bands of near-empty page running across the sheet.

    ``axis`` 0 looks for vertical gutters between plans laid side by side,
    1 for horizontal ones between plans stacked top to bottom. Sheets do
    both, and looking only for one missed every stacked sheet outright.
    """
    length = ink.shape[1] if axis == 0 else ink.shape[0]
    across = ink.shape[0] if axis == 0 else ink.shape[1]

    profile = ink.sum(axis=axis) / across
    empty = _bridge(
        profile < _gutter_threshold(profile),
        max(int(length * GUTTER_RULE_FRACTION), 3),
    )

    minimum_gutter = length * MIN_GUTTER_FRACTION
    minimum_piece = length * MIN_PIECE_FRACTION
    cuts = [
        (start + end) // 2
        for start, end in _runs(empty)
        if end - start >= minimum_gutter
        and start > minimum_piece
        and end < length - minimum_piece
    ]
    return _merge_close_cuts(cuts, minimum_piece)


def _merge_close_cuts(cuts: list[int], minimum_piece: float) -> list[int]:
    """Collapse cuts too close together to have a plan between them.

    A gutter is rarely one clean band. The gap between two plans carries a
    plot boundary, a dimension string, a north point -- enough ink at one
    point to break the quiet run in two and report two cuts a few dozen
    pixels apart. Split on both and the sheet comes back as three pieces,
    the middle one a sliver, which is then rejected as not looking like
    plans -- so a sheet with a perfectly clear gutter does not get split at
    all, and two houses are reconstructed as one.

    Two cuts closer together than the narrowest allowable plan are one
    gutter seen twice, so they collapse to their midpoint.
    """
    merged: list[int] = []
    for cut in sorted(cuts):
        if merged and cut - merged[-1] < minimum_piece:
            merged[-1] = (merged[-1] + cut) // 2
        else:
            merged.append(cut)
    return merged


def _pieces_look_like_plans(
    ink: np.ndarray, divisions: list[int], axis: int = 0
) -> bool:
    """Whether cutting here leaves every piece substantial enough to be a plan.

    A sheet holding two floors of one building divides fairly evenly: both
    are drawn at the same scale and carry comparable detail. A cut through
    the middle of a single plan does not -- it leaves a sliver on one side,
    and measured across sixty sheets every false split left at least one
    piece below a fifth of the ink.

    Checked on ink rather than on width, because a wide margin of blank page
    says nothing about whether there is a drawing in it.
    """
    total = float(ink.sum())
    if total <= 0:
        return False

    edges = [0, *divisions, ink.shape[1] if axis == 0 else ink.shape[0]]
    shares = [
        float((ink[:, a:b] if axis == 0 else ink[a:b, :]).sum()) / total
        for a, b in zip(edges, edges[1:])
    ]
    thinnest = min(shares)
    if thinnest < MIN_PLAN_INK_SHARE:
        logger.info(
            "rejecting split into %d: thinnest piece holds %.0f%% of the ink",
            len(shares),
            thinnest * 100,
        )
        return False

    # Carrying ink is not the same as being a plan. A legend, a key, a
    # title block or a revision table sits behind a gutter just as a second
    # plan does and carries plenty of ink -- sheet 8150 is a sketch with a
    # key beneath it, and the key was built as a second storey. What tells
    # them apart is that a plan encloses space and a list of symbols does
    # not.
    for index, (a, b) in enumerate(zip(edges, edges[1:])):
        piece = ink[:, a:b] if axis == 0 else ink[a:b, :]
        enclosed = _enclosed_share(piece)
        if enclosed < MIN_ENCLOSED_SHARE:
            logger.info(
                "rejecting split into %d: piece %d encloses %.1f%% of itself, "
                "so it is a legend or a title block rather than a plan",
                len(shares),
                index,
                enclosed * 100,
            )
            return False
    return True


def _enclosed_share(ink: np.ndarray) -> float:
    """Share of a piece's empty area that its own border cannot reach.

    Flood fill inward from every edge; whatever the fill cannot reach is
    sealed off by ink, which on a floor plan means it is a room. Returns 0
    for a piece with no empty space at all.
    """
    height, width = ink.shape
    if height < 3 or width < 3:
        return 0.0

    free = (~ink.astype(bool)).astype(np.uint8)
    total = float(free.sum())
    if total <= 0:
        return 0.0

    # cv2.floodFill needs a mask two pixels larger than the image.
    reachable = free.copy()
    scratch = np.zeros((height + 2, width + 2), np.uint8)
    for x in range(0, width, max(width // BORDER_PROBES, 1)):
        for y in (0, height - 1):
            if reachable[y, x]:
                cv2.floodFill(reachable, scratch, (x, y), 0)
    for y in range(0, height, max(height // BORDER_PROBES, 1)):
        for x in (0, width - 1):
            if reachable[y, x]:
                cv2.floodFill(reachable, scratch, (x, y), 0)

    return float(reachable.sum()) / total


def split_sheet(image: np.ndarray) -> list[np.ndarray]:
    """Split a sheet carrying several floor plans into one image per plan.

    Sheets frequently lay basement, ground and first floor in a row, and
    reading such a sheet as a single storey reconstructs three buildings as
    one flat floor -- confidently, with plausible numbers, which is worse
    than failing.

    Plans are separated by gutters: bands of near-empty page far wider than
    the gaps inside a drawing. A gutter must be wide relative to the sheet
    rather than a fixed number of pixels, so the rule holds at any
    resolution, and the pieces either side must each be substantial enough
    to be a plan rather than a stray annotation.

    Returns the original image unchanged when no such split is found, which
    is the common case for a proper drawing set.
    """
    ink = _ink_mask(image)
    height, width = ink.shape
    if width < MIN_SPLIT_WIDTH:
        return [image]

    # Side by side first, then stacked. Sheets are laid out both ways and
    # looking only for vertical gutters missed every stacked sheet outright.
    # Columns are tried first because it is much the commoner arrangement,
    # and a sheet split correctly one way should not then be cut the other.
    for axis in (0, 1):
        length = width if axis == 0 else height
        minimum_piece = length * MIN_PIECE_FRACTION

        # Blank page between plans is unambiguous, so gutters are believed
        # before anything else. Where a sheet draws each plan inside a
        # boundary box there is no blank page to find, so the boxes
        # themselves are the fallback -- but only across columns, which is
        # the only direction that fallback was ever built for.
        divisions = _gutter_cuts(ink, axis)
        if not divisions and axis == 0:
            divisions = _boundary_cuts(ink)
            # A lone boundary line is not evidence of anything. A plot
            # border, a dimension band and a strong internal wall all look
            # exactly like the edge between two plans, and the rule cannot
            # tell them apart -- it fires on three sheets and is right on
            # one. Both sheets it gets wrong are cut by a single line
            # through the middle of one apartment; the sheet it gets right
            # carries two lines and three plans.
            #
            # Two lines leaving three comparable pieces is a much stronger
            # claim than one line leaving two, and the gutter rules already
            # cover the ordinary two-plan sheet perfectly -- ten firings,
            # ten right. So the fallback is left to do only the thing it
            # can actually do.
            if len(divisions) < MIN_BOUNDARY_CUTS:
                divisions = []

        divisions = sorted(
            c for c in divisions if minimum_piece < c < length - minimum_piece
        )
        if not divisions:
            continue

        # Whatever found them, the pieces have to stand up as plans. The
        # boundary-rule fallback in particular fires readily on dimension
        # lines and plot borders, which run the full height of a drawing
        # just as a real separator does; unchecked it split eleven single
        # plans across sixty sheets and got one of them right.
        if not _pieces_look_like_plans(ink, divisions, axis):
            continue

        cuts = [0, *divisions, length]
        pieces = [
            image[:, a:b] if axis == 0 else image[a:b, :]
            for a, b in zip(cuts, cuts[1:])
            if b - a >= minimum_piece
        ]

        # Every piece must actually carry a drawing; a blank one means the
        # split found a margin rather than a gutter.
        pieces = [p for p in pieces if _ink_mask(p).mean() > MIN_PIECE_INK]
        if len(pieces) < 2:
            continue

        logger.info(
            "sheet holds %d plans %s; splitting",
            len(pieces),
            "side by side" if axis == 0 else "stacked",
        )
        return pieces

    return [image]


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
