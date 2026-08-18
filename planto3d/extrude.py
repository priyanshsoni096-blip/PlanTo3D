"""Turn wall geometry into a 3D mesh.

Plan coordinates are pixels on a page: X across, Y down. glTF is Y-up, so
the page's Y becomes the model's Z (depth) and Y becomes height. Everything
converts to metres on the way out, since glTF viewers assume metres.

Floors stack in the order given. They share a horizontal frame because the
sheets were cropped to a common box upstream, so no per-floor alignment is
applied here.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon
from trimesh.creation import extrude_polygon

from planto3d.geometry_types import FloorPlan, Opening, Wall
from planto3d.features import GROUND_COVERS, group_by_feature
from planto3d.site import (
    BOUNDARY_HEIGHT_FT,
    BOUNDARY_THICKNESS_FT,
    COVER_THICKNESS_FT,
    POOL_DEPTH_FT,
    RAILING_HEIGHT_FT,
    RAILING_THICKNESS_FT,
    SITE_MARGIN_FT,
    SITE_THICKNESS_FT,
    boundary_walls,
    site_outline,
)

logger = logging.getLogger(__name__)

FEET_TO_METRES = 0.3048
# Storey height used when the drawing does not state one.
DEFAULT_WALL_HEIGHT_FT = 9.0
# Walls shorter than this in pixels are extraction noise, not geometry.
MIN_WALL_PIXELS = 1e-6
# Structural slab thickness, and the parapet standing above the roof.
SLAB_THICKNESS_FT = 0.5
PARAPET_HEIGHT_FT = 3.0
PARAPET_THICKNESS_FT = 0.6
# A footprint needs this many vertices to enclose an area.
MIN_FOOTPRINT_VERTICES = 3
# Door and window heads sit at a common height; windows also get a sill.
OPENING_HEAD_FT = 7.0
SILL_HEIGHT_FT = 3.0
# Slivers below these sizes are dropped rather than modelled -- a millimetre
# of wall between two openings is noise, not architecture.
MIN_OPENING_WIDTH_M = 0.15
MIN_PIER_M = 0.05
# Glazing is thinner than the wall it sits in, so it reads as a pane rather
# than a plug and does not z-fight with the reveal.
GLASS_THICKNESS_RATIO = 0.25


def _place(
    extents: tuple[float, float, float],
    wall: Wall,
    along_m: float,
    centre_height_m: float,
    scale: float,
) -> trimesh.Trimesh:
    """Build a box and place it along a wall at a given distance and height."""
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    direction = end - start
    length_m = float(np.linalg.norm(direction))
    unit = direction / length_m

    box = trimesh.creation.box(extents=list(extents))

    # Rotate about the vertical axis to align with the wall's direction. The
    # page's downward Y maps to +Z, so the angle is measured in the XZ plane.
    angle = -np.arctan2(direction[1], direction[0])
    box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))

    centre = start + unit * along_m
    box.apply_transform(
        trimesh.transformations.translation_matrix([centre[0], centre_height_m, centre[1]])
    )
    return box


def _wall_parts(
    wall: Wall,
    openings: list[Opening],
    height_m: float,
    scale: float,
    base_m: float,
) -> list[trimesh.Trimesh]:
    """A wall as solid pieces, with gaps left where openings interrupt it.

    Built by splitting rather than boolean subtraction: the pier either side
    of an opening, plus a lintel over it and a sill under a window. CSG on
    this many boxes is slow and fragile, and splitting yields clean geometry
    with no dependency on a boolean backend.
    """
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    thickness_m = max(wall.thickness / scale * FEET_TO_METRES, 1e-4)

    length_m = float(np.linalg.norm(end - start))
    if length_m <= MIN_WALL_PIXELS:
        return []

    if not openings:
        return [
            _place(
                (length_m, height_m, thickness_m),
                wall,
                length_m / 2,
                base_m + height_m / 2,
                scale,
            )
        ]

    # Openings in order along the wall, clipped to it and non-overlapping.
    spans: list[tuple[float, float, str]] = []
    for opening in sorted(openings, key=lambda o: o.position):
        half = (opening.width / scale * FEET_TO_METRES) / 2
        centre = opening.position / scale * FEET_TO_METRES
        low, high = max(0.0, centre - half), min(length_m, centre + half)
        if high - low < MIN_OPENING_WIDTH_M:
            continue
        if spans and low < spans[-1][1]:
            low = spans[-1][1]
            if high - low < MIN_OPENING_WIDTH_M:
                continue
        spans.append((low, high, opening.type))

    if not spans:
        return [
            _place(
                (length_m, height_m, thickness_m),
                wall,
                length_m / 2,
                base_m + height_m / 2,
                scale,
            )
        ]

    parts: list[trimesh.Trimesh] = []
    cursor = 0.0

    for low, high, opening_type in spans:
        if low - cursor > MIN_PIER_M:
            pier = low - cursor
            parts.append(
                _place(
                    (pier, height_m, thickness_m),
                    wall,
                    cursor + pier / 2,
                    base_m + height_m / 2,
                    scale,
                )
            )

        span = high - low
        head_m = min(OPENING_HEAD_FT * FEET_TO_METRES, height_m)
        sill_m = SILL_HEIGHT_FT * FEET_TO_METRES if opening_type == "window" else 0.0

        if sill_m > 0:
            parts.append(
                _place(
                    (span, sill_m, thickness_m), wall, low + span / 2, base_m + sill_m / 2, scale
                )
            )

        lintel = height_m - head_m
        if lintel > MIN_PIER_M:
            parts.append(
                _place(
                    (span, lintel, thickness_m),
                    wall,
                    low + span / 2,
                    base_m + head_m + lintel / 2,
                    scale,
                )
            )

        cursor = high

    if length_m - cursor > MIN_PIER_M:
        remainder = length_m - cursor
        parts.append(
            _place(
                (remainder, height_m, thickness_m),
                wall,
                cursor + remainder / 2,
                base_m + height_m / 2,
                scale,
            )
        )

    return parts


def opening_panes(
    wall: Wall,
    openings: list[Opening],
    height_m: float,
    scale: float,
    base_m: float,
) -> list[trimesh.Trimesh]:
    """Thin panes filling each window opening, to be given a glass material.

    Doors are left empty: an open doorway reads correctly, whereas a glazed
    one does not.
    """
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    length_m = float(np.linalg.norm(end - start))
    if length_m <= MIN_WALL_PIXELS:
        return []

    thickness_m = max(wall.thickness / scale * FEET_TO_METRES, 1e-4)
    head_m = min(OPENING_HEAD_FT * FEET_TO_METRES, height_m)
    sill_m = SILL_HEIGHT_FT * FEET_TO_METRES

    panes = []
    for opening in openings:
        if opening.type != "window":
            continue

        half = (opening.width / scale * FEET_TO_METRES) / 2
        centre = opening.position / scale * FEET_TO_METRES
        low, high = max(0.0, centre - half), min(length_m, centre + half)
        span = high - low
        if span < MIN_OPENING_WIDTH_M or head_m <= sill_m:
            continue

        pane_height = head_m - sill_m
        panes.append(
            _place(
                (span, pane_height, thickness_m * GLASS_THICKNESS_RATIO),
                wall,
                low + span / 2,
                base_m + sill_m + pane_height / 2,
                scale,
            )
        )

    return panes


def slab_mesh(
    footprint: list[tuple[float, float]],
    thickness_ft: float,
    base_ft: float,
    scale: float,
    holes: list[list[tuple[float, float]]] | None = None,
) -> trimesh.Trimesh | None:
    """A horizontal slab covering a storey's footprint.

    ``holes`` are regions to cut out -- an open terrace garden, for instance,
    which must not be roofed over.

    Returns None when the outline cannot enclose an area, so a bad contour
    costs a slab rather than the whole model.
    """
    if len(footprint) < MIN_FOOTPRINT_VERTICES:
        return None

    def to_metres(points):
        return [(x / scale * FEET_TO_METRES, y / scale * FEET_TO_METRES) for x, y in points]

    try:
        polygon = Polygon(to_metres(footprint))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # repair self-intersections

        for hole in holes or []:
            if len(hole) < MIN_FOOTPRINT_VERTICES:
                continue
            cut = Polygon(to_metres(hole))
            if not cut.is_valid:
                cut = cut.buffer(0)
            if cut.is_valid and not cut.is_empty:
                polygon = polygon.difference(cut)

        # A difference can split the slab into several pieces; keep the
        # largest, since a roof reduced to slivers is worse than none.
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda part: part.area)

        if polygon.is_empty or polygon.area <= 0:
            return None
        slab = extrude_polygon(polygon, height=thickness_ft * FEET_TO_METRES)
    except Exception as error:
        logger.warning("could not build slab from footprint: %s", error)
        return None

    # extrude_polygon builds the outline in XY and extrudes along +Z. Rotating
    # +90 degrees about X sends the page's downward Y to +Z, matching how
    # walls are placed -- the opposite rotation mirrors the slab and doubles
    # the model's depth. That rotation leaves the slab hanging below the
    # origin, so it is lifted by its own thickness as well as the storey base.
    thickness_m = thickness_ft * FEET_TO_METRES
    slab.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    slab.apply_transform(
        trimesh.transformations.translation_matrix(
            [0, base_ft * FEET_TO_METRES + thickness_m, 0]
        )
    )
    return slab


def _railing_parts(
    polygon: list[tuple[float, float]], base_m: float, scale: float
) -> list[trimesh.Trimesh]:
    """A waist-high rail running around an open-edged room."""
    if len(polygon) < MIN_FOOTPRINT_VERTICES:
        return []

    thickness_px = RAILING_THICKNESS_FT * scale
    height_m = RAILING_HEIGHT_FT * FEET_TO_METRES

    parts = []
    for index in range(len(polygon)):
        rail = Wall(
            start=polygon[index],
            end=polygon[(index + 1) % len(polygon)],
            thickness=thickness_px,
        )
        parts.extend(_wall_parts(rail, [], height_m, scale, base_m))
    return parts


def _parapet_walls(footprint: list[tuple[float, float]], base_ft: float, scale: float) -> list[Wall]:
    """The low wall running around a flat roof's edge."""
    if len(footprint) < MIN_FOOTPRINT_VERTICES:
        return []

    thickness_px = PARAPET_THICKNESS_FT * scale
    return [
        Wall(start=footprint[i], end=footprint[(i + 1) % len(footprint)], thickness=thickness_px)
        for i in range(len(footprint))
    ]


def walls_to_mesh(
    walls: list[Wall],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
    base_ft: float = 0.0,
    openings: list[Opening] | None = None,
) -> trimesh.Trimesh:
    """Extrude one floor's walls into a mesh, leaving gaps for openings.

    ``scale`` is pixels per foot, as measured by the calibration stage.
    ``base_ft`` lifts the floor, for stacking storeys.
    """
    if not walls:
        raise ValueError("no walls to extrude")

    height_m = wall_height_ft * FEET_TO_METRES
    base_m = base_ft * FEET_TO_METRES

    by_wall: dict[int, list[Opening]] = {}
    for opening in openings or []:
        if 0 <= opening.wall_id < len(walls):
            by_wall.setdefault(opening.wall_id, []).append(opening)

    parts: list[trimesh.Trimesh] = []
    for wall_id, wall in enumerate(walls):
        parts.extend(
            _wall_parts(wall, by_wall.get(wall_id, []), height_m, scale, base_m)
        )

    if not parts:
        raise ValueError("every wall was degenerate; nothing to extrude")

    mesh = trimesh.util.concatenate(parts)
    logger.info(
        "extruded %d wall(s) with %d opening(s) into %d faces",
        len(walls),
        sum(len(v) for v in by_wall.values()),
        len(mesh.faces),
    )
    return mesh


def floors_to_mesh(
    floors: list[FloorPlan],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
) -> trimesh.Trimesh:
    """Extrude several floors and stack them, lowest first.

    Each storey gets a floor slab under its walls, and the topmost gains a
    roof slab with a parapet around the edge. Walls alone leave a building
    open top and bottom, which reads as floating fragments rather than a
    house.
    """
    if not floors:
        raise ValueError("no floors to extrude")

    meshes: list[trimesh.Trimesh] = []

    for index, floor in enumerate(floors):
        base_ft = index * wall_height_ft

        slab = slab_mesh(floor.footprint, SLAB_THICKNESS_FT, base_ft, scale)
        if slab is not None:
            meshes.append(slab)
        elif floor.footprint:
            logger.warning("floor %d footprint gave no slab", index)

        if floor.walls:
            meshes.append(
                walls_to_mesh(
                    floor.walls,
                    wall_height_ft=wall_height_ft,
                    scale=scale,
                    base_ft=base_ft + SLAB_THICKNESS_FT,
                    openings=floor.openings,
                )
            )
        else:
            logger.warning("floor %d has no walls", index)

    # Cap the building: a roof slab over the top storey, with a parapet.
    roof_base_ft = len(floors) * wall_height_ft
    top = floors[-1]
    roof = slab_mesh(top.footprint, SLAB_THICKNESS_FT, roof_base_ft, scale)
    if roof is not None:
        meshes.append(roof)
        parapet = _parapet_walls(top.footprint, roof_base_ft, scale)
        if parapet:
            meshes.append(
                walls_to_mesh(
                    parapet,
                    wall_height_ft=PARAPET_HEIGHT_FT,
                    scale=scale,
                    base_ft=roof_base_ft + SLAB_THICKNESS_FT,
                )
            )

    if not meshes:
        raise ValueError("no floor produced any geometry")

    logger.info("stacked %d floor(s) into %d part(s)", len(floors), len(meshes))
    return trimesh.util.concatenate(meshes)


def floors_to_parts(
    floors: list[FloorPlan],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
    page_size: tuple[int, int] | None = None,
) -> dict[str, list[trimesh.Trimesh]]:
    """Build the building's geometry grouped by what it is made of.

    Keys name a material rather than a floor, so the scene builder can give
    walls, slabs, the roof and glazing distinct appearances.
    """
    if not floors:
        raise ValueError("no floors to extrude")

    height_m = wall_height_ft * FEET_TO_METRES
    parts: dict[str, list[trimesh.Trimesh]] = {
        "wall": [],
        "floor": [],
        "roof": [],
        "glass": [],
    }

    for index, floor in enumerate(floors):
        base_ft = index * wall_height_ft
        wall_base_m = (base_ft + SLAB_THICKNESS_FT) * FEET_TO_METRES

        features = group_by_feature(floor.rooms)

        # A double-height space, atrium or shaft is a hole through this
        # storey's floor. Slabbed over, the drawing's most dramatic space
        # becomes an ordinary ceiling.
        voids = [room.polygon for room in features.get("void", [])]
        slab = slab_mesh(
            floor.footprint, SLAB_THICKNESS_FT, base_ft, scale, holes=voids
        )
        if slab is not None:
            parts["floor"].append(slab)

        by_wall: dict[int, list[Opening]] = {}
        for opening in floor.openings:
            if 0 <= opening.wall_id < len(floor.walls):
                by_wall.setdefault(opening.wall_id, []).append(opening)

        for wall_id, wall in enumerate(floor.walls):
            openings = by_wall.get(wall_id, [])
            parts["wall"].extend(
                _wall_parts(wall, openings, height_m, scale, wall_base_m)
            )
            parts["glass"].extend(
                opening_panes(wall, openings, height_m, scale, wall_base_m)
            )

        # Balconies and terraces are open to the air. Without a railing an
        # upper floor reads as a hole punched in the facade.
        for room in features.get("open", []):
            parts.setdefault("railing", []).extend(
                _railing_parts(room.polygon, wall_base_m, scale)
            )

        # A void's edge is a drop, so it needs a railing too.
        for room in features.get("void", []):
            parts.setdefault("railing", []).extend(
                _railing_parts(room.polygon, wall_base_m, scale)
            )

        # Ground cover for this storey: lawn, paving, and water recessed into
        # the surface. A pool laid flat on the ground reads as a blue carpet.
        surface_ft = base_ft + SLAB_THICKNESS_FT if index else 0.0
        for category in GROUND_COVERS:
            for room in features.get(category, []):
                if category == "water":
                    patch = slab_mesh(
                        room.polygon,
                        POOL_DEPTH_FT,
                        surface_ft - POOL_DEPTH_FT,
                        scale,
                    )
                else:
                    patch = slab_mesh(
                        room.polygon, COVER_THICKNESS_FT, surface_ft, scale
                    )
                if patch is not None:
                    parts.setdefault(category, []).append(patch)

    roof_base_ft = len(floors) * wall_height_ft
    top = floors[-1]

    # A planted terrace is open to the sky. Roofing over it hides the garden
    # entirely and makes the top storey read as a sealed box.
    roof = slab_mesh(
        top.footprint, SLAB_THICKNESS_FT, roof_base_ft, scale, holes=top.planting
    )
    if roof is not None:
        parts["roof"].append(roof)
        parapet = _parapet_walls(top.footprint, roof_base_ft, scale)
        for wall in parapet:
            parts["roof"].extend(
                _wall_parts(
                    wall,
                    [],
                    PARAPET_HEIGHT_FT * FEET_TO_METRES,
                    scale,
                    (roof_base_ft + SLAB_THICKNESS_FT) * FEET_TO_METRES,
                )
            )

    # Merged rather than assigned: the site block also produces lawn and
    # paving, and replacing the keys would discard whatever the storeys
    # contributed under the same names.
    for name, meshes in _site_parts(floors, scale, wall_height_ft, page_size).items():
        parts.setdefault(name, []).extend(meshes)

    return {name: meshes for name, meshes in parts.items() if meshes}


def _site_parts(
    floors: list[FloorPlan],
    scale: float,
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    page_size: tuple[int, int] | None = None,
) -> dict[str, list[trimesh.Trimesh]]:
    """The ground the building stands on, with lawn, paving and a boundary.

    Sits just below zero so the building's own ground-floor slab reads as
    resting on it rather than z-fighting with it.
    """
    outline = site_outline(
        [floor.footprint for floor in floors if floor.footprint],
        margin_px=SITE_MARGIN_FT * scale,
        page_size=page_size,
    )
    if not outline:
        return {}

    parts: dict[str, list[trimesh.Trimesh]] = {
        "ground": [],
        "lawn": [],
        "paving": [],
        "boundary": [],
    }

    ground = slab_mesh(outline, SITE_THICKNESS_FT, -SITE_THICKNESS_FT, scale)
    if ground is not None:
        parts["ground"].append(ground)

    # A compound wall around the plot, as the reference elevations show.
    for wall in boundary_walls(outline, BOUNDARY_THICKNESS_FT * scale):
        parts["boundary"].extend(
            _wall_parts(
                wall, [], BOUNDARY_HEIGHT_FT * FEET_TO_METRES, scale, 0.0
            )
        )

    # Planting is found by colour rather than by label. The segmentation
    # model is trained on interiors and does not mark a garden at all, since
    # a garden is not a room -- so without this the lawns never appear.
    #
    # Laid on top of the storey's floor slab, not at its base. Placed at the
    # base it sits inside the slab -- a terrace garden simply vanished into
    # the floor it was supposed to be growing on.
    for floor_index, floor in enumerate(floors):
        base_ft = floor_index * wall_height_ft
        surface_ft = base_ft + SLAB_THICKNESS_FT if floor_index else 0.0
        for region in floor.planting:
            patch = slab_mesh(region, COVER_THICKNESS_FT, surface_ft, scale)
            if patch is not None:
                parts["lawn"].append(patch)

    return parts


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> Path:
    """Write a mesh as binary glTF, the format web viewers expect."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    logger.info("wrote %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
