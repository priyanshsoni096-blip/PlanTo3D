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

from planto3d.geometry_types import FloorPlan, Wall

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


def _wall_box(wall: Wall, height_m: float, scale: float, base_m: float) -> trimesh.Trimesh | None:
    """One wall as a box, positioned in the model's coordinate frame."""
    start = np.array(wall.start, dtype=float) / scale * FEET_TO_METRES
    end = np.array(wall.end, dtype=float) / scale * FEET_TO_METRES
    thickness_m = max(wall.thickness / scale * FEET_TO_METRES, 1e-4)

    direction = end - start
    length_m = float(np.linalg.norm(direction))
    if length_m <= MIN_WALL_PIXELS:
        return None

    box = trimesh.creation.box(extents=[length_m, height_m, thickness_m])

    # Rotate about the vertical axis to align with the wall's direction. The
    # page's downward Y maps to +Z, so the angle is measured in the XZ plane.
    angle = -np.arctan2(direction[1], direction[0])
    box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))

    midpoint = (start + end) / 2
    box.apply_transform(
        trimesh.transformations.translation_matrix(
            [midpoint[0], base_m + height_m / 2, midpoint[1]]
        )
    )
    return box


def slab_mesh(
    footprint: list[tuple[float, float]],
    thickness_ft: float,
    base_ft: float,
    scale: float,
) -> trimesh.Trimesh | None:
    """A horizontal slab covering a storey's footprint.

    Returns None when the outline cannot enclose an area, so a bad contour
    costs a slab rather than the whole model.
    """
    if len(footprint) < MIN_FOOTPRINT_VERTICES:
        return None

    points = [(x / scale * FEET_TO_METRES, y / scale * FEET_TO_METRES) for x, y in footprint]

    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # repair self-intersections
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
) -> trimesh.Trimesh:
    """Extrude one floor's walls into a mesh.

    ``scale`` is pixels per foot, as measured by the calibration stage.
    ``base_ft`` lifts the floor, for stacking storeys.
    """
    if not walls:
        raise ValueError("no walls to extrude")

    height_m = wall_height_ft * FEET_TO_METRES
    base_m = base_ft * FEET_TO_METRES

    boxes = [_wall_box(wall, height_m, scale, base_m) for wall in walls]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        raise ValueError("every wall was degenerate; nothing to extrude")

    mesh = trimesh.util.concatenate(boxes)
    logger.info("extruded %d wall(s) into %d faces", len(boxes), len(mesh.faces))
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


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> Path:
    """Write a mesh as binary glTF, the format web viewers expect."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    logger.info("wrote %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
