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

from planto3d.geometry_types import Wall

logger = logging.getLogger(__name__)

FEET_TO_METRES = 0.3048
# Storey height used when the drawing does not state one.
DEFAULT_WALL_HEIGHT_FT = 9.0
# Walls shorter than this in pixels are extraction noise, not geometry.
MIN_WALL_PIXELS = 1e-6


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
    floors: list[list[Wall]],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
) -> trimesh.Trimesh:
    """Extrude several floors and stack them, lowest first."""
    if not floors:
        raise ValueError("no floors to extrude")

    meshes = []
    for index, walls in enumerate(floors):
        if not walls:
            logger.warning("floor %d has no walls; skipping", index)
            continue
        meshes.append(
            walls_to_mesh(
                walls,
                wall_height_ft=wall_height_ft,
                scale=scale,
                base_ft=index * wall_height_ft,
            )
        )

    if not meshes:
        raise ValueError("no floor produced any geometry")

    logger.info("stacked %d floor(s)", len(meshes))
    return trimesh.util.concatenate(meshes)


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> Path:
    """Write a mesh as binary glTF, the format web viewers expect."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    logger.info("wrote %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
