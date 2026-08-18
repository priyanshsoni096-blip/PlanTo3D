"""Render a mesh to a PNG without a GPU or display.

Useful for checking the model in a terminal, in CI, or anywhere an OpenGL
context is unavailable. Faces are painted back to front with a light-angle
shade, which is enough to read the building's shape.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# Viewing angles for a three-quarter aerial, matching how plans are presented.
DEFAULT_AZIMUTH = 40.0
DEFAULT_ELEVATION = 30.0
LIGHT_DIRECTION = np.array([0.4, 0.85, 0.35])
BASE_COLOUR = np.array([214, 208, 198], dtype=float)
BACKGROUND = (18, 18, 22)


def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    yaw = np.array(
        [
            [np.cos(azimuth), 0, np.sin(azimuth)],
            [0, 1, 0],
            [-np.sin(azimuth), 0, np.cos(azimuth)],
        ]
    )
    pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(elevation), -np.sin(elevation)],
            [0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    return pitch @ yaw


def render(
    mesh: trimesh.Trimesh,
    output_path: Path,
    resolution: tuple[int, int] = (1200, 900),
    azimuth: float = DEFAULT_AZIMUTH,
    elevation: float = DEFAULT_ELEVATION,
    face_colours: np.ndarray | None = None,
) -> Path:
    """Paint a shaded view of ``mesh`` to a PNG.

    ``face_colours`` gives a per-face base colour; without it the whole mesh
    takes one stone tone.
    """
    from PIL import Image, ImageDraw

    width, height = resolution
    rotation = _rotation(azimuth, elevation)

    vertices = (rotation @ mesh.vertices.T).T
    normals = (rotation @ mesh.face_normals.T).T

    # Fit the model to the canvas with a margin.
    lo, hi = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = 0.88 * min(width / span[0], height / span[1])
    offset = np.array([width, height]) / 2 - (lo + hi) / 2 * scale

    projected = vertices[:, :2] * scale + offset
    projected[:, 1] = height - projected[:, 1]  # screen Y grows downward

    shade = np.clip(normals @ (LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)), 0, 1)
    brightness = 0.35 + 0.65 * shade

    image = Image.new("RGB", resolution, BACKGROUND)
    draw = ImageDraw.Draw(image)

    base = BASE_COLOUR if face_colours is None else face_colours

    # Painter's algorithm: furthest faces first.
    depth = vertices[mesh.faces][:, :, 2].mean(axis=1)
    for index in np.argsort(depth):
        tone = base if face_colours is None else base[index]
        colour = tuple(int(c) for c in np.clip(tone * brightness[index], 0, 255))
        polygon = [tuple(projected[v]) for v in mesh.faces[index]]
        draw.polygon(polygon, fill=colour, outline=colour)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info("wrote preview %s", output_path)
    return output_path


def _material_colour(geometry) -> np.ndarray | None:
    """Base colour of a geometry's material, as 0-255 RGB.

    glTF defines baseColorFactor in 0-1, but trimesh hands it back as 0-255
    bytes once a file has been reloaded. Both are accepted here -- scaling the
    byte form again saturates every surface to white.
    """
    material = getattr(geometry.visual, "material", None)
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        return None

    channels = np.array([float(c) for c in factor[:3]])
    return channels * 255 if channels.max() <= 1.0 else channels


def render_glb(model_path: Path, output_path: Path, **kwargs) -> Path:
    """Load a .glb and render it, honouring per-material colours."""
    loaded = trimesh.load(str(model_path))

    if not isinstance(loaded, trimesh.Scene):
        return render(loaded, output_path, **kwargs)

    # Bake each part's material colour into vertex colours before merging, so
    # a single painted mesh still shows stone against glass.
    painted = []
    for geometry in loaded.geometry.values():
        colour = _material_colour(geometry)
        copy = geometry.copy()
        copy.metadata["base_colour"] = colour if colour is not None else BASE_COLOUR
        painted.append(copy)

    merged = trimesh.util.concatenate(painted)
    face_colours = np.vstack(
        [
            np.tile(part.metadata["base_colour"], (len(part.faces), 1))
            for part in painted
        ]
    )
    return render(merged, output_path, face_colours=face_colours, **kwargs)
