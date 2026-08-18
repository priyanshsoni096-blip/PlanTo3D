"""Render a mesh to a PNG without a GPU or display.

Useful for checking the model in a terminal, in CI, or anywhere an OpenGL
context is unavailable.

Rasterizes with a per-pixel depth buffer rather than sorting faces by average
depth. Painter's ordering cannot resolve a large floor slab against the small
wall boxes standing on it -- whichever face wins the sort covers the other
whole -- which showed as diagonal streaks and wedges across otherwise flat
surfaces.

Shading is a key light with a softer fill from the opposite side, plus a
little ambient. A single light leaves faces pointing away from it completely
flat and unreadable.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# Viewing angles for a three-quarter aerial, matching how plans are presented.
DEFAULT_AZIMUTH = 35.0
DEFAULT_ELEVATION = 25.0

KEY_LIGHT = np.array([0.45, 0.8, 0.4])
FILL_LIGHT = np.array([-0.6, 0.35, -0.5])
KEY_STRENGTH = 0.62
FILL_STRENGTH = 0.22
AMBIENT = 0.28
# Glazing and water mirror the sky, so they never go as dark as masonry.
REFLECTIVE_FLOOR = 0.82
# Materials that reflect rather than scatter.
REFLECTIVE_MATERIALS = {"glass", "water"}

BASE_COLOUR = np.array([214, 208, 198], dtype=float)
SKY_TOP = np.array([28, 32, 44], dtype=float)
SKY_BOTTOM = np.array([16, 17, 22], dtype=float)


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


def _shading(normals: np.ndarray, reflective: np.ndarray | None = None) -> np.ndarray:
    """Brightness per face from a key light, a fill light and ambient.

    ``reflective`` marks faces that mirror the sky rather than scattering
    light -- glazing and water. Shaded diffusely they go almost black
    whenever they face away from the key light, which is most of the time,
    and every window on the shaded side of a building turns into a hole.
    """
    key = np.clip(normals @ (KEY_LIGHT / np.linalg.norm(KEY_LIGHT)), 0, 1)
    fill = np.clip(normals @ (FILL_LIGHT / np.linalg.norm(FILL_LIGHT)), 0, 1)
    brightness = AMBIENT + KEY_STRENGTH * key + FILL_STRENGTH * fill

    if reflective is not None:
        brightness = np.where(
            reflective, np.maximum(brightness, REFLECTIVE_FLOOR), brightness
        )
    return brightness


def _background(resolution: tuple[int, int]) -> np.ndarray:
    width, height = resolution
    ramp = np.linspace(0, 1, height)[:, None]
    gradient = SKY_TOP[None, :] * (1 - ramp) + SKY_BOTTOM[None, :] * ramp
    return np.repeat(gradient[:, None, :], width, axis=1)


def _rasterize(
    canvas: np.ndarray,
    depth_buffer: np.ndarray,
    triangle: np.ndarray,
    colour: np.ndarray,
) -> None:
    """Fill one projected triangle, keeping the nearest fragment per pixel."""
    height, width = depth_buffer.shape

    xs, ys, zs = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    left, right = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    top, bottom = int(np.floor(ys.min())), int(np.ceil(ys.max()))

    left, right = max(left, 0), min(right, width - 1)
    top, bottom = max(top, 0), min(bottom, height - 1)
    if left > right or top > bottom:
        return

    # Barycentric coordinates over the bounding box.
    area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
    if abs(area) < 1e-9:  # degenerate after projection
        return

    grid_y, grid_x = np.mgrid[top : bottom + 1, left : right + 1]
    w0 = ((xs[1] - xs[0]) * (grid_y - ys[0]) - (grid_x - xs[0]) * (ys[1] - ys[0])) / area
    w1 = ((grid_x - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (grid_y - ys[0])) / area
    inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
    if not inside.any():
        return

    depth = zs[0] + w1 * (zs[1] - zs[0]) + w0 * (zs[2] - zs[0])
    window = depth_buffer[top : bottom + 1, left : right + 1]
    nearer = inside & (depth > window)
    if not nearer.any():
        return

    window[nearer] = depth[nearer]
    canvas[top : bottom + 1, left : right + 1][nearer] = colour


def _draw(
    mesh: trimesh.Trimesh,
    resolution: tuple[int, int],
    azimuth: float,
    elevation: float,
    face_colours: np.ndarray | None,
    reflective: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a view, returning the colour canvas and its depth buffer."""
    width, height = resolution
    rotation = _rotation(azimuth, elevation)

    vertices = (rotation @ mesh.vertices.T).T
    normals = (rotation @ mesh.face_normals.T).T

    # Fit the model to the canvas with a margin.
    low, high = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = 0.86 * min(width / span[0], height / span[1])
    offset = np.array([width, height]) / 2 - (low + high) / 2 * scale

    projected = np.empty_like(vertices)
    projected[:, :2] = vertices[:, :2] * scale + offset
    projected[:, 1] = height - projected[:, 1]  # screen Y grows downward
    projected[:, 2] = vertices[:, 2]

    brightness = _shading(normals, reflective)
    base = BASE_COLOUR if face_colours is None else face_colours

    canvas = _background(resolution)
    depth_buffer = np.full((height, width), -np.inf)

    for index, face in enumerate(mesh.faces):
        tone = base if face_colours is None else base[index]
        colour = np.clip(tone * brightness[index], 0, 255)
        _rasterize(canvas, depth_buffer, projected[face], colour)

    return canvas, depth_buffer


def render(
    mesh: trimesh.Trimesh,
    output_path: Path,
    resolution: tuple[int, int] = (1400, 1000),
    azimuth: float = DEFAULT_AZIMUTH,
    elevation: float = DEFAULT_ELEVATION,
    face_colours: np.ndarray | None = None,
    reflective: np.ndarray | None = None,
) -> Path:
    """Paint a shaded view of ``mesh`` to a PNG.

    ``face_colours`` gives a per-face base colour; without it the whole mesh
    takes one stone tone. ``reflective`` marks faces that mirror the sky.
    """
    from PIL import Image

    canvas, _ = _draw(mesh, resolution, azimuth, elevation, face_colours, reflective)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.astype(np.uint8)).save(output_path)
    logger.info("wrote preview %s", output_path)
    return output_path


def render_depth(
    mesh: trimesh.Trimesh,
    output_path: Path,
    resolution: tuple[int, int] = (1024, 1024),
    azimuth: float = DEFAULT_AZIMUTH,
    elevation: float = DEFAULT_ELEVATION,
) -> Path:
    """Write a depth map for use as a ControlNet guide.

    Near surfaces are bright and far ones dark, which is the convention
    depth-conditioned models are trained on. Background -- where nothing was
    drawn -- is black, so the model reads it as infinitely far rather than as
    a surface pressed against the camera.
    """
    from PIL import Image

    _, depth = _draw(mesh, resolution, azimuth, elevation, None)

    drawn = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.float64)
    if drawn.any():
        near, far = depth[drawn].max(), depth[drawn].min()
        spread = max(near - far, 1e-9)
        normalized[drawn] = (depth[drawn] - far) / spread

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((normalized * 255).astype(np.uint8), mode="L").save(output_path)
    logger.info("wrote depth guide %s", output_path)
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


def _painted(
    model_path: Path,
) -> tuple[trimesh.Trimesh, np.ndarray | None, np.ndarray | None]:
    """Load a model merged with per-face colours and a reflective mask."""
    loaded = trimesh.load(str(model_path))

    if not isinstance(loaded, trimesh.Scene):
        return loaded, None, None

    parts, colours, reflective = [], [], []
    for name, geometry in loaded.geometry.items():
        colour = _material_colour(geometry)
        material = getattr(geometry.visual, "material", None)
        material_name = getattr(material, "name", None) or name

        parts.append(geometry)
        colours.append(
            np.tile(
                colour if colour is not None else BASE_COLOUR, (len(geometry.faces), 1)
            )
        )
        reflective.append(
            np.full(len(geometry.faces), material_name in REFLECTIVE_MATERIALS)
        )

    return (
        trimesh.util.concatenate(parts),
        np.vstack(colours),
        np.concatenate(reflective),
    )


def render_glb(model_path: Path, output_path: Path, **kwargs) -> Path:
    """Load a .glb and render it, honouring per-material colours."""
    mesh, colours, reflective = _painted(Path(model_path))
    return render(
        mesh, output_path, face_colours=colours, reflective=reflective, **kwargs
    )


# Standard architectural views. Azimuth turns the model about the vertical
# axis; elevation tips the camera. The elevations are taken square-on at zero
# elevation, so they read as drawings rather than perspectives.
VIEWS = {
    "top": (0.0, 90.0),
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (270.0, 0.0),
    "right": (90.0, 0.0),
    "aerial": (DEFAULT_AZIMUTH, 45.0),
}


def render_views(
    model_path: Path,
    output_dir: Path,
    views: dict[str, tuple[float, float]] | None = None,
    resolution: tuple[int, int] = (1200, 900),
    prefix: str = "view",
) -> dict[str, Path]:
    """Render a model from every standard view.

    The model is loaded once and reused, since parsing the glTF costs far
    more than drawing it.
    """
    mesh, colours, reflective = _painted(Path(model_path))
    output_dir = Path(output_dir)

    rendered = {}
    for name, (azimuth, elevation) in (views or VIEWS).items():
        rendered[name] = render(
            mesh,
            output_dir / f"{prefix}-{name}.png",
            resolution=resolution,
            azimuth=azimuth,
            elevation=elevation,
            face_colours=colours,
            reflective=reflective,
        )

    logger.info("rendered %d view(s) to %s", len(rendered), output_dir)
    return rendered
