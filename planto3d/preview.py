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

# Where the light comes from, in the camera's own frame rather than the
# world's, so a building is lit the same way from every standard view. Turn
# the lights with the world instead and the back elevation renders into its
# own shadow.
KEY_LIGHT = np.array([0.45, 0.8, 0.4])
FILL_LIGHT = np.array([-0.6, 0.35, -0.5])

# Light has colour, and giving it none was the single thing holding these
# renders back. Sun is warm, the sky it comes through is cool, and what
# bounces off the ground is warmer and much darker. A surface lit by one
# scalar brightness can only ever look like grey card: the eye reads the
# *shift* between a sunlit face and a shaded one, not the drop in level.
SUN_COLOUR = np.array([1.0, 0.93, 0.82])
SKY_COLOUR = np.array([0.58, 0.70, 0.88])
GROUND_BOUNCE = np.array([0.40, 0.35, 0.29])
FILL_COLOUR = np.array([0.70, 0.79, 0.94])

# Ambient is a hemisphere rather than a constant: a face looking up sees
# sky, one looking down sees the ground. That single gradient does most of
# the work of making a massing read as solid, because it separates roofs
# from walls from soffits before any direct light lands on them.
# The colours above are already the levels wanted, so ambient is applied at
# full strength: a face looking up sits at the sky's own 0.58-0.88, one
# looking down at the ground bounce's 0.29-0.40. Scaling it down as well
# put every shaded surface near a third of white, and the building came out
# reading as wet slate whatever it was made of.
AMBIENT_STRENGTH = 0.58
KEY_STRENGTH = 0.78
FILL_STRENGTH = 0.16

# A highlight where a surface catches the sun square on. Driven by the
# material's own roughness, so glass and metal flare and masonry does not.
SPECULAR_STRENGTH = 0.28
# Roughness below this is treated as a mirror finish for the exponent's
# sake; at zero the power blows up.
MIN_ROUGHNESS = 0.04

# Glazing and water mirror the sky rather than scattering light. Shaded
# diffusely they go almost black whenever they face away from the key
# light, which is most of the time, and every window on the shaded side of
# a building turns into a hole.
REFLECTIVE_FLOOR = 0.82

REFLECTIVE_MATERIALS = {"glass", "water"}

BASE_COLOUR = np.array([214, 208, 198], dtype=float)

# The sky behind the building: deeper overhead, paler at the horizon, with
# the light gathering towards where the sun sits.
SKY_TOP = np.array([96, 140, 190], dtype=float)
SKY_BOTTOM = np.array([206, 220, 232], dtype=float)
SKY_GLOW = np.array([255, 244, 226], dtype=float)
# How far across the frame the glow sits, and how wide it spreads.
GLOW_CENTRE = np.array([0.72, 0.30])
GLOW_RADIUS = 0.55
GLOW_STRENGTH = 0.42

SHADOW_DIRECTION = np.array([0.55, 0.35])
SHADOW_STRENGTH = 0.38
SHADOW_BLUR = 9

# Rendering larger than asked for and averaging down. Every edge in a
# building is a straight line at some angle to the pixel grid, and without
# this they all come out as staircases -- the one artefact that most says
# "not a photograph".
# Lifted before the filmic curve, which pulls mid tones down as it rolls
# the highlights off. Without this the whole image simply darkens.
EXPOSURE = 1.05

SUPERSAMPLE = 2

# Contact darkening, worked out from the depth buffer: where a surface sits
# behind its own neighbourhood it is in a crevice, and crevices are darker.
# This is what puts a building on the ground rather than in front of it.
OCCLUSION_STRENGTH = 0.3
OCCLUSION_RADIUS = 9
# Depth difference, as a share of the model's depth range, at which a pixel
# counts as fully occluded.
OCCLUSION_DEPTH = 0.012


def _to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB 0-255 to linear 0-1.

    Light adds and multiplies linearly and sRGB does not, so doing the
    arithmetic on the stored bytes quietly gets every shaded surface wrong
    -- most visibly in the mid tones, where the encoding curve is steepest.
    """
    normalised = np.asarray(srgb, dtype=float) / 255.0
    return np.where(
        normalised <= 0.04045,
        normalised / 12.92,
        ((normalised + 0.055) / 1.055) ** 2.4,
    )


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear 0-1 back to sRGB 0-255."""
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * clipped ** (1 / 2.4) - 0.055,
    )
    return encoded * 255.0


def _tonemap(linear: np.ndarray) -> np.ndarray:
    """Roll highlights off instead of clipping them flat.

    A sunlit parapet against a bright sky goes past white long before
    anything else does, and clipping turns it into a hard-edged blank
    shape. This is the fitted ACES curve: it holds the mid tones roughly
    where they were, compresses what is above them, and keeps colour in the
    highlights rather than letting every channel hit the ceiling together.

    It is the difference between a render that looks computed and one that
    looks photographed, and it costs two multiplies.
    """
    exposed = linear * EXPOSURE
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip(
        (exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e), 0.0, 1.0
    )


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


def _shading(
    normals: np.ndarray,
    world_normals: np.ndarray | None = None,
    reflective: np.ndarray | None = None,
    roughness: np.ndarray | None = None,
) -> np.ndarray:
    """Per-face RGB light: a warm sun, a cool sky, and bounce off the ground.

    Returns a multiplier per face and channel rather than one number per
    face. That is the whole point -- lit by a single scalar, every surface
    is the same hue at a different level, which is what grey card looks
    like. What reads as sunlight is the *shift* towards warm on the lit
    face and towards blue in the shade.

    ``world_normals`` are the unrotated normals, needed because the
    hemisphere ambient has to know which way is up in the world while the
    key and fill are fixed to the camera.
    """
    key_direction = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
    fill_direction = FILL_LIGHT / np.linalg.norm(FILL_LIGHT)

    # Hemisphere ambient: sky above, ground bounce below.
    upness = 0.5 + 0.5 * (
        normals[:, 1] if world_normals is None else world_normals[:, 1]
    )
    ambient = GROUND_BOUNCE + (SKY_COLOUR - GROUND_BOUNCE) * upness[:, None]
    light = AMBIENT_STRENGTH * ambient

    key = np.clip(normals @ key_direction, 0, 1)
    light = light + KEY_STRENGTH * key[:, None] * SUN_COLOUR

    fill = np.clip(normals @ fill_direction, 0, 1)
    light = light + FILL_STRENGTH * fill[:, None] * FILL_COLOUR

    if reflective is not None:
        floor = np.maximum(light, REFLECTIVE_FLOOR * SKY_COLOUR)
        light = np.where(reflective[:, None], floor, light)

    return light


def _specular(normals: np.ndarray, roughness: np.ndarray | None) -> np.ndarray:
    """Highlight where a face catches the sun square on.

    Blinn-Phong against the same key light, with the exponent taken from
    the material's own roughness so glass and metal flare while masonry
    stays matt. Added to the shaded colour rather than multiplied into it,
    because a highlight is light arriving at the eye, not the surface
    becoming paler.
    """
    if roughness is None:
        return np.zeros((len(normals), 1))

    key_direction = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
    # The camera looks down -Z, so the halfway vector is between the light
    # and straight out of the screen.
    halfway = key_direction + np.array([0.0, 0.0, 1.0])
    halfway /= np.linalg.norm(halfway)

    alignment = np.clip(normals @ halfway, 0, 1)
    sharpness = np.clip(roughness, MIN_ROUGHNESS, 1.0)
    exponent = 2.0 / sharpness**4
    # Narrow highlights are brighter for the same energy, which is what
    # stops a polished surface reading as merely pale.
    strength = SPECULAR_STRENGTH * (1.0 - sharpness) ** 2

    return (strength * alignment**exponent)[:, None]


def _box_mean(image: np.ndarray, size: int) -> np.ndarray:
    """Mean over a square window, by summed-area table.

    Written out rather than pulled from OpenCV or SciPy because this module
    is deliberately light on imports -- the geometry stages load it without
    wanting an image library behind them.
    """
    before = size // 2
    after = size - 1 - before
    padded = np.pad(image, ((before, after), (before, after)), mode="edge")

    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(0).cumsum(1)

    height, width = image.shape
    total = (
        integral[size : size + height, size : size + width]
        - integral[:height, size : size + width]
        - integral[size : size + height, :width]
        + integral[:height, :width]
    )
    return total / float(size * size)


def _occlusion(depth_buffer: np.ndarray) -> np.ndarray:
    """Darkening per pixel where a surface sits behind its neighbourhood.

    A crude ambient occlusion, and the cheapest large improvement available
    to a rasterizer with no ray casting in it. Where the geometry around a
    pixel is nearer than the pixel itself, that pixel is down in a crevice
    -- an inside corner, the joint of a wall and the ground -- and less of
    the sky reaches it.

    Without this a building floats: every junction renders as a clean seam
    between two evenly lit planes, which is not what daylight does.
    """
    covered = np.isfinite(depth_buffer)
    if not covered.any():
        return np.ones_like(depth_buffer)

    depth = np.where(covered, depth_buffer, 0.0)
    spread = float(np.ptp(depth[covered])) or 1.0

    # Averaged over covered pixels only, and normalised by how many there
    # were. Blurring across the background as though it were at depth zero
    # would drag the whole silhouette dark.
    mask = covered.astype(np.float64)
    average = _box_mean(depth, OCCLUSION_RADIUS) / np.maximum(
        _box_mean(mask, OCCLUSION_RADIUS), 1e-6
    )

    # Larger depth is nearer, so a pixel behind its neighbours is occluded.
    behind = np.clip((average - depth) / (spread * OCCLUSION_DEPTH), 0, 1)
    return np.where(covered, 1.0 - OCCLUSION_STRENGTH * behind, 1.0)


def _background(resolution: tuple[int, int]) -> np.ndarray:
    """The sky: deeper overhead, paler at the horizon, warm where the sun is.

    The glow is not decoration. A flat gradient reads as a backdrop behind
    the building; a sky with a light source in it reads as the place the
    building's own highlights are coming from, and the two then agree.
    """
    width, height = resolution

    ramp = np.linspace(0, 1, height)[:, None]
    top, bottom = _to_linear(SKY_TOP), _to_linear(SKY_BOTTOM)
    gradient = top[None, :] * (1 - ramp) + bottom[None, :] * ramp
    sky = np.repeat(gradient[:, None, :], width, axis=1)

    # Distance from the glow's centre, measured on the shorter side so the
    # spread stays circular whatever the frame's proportions.
    across = np.linspace(0, 1, width)[None, :]
    down = np.linspace(0, 1, height)[:, None]
    aspect = width / height
    distance = np.sqrt(
        ((across - GLOW_CENTRE[0]) * aspect) ** 2 + (down - GLOW_CENTRE[1]) ** 2
    )

    falloff = np.clip(1.0 - distance / GLOW_RADIUS, 0, 1) ** 2
    glow = _to_linear(SKY_GLOW)
    return sky + GLOW_STRENGTH * falloff[:, :, None] * (glow - sky)


def _ground_shadow(
    canvas: np.ndarray,
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    scale: float,
    offset: np.ndarray,
    height: int,
) -> None:
    """Darken the ground beneath the building.

    Flat shading gives a model no relationship to what it stands on, so it
    reads as floating however solid the geometry is. A shadow is the cheapest
    cue that fixes that, and the sun direction is already known.
    """
    from PIL import Image, ImageDraw, ImageFilter

    # Drop every vertex to the ground plane, offset along the light.
    flattened = mesh.vertices.copy()
    lift = flattened[:, 1] - mesh.bounds[0][1]
    direction = SHADOW_DIRECTION
    flattened[:, 0] += lift * direction[0]
    flattened[:, 2] += lift * direction[1]
    flattened[:, 1] = mesh.bounds[0][1]

    projected = (rotation @ flattened.T).T[:, :2] * scale + offset
    projected[:, 1] = height - projected[:, 1]

    stencil = Image.new("L", canvas.shape[1::-1], 0)
    painter = ImageDraw.Draw(stencil)
    for face in mesh.faces:
        painter.polygon([tuple(projected[v]) for v in face], fill=255)

    softened = np.asarray(
        stencil.filter(ImageFilter.GaussianBlur(SHADOW_BLUR)), dtype=np.float32
    )
    weight = (softened / 255.0 * SHADOW_STRENGTH)[:, :, None]
    canvas *= 1.0 - weight


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
    roughness: np.ndarray | None = None,
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

    light = _shading(normals, mesh.face_normals, reflective)
    highlight = _specular(normals, roughness)
    base = BASE_COLOUR if face_colours is None else face_colours

    canvas = _background(resolution)
    # The shadow is a multiplier, so it works the same in linear.
    _ground_shadow(canvas, mesh, rotation, scale, offset, height)
    depth_buffer = np.full((height, width), -np.inf)

    # Everything from here is linear light, not stored colour.
    albedo = _to_linear(base)

    for index, face in enumerate(mesh.faces):
        tone = albedo if face_colours is None else albedo[index]
        # The highlight is added rather than multiplied in: it is light
        # arriving at the eye, not the surface turning paler.
        colour = tone * light[index] + highlight[index]
        _rasterize(canvas, depth_buffer, projected[face], colour)

    canvas *= _occlusion(depth_buffer)[:, :, None]
    return canvas, depth_buffer


def render(
    mesh: trimesh.Trimesh,
    output_path: Path,
    resolution: tuple[int, int] = (1400, 1000),
    azimuth: float = DEFAULT_AZIMUTH,
    elevation: float = DEFAULT_ELEVATION,
    face_colours: np.ndarray | None = None,
    reflective: np.ndarray | None = None,
    roughness: np.ndarray | None = None,
    supersample: int = SUPERSAMPLE,
) -> Path:
    """Paint a shaded view of ``mesh`` to a PNG.

    ``face_colours`` gives a per-face base colour; without it the whole mesh
    takes one stone tone. ``reflective`` marks faces that mirror the sky,
    and ``roughness`` drives the highlight.

    Drawn larger than asked for and averaged down. Every edge in a building
    is a straight line at some angle to the pixel grid, and drawn at final
    size they all come out as staircases -- the one artefact that most says
    "not a photograph". Pass ``supersample=1`` where speed matters more.
    """
    from PIL import Image

    width, height = resolution
    factor = max(int(supersample), 1)
    canvas, _ = _draw(
        mesh,
        (width * factor, height * factor),
        azimuth,
        elevation,
        face_colours,
        reflective,
        roughness,
    )

    if factor > 1:
        # Averaged over each block rather than resampled by a filter, so
        # every subpixel counts once and thin members keep their weight.
        # Averaged in linear light, which is the only place averaging means
        # anything -- done on encoded bytes, every antialiased edge comes
        # out darker than both surfaces meeting at it.
        canvas = canvas.reshape(height, factor, width, factor, 3).mean(axis=(1, 3))

    canvas = _to_srgb(_tonemap(canvas))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(output_path)
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


# What a material's roughness is taken to be when the file does not say.
# Matt, because most of a building is.
DEFAULT_ROUGHNESS = 0.85


def _painted(
    model_path: Path,
) -> tuple[trimesh.Trimesh, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load a model merged with per-face colour, reflectivity and roughness.

    The roughness was being discarded, which is why every surface used to
    render equally matt -- the materials had said all along which ones were
    polished and nothing was reading it.
    """
    loaded = trimesh.load(str(model_path))

    if not isinstance(loaded, trimesh.Scene):
        return loaded, None, None, None

    parts, colours, reflective, roughness = [], [], [], []
    for name, geometry in loaded.geometry.items():
        colour = _material_colour(geometry)
        material = getattr(geometry.visual, "material", None)
        material_name = getattr(material, "name", None) or name

        factor = getattr(material, "roughnessFactor", None)
        polish = DEFAULT_ROUGHNESS if factor is None else float(factor)

        count = len(geometry.faces)
        parts.append(geometry)
        colours.append(
            np.tile(colour if colour is not None else BASE_COLOUR, (count, 1))
        )
        reflective.append(np.full(count, material_name in REFLECTIVE_MATERIALS))
        roughness.append(np.full(count, polish))

    return (
        trimesh.util.concatenate(parts),
        np.vstack(colours),
        np.concatenate(reflective),
        np.concatenate(roughness),
    )


def render_glb(model_path: Path, output_path: Path, **kwargs) -> Path:
    """Load a .glb and render it, honouring per-material colours."""
    mesh, colours, reflective, roughness = _painted(Path(model_path))
    return render(
        mesh,
        output_path,
        face_colours=colours,
        reflective=reflective,
        roughness=roughness,
        **kwargs,
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
    mesh, colours, reflective, roughness = _painted(Path(model_path))
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
            roughness=roughness,
        )

    logger.info("rendered %d view(s) to %s", len(rendered), output_dir)
    return rendered
