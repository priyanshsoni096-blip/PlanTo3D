"""Prepare a geometrically-correct model for a diffusion stylization pass.

This is the one stage that invents rather than measures. Everything before it
traces back to the drawing; a diffusion model adds stone coursing, dusk
lighting, planting and reflections that a floor plan simply does not contain.
Guiding it with our own depth and edges keeps the invention pinned to the
real massing, so the result is this house rendered convincingly rather than a
plausible house that happens to look similar.

Only the guides and the prompt are built here. The generation itself wants a
GPU and runs in notebooks/photoreal_on_colab.ipynb.

Rendering and image dependencies are imported inside the functions that need
them. The generation environment installs diffusion packages, not the 3D
stack, and it only ever wants the prompt -- a module-level ``import trimesh``
would break it on a machine that has no reason to hold a mesh library.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the guides are shot from. A three-quarter aerial reads as
# architectural photography and shows massing, roof and site at once.
GUIDE_AZIMUTH = 38.0
GUIDE_ELEVATION = 30.0
GUIDE_RESOLUTION = (1024, 768)

# Canny thresholds. Deliberately loose: the render is flat-shaded, so its
# edges are already clean, and tight thresholds drop the subtle steps between
# storeys that give the facade its depth.
CANNY_LOW = 40
CANNY_HIGH = 130

BASE_PROMPT = (
    "professional architectural visualization, modern {storeys}-storey "
    "residence, warm evening light at dusk, glowing interior lighting through "
    "large windows, limestone and glass facade, flat roof with parapet, "
    "landscaped garden, paved driveway, soft ambient sky, photorealistic, "
    "high detail, architectural photography, 8k"
)

NEGATIVE_PROMPT = (
    "cartoon, illustration, sketch, blurry, distorted perspective, "
    "warped walls, floating geometry, watermark, text, people, oversaturated"
)


def build_prompt(storeys: int, room_labels: list[str] | None = None) -> str:
    """Compose the prompt, naming what the plan actually contains.

    Grounding the description in extracted labels keeps the model from
    inventing features the house does not have -- a pool, a pitched roof --
    which is the usual way a stylization stops resembling its subject.
    """
    prompt = BASE_PROMPT.format(storeys=storeys)

    labels = {label.upper() for label in room_labels or []}
    if any("PARKING" in label for label in labels):
        prompt += ", cars parked in the driveway"
    if any("GARDEN" in label or "LANDSCAPE" in label for label in labels):
        prompt += ", lush green lawn"
    if any("TERRACE" in label for label in labels):
        prompt += ", roof terrace with planting"

    return prompt


def edge_guide(render_path: Path, output_path: Path) -> Path:
    """Canny edges of a render, for an edge-conditioned ControlNet."""
    import cv2

    image = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read render: {render_path}")

    edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), edges)
    logger.info("wrote edge guide %s", output_path)
    return output_path


def build_guides(
    model_path: Path,
    output_dir: Path,
    azimuth: float = GUIDE_AZIMUTH,
    elevation: float = GUIDE_ELEVATION,
    resolution: tuple[int, int] = GUIDE_RESOLUTION,
) -> dict[str, Path]:
    """Render the shaded view, depth map and edge map a diffusion pass needs.

    All three share one camera, so the guides agree with each other -- a
    depth map and an edge map shot from different angles fight, and the model
    resolves the conflict by warping the building.
    """
    from planto3d.preview import _painted, render, render_depth

    mesh, colours = _painted(Path(model_path))
    output_dir = Path(output_dir)

    shaded = render(
        mesh,
        output_dir / "guide-render.png",
        resolution=resolution,
        azimuth=azimuth,
        elevation=elevation,
        face_colours=colours,
    )
    depth = render_depth(
        mesh,
        output_dir / "guide-depth.png",
        resolution=resolution,
        azimuth=azimuth,
        elevation=elevation,
    )
    edges = edge_guide(shaded, output_dir / "guide-edges.png")

    return {"render": shaded, "depth": depth, "edges": edges}
