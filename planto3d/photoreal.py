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

# Where the guides are shot from. Architectural photography sits low enough
# that the facade dominates and the roof is only glimpsed. Shooting from
# higher gives a plan-like view where most of the frame is roof, and the
# elevation -- which is what makes a house look like a house -- is lost.
GUIDE_AZIMUTH = 38.0
GUIDE_ELEVATION = 26.0
GUIDE_RESOLUTION = (1024, 768)

# Canny thresholds. Deliberately loose: the render is flat-shaded, so its
# edges are already clean, and tight thresholds drop the subtle steps between
# storeys that give the facade its depth.
CANNY_LOW = 40
CANNY_HIGH = 130

BASE_PROMPT = (
    "professional architectural visualization of a modern {storeys}-storey "
    "luxury residence at dusk, warm honey-toned limestone cladding with "
    "visible stone coursing, floor-to-ceiling glazing in slim dark frames, "
    "flat roof with a stone parapet, "
    # Light is what separates an evening render from a daytime one, and it
    # has to be described as fittings rather than as a mood.
    "warm amber interior lighting glowing through every window, "
    "recessed wall washers along the facade, landscape uplighting in the "
    "planting beds, small warm lights lining the terrace, "
    # Only what any house has. Planting, lawns and cars are added below,
    # and only where the plan actually shows them -- a render claiming a
    # garden the drawing does not have stops describing this building.
    "mature trees beyond the plot, "
    "deep blue twilight sky, long soft shadows, "
    "photorealistic architectural photography, ultra detailed, 8k"
)

NEGATIVE_PROMPT = (
    "cartoon, illustration, sketch, diagram, blurry, distorted perspective, "
    "warped walls, floating geometry, watermark, text, signage, people, "
    "oversaturated, flat lighting, daylight, overcast, bare concrete, "
    "unfinished construction, empty plot, barren ground"
)


def build_prompt(storeys: int, room_labels: list[str] | None = None) -> str:
    """Compose the prompt, naming what the plan actually contains.

    Grounding the description in extracted labels keeps the model from
    inventing features the house does not have -- a pool, a pitched roof --
    which is the usual way a stylization stops resembling its subject.
    """
    prompt = BASE_PROMPT.format(storeys=storeys)

    labels = {label.upper() for label in room_labels or []}
    if any("PARKING" in label or "GARAGE" in label for label in labels):
        prompt += ", two parked cars on a block-paved driveway"
    if any("GARDEN" in label or "LANDSCAPE" in label or "LAWN" in label for label in labels):
        prompt += (
            ", manicured lawn edged with clipped hedges and flowering shrubs, "
            "landscape uplighting through the planting"
        )
    if any("TERRACE" in label for label in labels):
        prompt += ", roof terrace laid to lawn with planters around its edge"
    if any("BALCONY" in label or "BAL" == label for label in labels):
        prompt += ", balconies with slim metal railings"
    if any("POOL" in label or "SWIMMING" in label for label in labels):
        prompt += ", lit swimming pool"
    if any("TEMPLE" in label or "POOJA" in label for label in labels):
        prompt += ", warm timber detailing"

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

    mesh, colours, reflective, roughness = _painted(Path(model_path))
    output_dir = Path(output_dir)

    shaded = render(
        mesh,
        output_dir / "guide-render.png",
        resolution=resolution,
        azimuth=azimuth,
        elevation=elevation,
        face_colours=colours,
        reflective=reflective,
        roughness=roughness,
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
