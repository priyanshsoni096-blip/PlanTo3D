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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only for the type hint on build_prompt's design parameter. A runtime
    # import is harmless here too (design.py pulls in nothing heavier than
    # colorsys), but keeping it type-checking-only means this module's
    # "no 3D stack required" guarantee never depends on that staying true.
    from planto3d.design import Design

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

# CLIP reads 77 tokens and silently discards the rest. The prompt ran to
# about 144, so half of it was thrown away -- and being the tail, the half
# thrown away was everything derived from the drawing itself: the cars,
# the lawn, the terrace, the balconies. The generic opening survived and
# the specific ending did not, which is exactly backwards.
#
# So the prompt is now assembled to a budget, most important first, and
# stops when it is full.
MAX_PROMPT_TOKENS = 77

# CLIP splits on more than whitespace -- hyphenated and unusual words cost
# several tokens each -- so words are counted at this rate and the budget
# kept short of the limit. Overshooting costs the tail again.
TOKENS_PER_WORD = 1.35
PROMPT_SAFETY = 4

# What each design choice means to a diffusion model, in its own words.
#
# The 3D model already honours these choices through Design.palette() and
# Design.lighting(); until now the photoreal pass did not see them at all,
# so a traditional house at night was still described as a modern one at
# dusk. Two outputs of the same pipeline disagreeing about what building
# they show is worse than either being plain.
#
# Kept short deliberately. Every word here is spent out of a 77-token
# budget that the drawing-derived detail also has to fit inside.

# The material axis. This mirrors STYLES in design.py, where each style's
# own wall colour is the material that actually renders:
#   modern       (216,213,207) pale grey
#   luxury       (206,178,138) sandstone / honey
#   traditional  (176,124,96) brick red, roof (146,84,62) pitched
#   minimalist   (234,233,230) near white
# TONES (below) only tints whatever material is named here -- it names no
# material of its own -- so material belongs to style, not to tone.
STYLE_SUBJECTS = {
    "modern": "modern residence, clean rectilinear massing, pale grey render",
    "luxury": "luxury residence, honey-toned limestone, deep reveals",
    "traditional": "traditional residence, red brick with a pitched tiled roof",
    "minimalist": "minimalist residence, unadorned white planes",
}

# The tint axis. This mirrors TONES in design.py: light lifts lightness,
# dark lowers it, warm rotates the hue towards orange -- a shift applied to
# whatever material STYLE_SUBJECTS names, not a cladding in its own right.
TONE_TINT = {
    "light": "bright pale tones",
    "dark": "deep shadowed tones",
    "warm": "warm golden tones",
}

# Paired with the lighting preset design.py's TIMES maps each choice to
# (day -> "midday", sunset -> "golden hour", night -> "dusk"), so the render
# beside it and the diffusion image agree about the hour. "night" reads as
# dusk deliberately: Design.lighting() gives that choice the dusk preset --
# a twilight sky with a low orange sun, not true darkness -- so describing
# it as plain "night" would contradict the render sitting next to it.
TIME_LIGHT = {
    "day": "clear midday daylight, crisp shadows",
    "sunset": "low golden sunset light, long shadows",
    "night": "dusk, deep blue sky glowing warm at the horizon, "
    "amber interior lighting in every window",
}

# The subject, and nothing that could be dropped without changing what the
# picture is of.
# Never dropped: the subject, what it is made of, and the light. A
# building of the wrong material is wrong in every frame, and a dusk
# render without described lighting is just a dark one -- so these come
# out of the budget before anything competes for it.
BASE_PROMPT = (
    "professional architectural visualization of a {storeys}-storey "
    "modern luxury residence at dusk, warm honey-toned limestone cladding, "
    "warm amber interior lighting in every window"
)

# Everything else, in the order it earns its place. Each is added whole or
# not at all: half a clause describes nothing.
#
# Materials come before light because a building of the wrong material is
# wrong in every frame, while poor light is merely dull. Site features come
# after both, but before the quality words, which are the least specific
# thing in the prompt and the right thing to lose first.
STYLE_PHRASES = (
    "floor-to-ceiling glazing in slim dark frames",
    "landscape uplighting",
    "deep blue twilight sky",
    "flat roof with a stone parapet",
)

QUALITY_PHRASES = (
    "photorealistic architectural photography",
    "ultra detailed",
    "8k",
)


def _tokens(text: str) -> int:
    """Roughly how many tokens a phrase costs CLIP."""
    return int(len(text.replace(",", " ").split()) * TOKENS_PER_WORD)


# Also capped at 77, and trimmed to the terms that actually change the
# image: the things a diffusion model most readily does to a building when
# left to itself.
NEGATIVE_PROMPT = (
    "cartoon, illustration, sketch, diagram, blurry, distorted perspective, "
    "warped walls, floating geometry, watermark, text, people, "
    "flat lighting, daylight, unfinished construction, barren ground"
)


def build_prompt(
    storeys: int,
    room_labels: list[str] | None = None,
    design: "Design | None" = None,
) -> str:
    """Compose the prompt, naming what the plan actually contains.

    Grounding the description in extracted labels keeps the model from
    inventing features the house does not have -- a pool, a pitched roof --
    which is the usual way a stylization stops resembling its subject.

    ``design`` is the same ``planto3d.design.Design`` the renderer uses. Passing
    it makes the prompt describe the house the user asked for; omitting it
    keeps the original wording, so existing callers are unaffected.
    """
    labels = {label.upper() for label in room_labels or []}

    def has(*needles: str) -> bool:
        return any(needle in label for label in labels for needle in needles)

    # What this house has, ahead of what any house has. These are the only
    # phrases the drawing actually justifies, so they are the last thing
    # that should be dropped for want of room -- and under the old order
    # they were the first.
    site = []
    if has("PARKING", "GARAGE", "PORCH"):
        site.append("parked cars on a paved driveway")
    if has("GARDEN", "LANDSCAPE", "LAWN"):
        site.append("manicured lawn with clipped hedges")
    if has("TERRACE"):
        site.append("planted roof terrace")
    if has("BALCONY", "BAL"):
        site.append("balconies with slim metal railings")
    if has("POOL", "SWIMMING"):
        site.append("lit swimming pool")
    if has("TEMPLE", "POOJA"):
        site.append("warm timber detailing")

    if design is None:
        opening = BASE_PROMPT.format(storeys=storeys)
        style_phrases = STYLE_PHRASES
    else:
        # Subject, tint and light, in that order and never dropped: a
        # building of the wrong material is wrong in every frame, and an
        # hour that contradicts the render beside it is worse than plain.
        opening = (
            f"professional architectural visualization of a {storeys}-storey "
            f"{STYLE_SUBJECTS.get(design.style, STYLE_SUBJECTS['modern'])}, "
            f"{TONE_TINT.get(design.colour, TONE_TINT['warm'])}, "
            f"{TIME_LIGHT.get(design.time, TIME_LIGHT['day'])}"
        )

        # With no design, STYLE_PHRASES' fixed twilight sky and flat roof
        # were always true of BASE_PROMPT's one hardcoded house. With a
        # design, TIME_LIGHT and STYLE_SUBJECTS already say what the sky
        # and the roof are -- a midday prompt asking for a twilight sky, or
        # a traditional (pitched-roof) house asking for a flat one,
        # contradicts the opening that was just built. Drop only those two;
        # the other two are style-neutral and still earn their place.
        style_phrases = tuple(
            phrase
            for phrase in STYLE_PHRASES
            if phrase not in ("deep blue twilight sky", "flat roof with a stone parapet")
        )

    parts = [opening]
    budget = MAX_PROMPT_TOKENS - PROMPT_SAFETY - _tokens(parts[0])

    for phrase in [*site, *style_phrases, *QUALITY_PHRASES]:
        cost = _tokens(phrase) + 1
        if cost > budget:
            continue
        parts.append(phrase)
        budget -= cost

    return ", ".join(parts)


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
