"""The handful of choices that decide how the finished house looks.

A drawing fixes the geometry and says nothing about the building: not what
it is clad in, not what hour it is seen at, not whether there is a garden.
Those are decisions, and they were being made in code.

The whole surface is five choices. That is deliberate. An earlier version
offered a colour picker per surface -- ten of them -- which is a spreadsheet
rather than a choice: it asks someone to design a palette when what they
wanted was a house that looks a particular way. These five each change the
image substantially, none of them can produce an ugly result, and together
they cover the range worth covering.

    style        the building's character
    colour       how light or dark, how warm
    time         the hour it is seen at
    landscaping  how much of a setting it gets
    creativity   how far the photoreal pass may stray from the plan

``style`` and ``colour`` compose rather than enumerate: four characters
times three tones would be twelve palettes to write and keep consistent,
so the style says what the building is made of and the tone says how light
and how warm to take it.
"""

import colorsys
from dataclasses import dataclass

from planto3d.style import LIGHTING_PRESETS, Colour, Lighting, Palette

# --- the building's character ------------------------------------------------
#
# Each names only the surfaces that give it its character. Anything absent
# keeps the built-in material, which is why these are a handful of lines
# rather than a full palette each.

STYLES: dict[str, dict[str, Colour]] = {
    # Pale render, dark slim frames, a flat grey deck. The default, and the
    # look most new-build houses are drawn as.
    "modern": {
        "wall": (216, 213, 207),
        "coping": (198, 196, 192),
        "plinth": (156, 154, 150),
        "frame": (44, 44, 46),
        "boundary": (206, 203, 198),
        "canopy": (210, 207, 202),
    },
    # Warm stone, bronze frames, a deeper base. Heavier and more expensive
    # looking, which is what "luxury" means on an elevation.
    "luxury": {
        "wall": (206, 178, 138),
        "coping": (188, 172, 146),
        "plinth": (142, 120, 92),
        "frame": (78, 62, 42),
        "railing": (150, 126, 84),
        "boundary": (196, 172, 138),
        "canopy": (200, 176, 142),
        "dome": (206, 186, 150),
        "tower": (202, 178, 140),
    },
    # Brick and tile: the vernacular of most of the plans this reads.
    "traditional": {
        "wall": (176, 124, 96),
        "coping": (186, 178, 166),
        "plinth": (128, 92, 72),
        "roof": (146, 84, 62),
        "pitched": (150, 82, 58),
        "frame": (72, 58, 48),
        "boundary": (170, 122, 96),
        "canopy": (182, 134, 104),
    },
    # White, flat, and almost nothing else. The frames go pale too, because
    # a dark frame is a detail and this style is the absence of detail.
    "minimalist": {
        "wall": (234, 233, 230),
        "coping": (236, 235, 232),
        "plinth": (198, 197, 194),
        "roof": (206, 205, 202),
        "frame": (128, 128, 130),
        "railing": (190, 190, 192),
        "boundary": (228, 227, 224),
        "canopy": (232, 231, 228),
    },
}

DEFAULT_STYLE = "modern"


# --- how light, and how warm -------------------------------------------------


@dataclass(frozen=True)
class Tone:
    """A shift applied to every surface a style names.

    ``lightness`` moves towards white or black; ``warmth`` rotates the hue
    towards orange and lifts saturation a little. Applied in HLS so a brick
    stays recognisably brick when it darkens instead of turning grey, which
    is what happens if the channels are scaled directly.
    """

    lightness: float = 0.0
    warmth: float = 0.0


TONES: dict[str, Tone] = {
    "light": Tone(lightness=0.16, warmth=0.0),
    "dark": Tone(lightness=-0.3, warmth=-0.02),
    "warm": Tone(lightness=0.04, warmth=0.12),
}

DEFAULT_TONE = "warm"

# Surfaces a tone must not touch. Glazing and water take their colour from
# the sky rather than from a scheme, and darkening a window is how you get a
# building with no windows.
UNTONED = frozenset({"glass", "water", "lawn"})


def apply_tone(colour: Colour, tone: Tone) -> Colour:
    """Shift one colour towards a lighter, darker or warmer reading."""
    red, green, blue = (channel / 255 for channel in colour)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    # Towards white or black rather than by a flat amount, so a dark
    # surface cannot be pushed past black and a pale one past white.
    if tone.lightness >= 0:
        lightness += (1.0 - lightness) * tone.lightness
    else:
        lightness += lightness * tone.lightness

    if tone.warmth:
        # 0.08 in HLS is roughly orange. Pull the hue towards it rather
        # than setting it, so a brick stays brick and a grey warms slightly.
        hue += (0.08 - hue) * min(abs(tone.warmth) * 2.5, 1.0) * (
            1 if tone.warmth > 0 else -1
        )
        saturation = min(saturation + tone.warmth * 0.5, 1.0)

    hue %= 1.0
    lightness = min(max(lightness, 0.0), 1.0)
    saturation = min(max(saturation, 0.0), 1.0)

    return tuple(
        int(round(channel * 255))
        for channel in colorsys.hls_to_rgb(hue, lightness, saturation)
    )


# --- the hour ----------------------------------------------------------------

TIMES: dict[str, str] = {
    "day": "midday",
    "sunset": "golden hour",
    "night": "dusk",
}

DEFAULT_TIME = "day"


# --- how much of a setting ---------------------------------------------------


@dataclass(frozen=True)
class Landscaping:
    """How much ground the building is given.

    ``ground`` is the plot it stands on, ``planting`` the lawn and beds the
    drawing marks, ``boundary`` the compound wall around the plot. Turning
    the lot off leaves the building alone against the sky, which is what a
    massing study wants and a presentation render does not.
    """

    ground: bool = True
    planting: bool = True
    boundary: bool = True


LANDSCAPING: dict[str, Landscaping] = {
    "none": Landscaping(ground=False, planting=False, boundary=False),
    "basic": Landscaping(ground=True, planting=True, boundary=False),
    "premium": Landscaping(ground=True, planting=True, boundary=True),
}

DEFAULT_LANDSCAPING = "basic"


# --- how far the photoreal pass may stray ------------------------------------
#
# ControlNet's conditioning scale. High holds the geometry tightly and looks
# increasingly like a shaded model; low invents freely and stops describing
# this building. Measured on the reference sheet, 0.5 gave the richest image
# that was still recognisably the same house.

CREATIVITY: dict[str, float] = {
    "strict": 1.0,
    "balanced": 0.7,
    "creative": 0.5,
}

DEFAULT_CREATIVITY = "balanced"


# --- the whole choice --------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """Everything the drawing does not say, in five choices."""

    style: str = DEFAULT_STYLE
    colour: str = DEFAULT_TONE
    time: str = DEFAULT_TIME
    landscaping: str = DEFAULT_LANDSCAPING
    creativity: str = DEFAULT_CREATIVITY

    def palette(self) -> Palette:
        """The style's surfaces, shifted by the tone."""
        base = STYLES.get(self.style, STYLES[DEFAULT_STYLE])
        tone = TONES.get(self.colour, TONES[DEFAULT_TONE])
        return Palette(
            {
                surface: colour if surface in UNTONED else apply_tone(colour, tone)
                for surface, colour in base.items()
            }
        )

    def lighting(self) -> Lighting:
        preset = TIMES.get(self.time, TIMES[DEFAULT_TIME])
        return LIGHTING_PRESETS[preset]

    def site(self) -> Landscaping:
        return LANDSCAPING.get(self.landscaping, LANDSCAPING[DEFAULT_LANDSCAPING])

    def conditioning(self) -> float:
        return CREATIVITY.get(self.creativity, CREATIVITY[DEFAULT_CREATIVITY])
