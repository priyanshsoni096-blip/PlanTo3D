"""What the finished house is made of and what light falls on it.

Everything here was a constant buried in ``materials`` or ``preview``, which
meant one house: pale render walls, a grey deck, midday sun. A drawing does
not say what a building is clad in -- that is the architect's decision, not
the plan's -- so it belongs to whoever is running the pipeline rather than
to the code.

Two independent things are settled here and they are worth keeping apart:

``Lighting``  the time of day and the weather. Changes the whole image.
``Palette``   what each surface is made of. Changes one part of it.

Both carry the previous hard-coded values as their defaults, so a caller
that asks for nothing gets exactly what it got before.
"""

from dataclasses import dataclass, field, replace

Colour = tuple[int, int, int]


@dataclass(frozen=True)
class Lighting:
    """Where the light comes from and what colour it is.

    Colours are 0-255 so they can be typed into a colour picker like any
    other; the renderer converts them. Strengths are multipliers, and the
    defaults are balanced so a mid-toned wall in full sun lands just under
    white before the filmic curve rolls it off.
    """

    sun: Colour = (255, 237, 209)
    sky: Colour = (148, 179, 224)
    bounce: Colour = (102, 89, 74)
    fill: Colour = (179, 201, 240)

    # Ambient lights every surface whatever way it faces, so it is the one
    # control that flattens a building rather than lighting it. At 0.58,
    # with a key of 0.78 and a fill on top, more light arrived than the
    # filmic curve had room for: measured on a rendered plan, half the
    # picture sat above tone 189 with the 75th to 99th percentiles crushed
    # between 213 and 221, and median saturation was 23 of 255 -- close to
    # grey, from a palette spanning 145 levels of luminance.
    #
    # Lowering ambient and the exposure together, scored on what actually
    # reaches the pixels:
    #
    #     ambient  exposure   spread   saturation
    #       0.58      1.05        35           28    <- was
    #       0.42      0.90        71           34
    #     * 0.35      0.90        75           35
    #       0.28      0.82        80           41
    #
    # It keeps improving as both come down, so this is a judgement about
    # where to stop rather than a peak: at 0.35 and 0.90 the form reads and
    # nothing is crushed, and going further trades a murkier shaded wall
    # for a few more points. Raising the key instead does not work -- it
    # pushes the lit faces back into the roll-off and the spread falls
    # again, to 34.
    ambient_strength: float = 0.35
    key_strength: float = 0.78
    fill_strength: float = 0.16
    specular_strength: float = 0.28

    # Lifted before the filmic curve, which pulls mid tones down as it
    # rolls the highlights off. Under one, because the curve was being
    # asked to roll off more than it could.
    exposure: float = 0.90

    sky_top: Colour = (96, 140, 190)
    sky_bottom: Colour = (206, 220, 232)
    sky_glow: Colour = (255, 244, 226)
    glow_strength: float = 0.42

    shadow_strength: float = 0.38
    occlusion_strength: float = 0.30


# Named conditions, because "warmer sun and a lower key" is not how anyone
# thinks about this. Each is the whole image at a time of day rather than a
# tint laid over one look: the sun colour, the sky behind it, what bounces
# off the ground and how hard the shadows fall all move together, because
# in daylight they do.
LIGHTING_PRESETS: dict[str, Lighting] = {
    "midday": Lighting(),
    "golden hour": Lighting(
        sun=(255, 208, 148),
        sky=(158, 172, 209),
        bounce=(122, 96, 71),
        fill=(186, 190, 224),
        # The sun is low, so it does more of the work and the sky less.
        ambient_strength=0.48,
        key_strength=0.92,
        exposure=1.0,
        sky_top=(112, 143, 191),
        sky_bottom=(247, 214, 176),
        sky_glow=(255, 226, 178),
        glow_strength=0.6,
        shadow_strength=0.46,
    ),
    "overcast": Lighting(
        # No sun to speak of: light arrives from the whole sky at once, so
        # the key almost vanishes and shadows go soft and shallow. This is
        # the honest setting for judging a massing, since nothing is
        # flattered by a good raking light.
        sun=(224, 227, 232),
        sky=(186, 193, 204),
        bounce=(128, 126, 122),
        fill=(199, 205, 214),
        ambient_strength=0.86,
        key_strength=0.22,
        fill_strength=0.20,
        specular_strength=0.1,
        exposure=1.0,
        sky_top=(176, 186, 199),
        sky_bottom=(214, 219, 224),
        sky_glow=(232, 236, 240),
        glow_strength=0.25,
        shadow_strength=0.18,
        occlusion_strength=0.36,
    ),
    "dusk": Lighting(
        sun=(255, 176, 122),
        sky=(94, 112, 163),
        bounce=(71, 63, 66),
        fill=(122, 138, 189),
        ambient_strength=0.42,
        key_strength=0.72,
        specular_strength=0.4,
        exposure=0.92,
        sky_top=(38, 58, 107),
        sky_bottom=(214, 156, 130),
        sky_glow=(255, 196, 145),
        glow_strength=0.68,
        shadow_strength=0.3,
    ),
}


# Surfaces worth offering, in the order they make sense to set: the ones
# that decide what the building looks like from across the street first.
# The rest of the palette follows whatever these do not cover.
PRINCIPAL_SURFACES = (
    "wall",
    "roof",
    "coping",
    "plinth",
    "glass",
    "frame",
    "railing",
    "lawn",
    "paving",
    "ground",
    "boundary",
    "pitched",
    "dome",
    "tower",
    "tank",
    "chimney",
    "canopy",
    "timber",
    "tile",
    "stone",
)


@dataclass(frozen=True)
class Palette:
    """Colour overrides by surface name.

    Only what the caller sets is changed; anything absent keeps the built-in
    material, including its roughness. Roughness is deliberately not exposed
    -- it is what separates glass from masonry, and a picker offering it
    invites a polished brick wall.
    """

    colours: dict[str, Colour] = field(default_factory=dict)

    def for_surface(self, name: str, default: Colour) -> Colour:
        return self.colours.get(name, default)

    def with_colour(self, name: str, colour: Colour) -> "Palette":
        return replace(self, colours={**self.colours, name: colour})


# Whole schemes, so a house can be given a character in one choice rather
# than twenty. Each names only what it needs to; everything else is left as
# it was, which is why a scheme is a handful of lines and not a full
# palette.
PALETTE_PRESETS: dict[str, dict[str, Colour]] = {
    "warm render": {},
    "white modern": {
        "wall": (232, 230, 226),
        "coping": (240, 239, 236),
        "plinth": (186, 184, 180),
        "boundary": (222, 220, 216),
        "frame": (42, 42, 44),
        "canopy": (236, 234, 230),
    },
    "grey stone": {
        "wall": (168, 166, 160),
        "coping": (150, 148, 143),
        "plinth": (124, 122, 118),
        "boundary": (158, 156, 150),
        "roof": (108, 106, 102),
        "canopy": (160, 158, 152),
    },
    "red brick": {
        "wall": (158, 96, 74),
        "coping": (188, 180, 168),
        "plinth": (118, 74, 58),
        "boundary": (150, 92, 72),
        "frame": (54, 48, 44),
        "canopy": (172, 108, 84),
    },
    "sandstone": {
        "wall": (206, 172, 122),
        "coping": (196, 178, 148),
        "plinth": (168, 136, 96),
        "boundary": (198, 166, 118),
        "canopy": (204, 176, 132),
    },
}


def parse_colour(value: str | Colour | None) -> Colour | None:
    """Accept a hex string, an ``rgb(...)`` string, or a triple.

    Colour pickers hand back whichever of these they feel like, and a
    picker's output arriving as an unusable string is a poor reason to lose
    a user's choice.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(int(channel) for channel in value[:3])  # type: ignore[return-value]

    text = str(value).strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(character * 2 for character in digits)
        if len(digits) != 6:
            return None
        return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    if text.lower().startswith("rgb"):
        numbers = [
            part.strip()
            for part in text[text.find("(") + 1 : text.rfind(")")].split(",")
        ]
        if len(numbers) >= 3:
            return tuple(int(round(float(number))) for number in numbers[:3])  # type: ignore[return-value]

    return None


def to_hex(colour: Colour) -> str:
    """A colour as ``#rrggbb``, for handing to a picker."""
    return "#{:02x}{:02x}{:02x}".format(*(int(channel) for channel in colour))
