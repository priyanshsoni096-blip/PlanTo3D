"""Render a measured model with a real renderer, deterministically.

The diffusion pass in ``photoreal.py`` is beautiful and invents -- by
construction, for every plan. This is the other half of that trade: an
image that looks like an architectural visualisation and in which every
surface is one the drawing actually supports. There is no generative
step anywhere in here.

Blender is driven headlessly through ``bpy``. That is a 659 MB
dependency, so it is optional: this module is safe to import without it
and ``available()`` says whether the path can run.

``preview.py`` stays as it is. It is fast, needs nothing but numpy, and
every measurement script uses it. This is the slow, pretty alternative,
not a replacement.
"""

import logging
from pathlib import Path

from planto3d import materials

logger = logging.getLogger(__name__)

# Cycles, not EEVEE -- measured on this project, twice, in both orders:
# a 640x480 frame takes 7.7s under Cycles and 57.3s under EEVEE. That is
# the opposite of the usual advice, and the reason is that EEVEE is a
# rasteriser needing a GL context; headless with no GPU it falls back to
# software. Cycles is a CPU path tracer and does not care. It is also the
# better-looking of the two, so there is no trade to make.
ENGINE = "CYCLES"

# Doubling samples from 32 to 64 costs 1.8s on that same frame, so
# quality is nearly free and the default sits above the floor rather than
# on it. Raise it for a hero image; the cost is close to linear.
DEFAULT_SAMPLES = 64

# How far the camera sits from the model, as a multiple of the model's
# own bounding radius. Far enough that a wide building still fits the
# frame at the standard focal length, close enough that the render is not
# mostly sky.
CAMERA_DISTANCE = 2.6

# How far the ground plane extends past the building, as a multiple of the
# model's own radius. Wide enough that the horizon is never visible inside
# the frame at any of the six standard views, which would read as the
# building standing on a table.
GROUND_EXTENT = 12.0

# The ground is darker than the sky it reflects, as real ground is. Kept
# neutral rather than green: the model already builds its own lawn where
# the drawing says there is one, and a green plane under a paved plot
# would contradict the drawing.
GROUND_TONE = 0.18

# style.py's strengths are multipliers tuned for its own rasterizer, not
# watts. These convert them into something Cycles can use. Measured by
# rendering data/bridge/11001.gif at midday, golden hour and dusk and
# looking: at the brief's starting guess of KEY_ENERGY=4.0, FILL_ENERGY=60
# the lit walls stayed muddy (mean pixel 113.6 on a front-ish view at
# midday) even though nothing was clipping. Raising both, in the same
# 1:15.6 ratio so the fill keeps the same relationship to the key that
# style.py intends, to 9.0/140.0 brought the same frame's mean up to
# 146.9 with no channel clipping (checked at all three presets, including
# golden hour's higher key_strength of 0.92 -- still 0.0 pixels at 253+).
KEY_ENERGY = 9.0
FILL_ENERGY = 140.0

# The sun's angular diameter in radians, which is what softens the shadow
# edge. The real sun subtends about 0.53 degrees; this is deliberately
# wider, because a hard-edged shadow is the most obvious tell of a
# synthetic render and a little softness reads better at these sizes.
SUN_ANGLE = 0.06


# What each exported surface is made of, as Principled BSDF settings.
#
# Fix round 1: this used to be a hand-written table of 13 entries, sized
# by importing one demo plan's glb and reading back what materials it
# happened to contain. planto3d/materials.py actually defines 24 named
# surfaces, and the missing eleven -- including "water", which
# materials.py already treats as essentially glass -- silently fell back
# to DEFAULT_SHADER on any plan that produced them. A hand-copied table
# is duplicated data, and duplicated data drifts; this one already had.
#
# So this is now a derivation, not a second table: every entry in
# materials.SURFACES maps straight across. roughness and metallic carry
# through unchanged. materials.py has no "transmission" concept -- it
# expresses translucency as glTF opacity instead -- so transmission is
# derived as 1 - opacity: fully opaque (1.0) means zero transmission,
# and near-transparent surfaces (glass at 0.35 opacity, water at 0.72)
# come out transmissive without a second hand-tuned number to keep in
# sync. This makes the Blender render, the rasterizer preview and the
# .glb itself all read from the one source of truth in materials.py, and
# makes this specific kind of drift impossible rather than just unlikely.
#
# Colour is deliberately absent: the glb already carries the palette the
# user chose through design.py, and overriding it here would silently
# discard that choice. Only the physical character is set.
SURFACE_SHADERS: dict[str, dict] = {
    name: {
        "metallic": surface.metallic,
        "roughness": surface.roughness,
        "transmission": 1.0 - surface.opacity,
    }
    for name, surface in materials.SURFACES.items()
}

# Anything the table does not name. Plaster-like, and deliberately dull
# so an unmapped surface looks wrong rather than plausible. Reaching this
# is meant to be a real anomaly now that the table above tracks
# materials.SURFACES automatically -- see the warning log in
# _apply_materials, which is how that anomaly gets noticed.
DEFAULT_SHADER = {"metallic": 0.0, "roughness": 0.80, "transmission": 0.0}

# Our key -> the socket Blender actually calls it. Blender 5.2 renamed
# "Transmission" to "Transmission Weight", and the obvious shortcut of
# title-casing our own key silently misses it: inputs.get() returns None,
# the setting is skipped, and glass renders as opaque plaster with no
# error anywhere. Indexing by an explicit name raises a KeyError instead,
# which is the failure we want on a version that renames a socket again.
#
# Confirmed on this build (Blender 5.2.1) by probing a fresh Principled
# BSDF's inputs directly: "Metallic" and "Roughness" are present,
# "Transmission" is absent, "Transmission Weight" is present.
SOCKET_NAMES = {
    "metallic": "Metallic",
    "roughness": "Roughness",
    "transmission": "Transmission Weight",
}


def _apply_materials(objects) -> int:
    """Give every imported surface its physical character.

    Returns how many materials were recognised, so a rename in
    materials.py shows up as a number rather than as a render that
    quietly looks like plaster.
    """
    recognised = 0
    for material in {slot.material for obj in objects for slot in obj.material_slots}:
        if material is None:
            continue
        surface = material.name.split("_")[0].split(".")[0].lower()
        settings = SURFACE_SHADERS.get(surface)
        if settings is None:
            # SURFACE_SHADERS is derived from materials.SURFACES, so
            # reaching here means a name materials.py can produce has no
            # entry -- the exact failure mode round 1 hit, just moved
            # from "silent" to "loud". It renders as plaster either way;
            # the warning is what lets someone notice and fix it.
            logger.warning(
                "material %r (surface %r) has no entry in SURFACE_SHADERS "
                "-- rendering as plaster",
                material.name,
                surface,
            )
            settings = DEFAULT_SHADER
        else:
            recognised += 1

        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is None:
            continue
        for key, value in settings.items():
            socket = principled.inputs[SOCKET_NAMES[key]]
            socket.default_value = value

    logger.info(
        "mapped %d of %d material(s) to a known surface",
        recognised,
        len({slot.material for obj in objects for slot in obj.material_slots}),
    )
    return recognised


def _to_linear(colour) -> tuple[float, float, float, float]:
    """A 0-255 style.py colour as the linear RGBA Blender wants.

    style.py stores colours the way a colour picker shows them, which is
    sRGB; Blender's shader sockets are linear, and handing sRGB straight
    over washes every tint out.
    """
    channels = []
    for value in colour:
        srgb = value / 255
        channels.append(
            srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4
        )
    return (*channels, 1.0)


def _add_world(scene, lighting) -> None:
    """A sky that graduates, rather than a flat grey void.

    Fix round 1: this used to set the Background node straight to
    ``lighting.sky``, a single flat colour -- so every render had a
    uniform-coloured void with no horizon, and dusk in particular came out
    as uniform brown, throwing away the twilight that ``sky_top`` /
    ``sky_bottom`` describe. preview.py draws a graduated sky from those two
    fields (plus a warm glow near the horizon); this now does the Cycles
    equivalent, so the Blender render and the rasterizer agree about what
    time it is, not just what colour the sun is.
    """
    import bpy

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    tree = world.node_tree
    background = tree.nodes["Background"]

    # The standard Cycles recipe for a world gradient: a Geometry node's
    # "Incoming" output is the world-space direction of the camera ray --
    # its Z component is 0 at the horizon and +/-1 at zenith/nadir.
    # Separating out Z and feeding it straight into a colour ramp's
    # Factor -- clamped to [0, 1] by default -- puts sky_bottom exactly at
    # the horizon and sky_top exactly at the zenith with no remapping
    # needed, and clamps anything below the horizon to sky_bottom rather
    # than extrapolating a colour that was never meant to be seen there.
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    tree.links.new(geometry.outputs["Incoming"], separate.inputs["Vector"])
    tree.links.new(separate.outputs["Z"], ramp.inputs["Factor"])
    tree.links.new(ramp.outputs["Color"], background.inputs["Color"])

    elements = ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = _to_linear(lighting.sky_bottom)
    elements[1].position = 1.0
    elements[1].color = _to_linear(lighting.sky_top)

    # sky_glow is the warm band near the sun that style.py's own gradient
    # uses; tried as a third stop low in the ramp (position 0.12, so it
    # sits just above the horizon rather than washing across the whole
    # sky) and kept -- on all three presets it reads as a warm horizon
    # band without muddying the blue above it, most visibly at dusk where
    # it is what makes the twilight recognisable rather than just "sky
    # gets darker at the top".
    glow = elements.new(0.12)
    glow.color = _to_linear(lighting.sky_glow)

    background.inputs["Strength"].default_value = lighting.ambient_strength
    scene.world = world


def _lowest_z() -> float:
    """The bottom of the model, so the ground meets it rather than cutting it."""
    import bpy
    import mathutils

    lows = [
        (obj.matrix_world @ mathutils.Vector(corner)).z
        for obj in bpy.data.objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    return min(lows) if lows else 0.0


def _add_ground(centre, radius, lighting) -> None:
    """A plane for the building to stand on and cast a shadow onto.

    Without it the model floats: Cycles has nothing to catch the contact
    shadow, which is most of what makes a render read as photographed
    rather than assembled.
    """
    import bpy

    bpy.ops.mesh.primitive_plane_add(
        size=radius * GROUND_EXTENT,
        location=(centre[0], centre[1], _lowest_z()),
    )
    plane = bpy.context.object
    material = bpy.data.materials.new("ground_plane")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (
            GROUND_TONE, GROUND_TONE, GROUND_TONE, 1.0
        )
        principled.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(material)


def available() -> bool:
    """Whether Blender can be driven in this environment.

    Callers ask before rendering rather than catching an ImportError from
    somewhere deep inside a scene build.
    """
    try:
        import bpy  # noqa: F401
    except ImportError:
        return False
    return True


def render_view(
    model_path: Path,
    output_path: Path,
    azimuth: float,
    elevation: float,
    resolution: tuple[int, int] = (1200, 900),
    samples: int = DEFAULT_SAMPLES,
    lighting=None,
) -> Path:
    """Render one view of a model, and return where it was written.

    ``azimuth`` and ``elevation`` are degrees, in the same convention
    ``preview.VIEWS`` uses, so the two renderers can be pointed at the
    same angles and compared frame for frame.

    ``lighting`` is a ``style.Lighting`` -- the same object preview.py
    already honours for the user's chosen hour. Left as ``None`` and
    defaulted here (rather than defaulted in the signature) so ``style``
    stays out of module scope, the same way ``bpy`` does.
    """
    import math

    import bpy

    from planto3d.style import Lighting

    lighting = lighting or Lighting()

    model_path, output_path = Path(model_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start from nothing. bpy holds one global scene for the life of the
    # process, so a second render would otherwise inherit the first one's
    # objects, lights and camera.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model_path))

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError(f"no geometry imported from {model_path}")

    _apply_materials(meshes)

    scene = bpy.context.scene
    scene.render.engine = ENGINE
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.filepath = str(output_path.resolve())

    centre, radius = _bounds(meshes)
    _add_camera(scene, centre, radius, azimuth, elevation)
    _add_ground(centre, radius, lighting)
    _add_lighting(scene, centre, radius, lighting)
    _add_world(scene, lighting)

    # Blender's exposure is in stops (each +1 doubles the light reaching
    # the sensor), while style.py's is a linear multiplier -- so this
    # needs a log2, not a direct assignment. Verified against this build,
    # not assumed: rendered a flat emissive plane with the filmic curve
    # switched off (view_transform="Standard", which still sRGB-encodes
    # for display but no longer rolls off highlights) at exposure stops
    # 0, 1 and 2, decoded the sRGB pixels back to linear, and got 0.039,
    # 0.077, 0.153 -- a 1.98x and 1.97x step per +1 stop, matching
    # 2**stops rather than the stops themselves. That confirms
    # log2(lighting.exposure) is the right conversion, not just a guess
    # carried over from the brief.
    scene.view_settings.exposure = math.log2(max(lighting.exposure, 1e-3))

    bpy.ops.render.render(write_still=True)
    logger.info("rendered %s", output_path)
    return output_path


def _bounds(meshes) -> tuple[tuple[float, float, float], float]:
    """The centre of the model and a radius that encloses it."""
    import mathutils

    corners = [
        obj.matrix_world @ mathutils.Vector(corner)
        for obj in meshes
        for corner in obj.bound_box
    ]
    low = mathutils.Vector((min(c[i] for c in corners) for i in range(3)))
    high = mathutils.Vector((max(c[i] for c in corners) for i in range(3)))
    centre = (low + high) / 2
    radius = max((high - low).length / 2, 1e-3)
    return tuple(centre), radius


def _add_camera(scene, centre, radius, azimuth: float, elevation: float) -> None:
    """Place a camera at the given angle, framing the whole model."""
    import math

    import bpy
    import mathutils

    az, el = math.radians(azimuth), math.radians(elevation)
    distance = radius * CAMERA_DISTANCE
    # The glTF importer remaps axes on the way in: X_blender = X_gltf,
    # Y_blender = -Z_gltf, Z_blender = Y_gltf. Against that remap the X
    # term needs a leading minus to keep azimuth increasing clockwise as
    # preview.VIEWS intends -- without it "right" shows the model's left
    # side and vice versa. Front, back and top are unaffected because
    # they don't depend on sin(az)'s sign.
    offset = mathutils.Vector(
        (
            -distance * math.cos(el) * math.sin(az),
            -distance * math.cos(el) * math.cos(az),
            distance * math.sin(el),
        )
    )
    position = mathutils.Vector(centre) + offset

    bpy.ops.object.camera_add(location=position)
    camera = bpy.context.object
    # Point it at the model rather than computing Euler angles by hand.
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (
        mathutils.Vector(centre) - position
    ).to_track_quat("-Z", "Y")
    scene.camera = camera


def _add_lighting(scene, centre, radius, lighting) -> None:
    """A key sun and a soft fill, coloured by the hour style.py chose.

    Mirrors the rasterizer's own rig rather than inventing a second one:
    style.py's Lighting carries a key, a fill and an ambient, and preview.py
    already honours all three. A renderer that ignored them would disagree
    with its own pipeline about what time of day it is.

    The sun's angular size is what softens the shadow. A hard-edged shadow
    is the single most obvious tell of a synthetic render, and the real sun
    subtends about half a degree; this is set wider because a slightly soft
    edge reads better at these resolutions than a physically exact one.
    """
    import bpy
    import mathutils

    key = mathutils.Vector(centre) + mathutils.Vector(
        (radius, -radius, radius * 2.0)
    )
    bpy.ops.object.light_add(type="SUN", location=key)
    sun = bpy.context.object.data
    sun.color = _to_linear(lighting.sun)[:3]
    sun.energy = lighting.key_strength * KEY_ENERGY
    sun.angle = SUN_ANGLE

    fill = mathutils.Vector(centre) + mathutils.Vector(
        (-radius * 1.5, -radius, radius)
    )
    bpy.ops.object.light_add(type="AREA", location=fill)
    area = bpy.context.object
    area.data.color = _to_linear(lighting.fill)[:3]
    area.data.energy = lighting.fill_strength * FILL_ENERGY
    area.data.size = radius
    area.rotation_mode = "QUATERNION"
    area.rotation_quaternion = (
        mathutils.Vector(centre) - area.location
    ).to_track_quat("-Z", "Y")
