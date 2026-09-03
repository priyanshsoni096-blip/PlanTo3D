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
) -> Path:
    """Render one view of a model, and return where it was written.

    ``azimuth`` and ``elevation`` are degrees, in the same convention
    ``preview.VIEWS`` uses, so the two renderers can be pointed at the
    same angles and compared frame for frame.
    """
    import math

    import bpy
    import mathutils

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
    _add_lighting(scene, centre, radius)

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


def _add_lighting(scene, centre, radius) -> None:
    """A sun and a sky, sized to the model.

    Deliberately plain at this stage: one key light and an ambient world,
    which is enough to prove the path renders. The material and lighting
    work that makes it look like a photograph is the next task.
    """
    import bpy
    import mathutils

    bpy.ops.object.light_add(
        type="SUN",
        location=mathutils.Vector(centre) + mathutils.Vector((radius, -radius, radius * 2)),
    )
    bpy.context.object.data.energy = 3.0

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world
