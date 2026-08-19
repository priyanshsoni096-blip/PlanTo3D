"""Give the model materials, so it reads as a building rather than a massing study.

Materials ride in the glTF file itself as PBR definitions, which every web
viewer understands -- including the one already in the app. No render engine
is involved, so the interactive model gains stone, concrete and glass without
a Blender round trip.

Roughness carries most of the read here. Plaster and concrete are almost
fully rough and diffuse; glazing is near-mirror and translucent, which is
what makes an opening look glazed rather than boarded up.
"""

import logging
from pathlib import Path

import trimesh
from trimesh.visual.material import PBRMaterial

from planto3d.extrude import DEFAULT_WALL_HEIGHT_FT, floors_to_parts
from planto3d.geometry_types import FloorPlan

logger = logging.getLogger(__name__)


class Surface:
    """A named appearance: colour, how rough it is, and how transparent."""

    def __init__(
        self,
        name: str,
        colour: tuple[int, int, int],
        roughness: float,
        metallic: float = 0.0,
        opacity: float = 1.0,
    ):
        self.name = name
        self.colour = colour
        self.roughness = roughness
        self.metallic = metallic
        self.opacity = opacity

    def to_material(self) -> PBRMaterial:
        red, green, blue = (channel / 255 for channel in self.colour)
        return PBRMaterial(
            name=self.name,
            baseColorFactor=[red, green, blue, self.opacity],
            roughnessFactor=self.roughness,
            metallicFactor=self.metallic,
            # Blending is only requested where it is needed; declaring it on
            # opaque surfaces makes viewers sort them needlessly and can make
            # solid walls flicker against each other.
            alphaMode="BLEND" if self.opacity < 1.0 else "OPAQUE",
            doubleSided=True,
        )


# Warm limestone and concrete, pitched for contrast rather than subtlety.
#
# An earlier, paler palette was architecturally tasteful and useless in
# practice: web viewers light a model brightly, and off-whites separated by a
# few percent all washed to the same pink-white. Surfaces are now spaced far
# enough apart in tone that walls, slabs and roof stay distinguishable
# whatever the viewer does with them.
SURFACES = {
    "wall": Surface("wall", (216, 201, 176), roughness=0.92),
    "floor": Surface("floor", (170, 163, 152), roughness=0.85),
    "roof": Surface("roof", (128, 124, 118), roughness=0.88),
    "glass": Surface("glass", (146, 190, 214), roughness=0.06, metallic=0.1, opacity=0.35),
    "ground": Surface("ground", (112, 108, 100), roughness=0.97),
    "lawn": Surface("lawn", (86, 132, 56), roughness=0.98),
    "paving": Surface("paving", (146, 139, 128), roughness=0.9),
    "boundary": Surface("boundary", (198, 186, 166), roughness=0.94),
    # Mid-grey metal rather than near-black. A rail is a slim element seen
    # against the sky, and at (72,70,68) it read as a solid black mass across
    # the facade instead of a balustrade.
    "railing": Surface("railing", (138, 136, 132), roughness=0.4, metallic=0.6),
    # Window frames: dark slim members, the way glazing is detailed in the
    # reference elevations.
    "frame": Surface("frame", (58, 56, 54), roughness=0.45, metallic=0.4),
    # Water: smooth and translucent, so a pool reads as depth rather than
    # a blue slab.
    "water": Surface("water", (58, 132, 168), roughness=0.04, opacity=0.72),
    # Tiled wet areas, lighter and glossier than the rooms around them.
    "wet": Surface("wet", (214, 214, 210), roughness=0.28),
    # Stairs, a shade lighter than the slabs so the flight reads against them.
    "stairs": Surface("stairs", (188, 184, 178), roughness=0.8),
    # The plinth, in a darker stone than the walls it carries -- a base
    # reads as a base by being heavier than what stands on it.
    "plinth": Surface("plinth", (168, 160, 148), roughness=0.9),
    # Interior floor finishes, chosen from what each room is for. Warm where
    # a room is lived in, cool and harder where it is worked in.
    "timber": Surface("timber", (152, 102, 58), roughness=0.55),
    "tile": Surface("tile", (198, 196, 190), roughness=0.32),
    "stone": Surface("stone", (172, 162, 146), roughness=0.6),
    # Coping, a shade paler than the parapet it caps so the roofline reads
    # as a defined edge rather than a raw extrusion.
    "coping": Surface("coping", (208, 202, 192), roughness=0.75),
}

FALLBACK = Surface("default", (200, 200, 200), roughness=0.9)


def build_scene(
    floors: list[FloorPlan],
    wall_height_ft: float = DEFAULT_WALL_HEIGHT_FT,
    scale: float = 1.0,
    page_size: tuple[int, int] | None = None,
) -> trimesh.Scene:
    """Assemble the building as a scene with one material per surface type."""
    parts = floors_to_parts(
        floors, wall_height_ft=wall_height_ft, scale=scale, page_size=page_size
    )

    scene = trimesh.Scene()
    for name, meshes in parts.items():
        combined = trimesh.util.concatenate(meshes)
        surface = SURFACES.get(name, FALLBACK)
        combined.visual = trimesh.visual.TextureVisuals(material=surface.to_material())
        scene.add_geometry(combined, node_name=name, geom_name=name)
        logger.info("%s: %d faces", name, len(combined.faces))

    return scene


def export_scene(scene: trimesh.Scene, output_path: Path) -> Path:
    """Write a materialled scene as binary glTF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
    logger.info("wrote %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
