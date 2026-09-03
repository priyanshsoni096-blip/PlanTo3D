# Deterministic Blender Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A render that is beautiful *and* is evidence — real materials, soft shadows and global illumination, driven entirely by the measured geometry, with no generative step anywhere in it.

**Architecture:** A new `planto3d/blender_render.py` drives Blender's `bpy` headlessly. It imports the `.glb` the pipeline already exports, maps each of its thirteen named surface materials to a physically-based Cycles shader, and renders the same six standard views `preview.py` produces — mirroring that module's interface so it is a drop-in alternative rather than a replacement. The existing rasterizer stays exactly as it is: it is fast, dependency-light, and the measurement scripts depend on it.

**Tech Stack:** Python 3.11+, `.venv`, pytest, `bpy` 5.2.1 LTS (a 659 MB optional dependency), existing `planto3d.preview` / `planto3d.style` for the view and lighting vocabulary.

**Spec:** `docs/superpowers/specs/2026-09-02-plan-accurate-3d-design.md` — this plan implements its workstream 2.

## Global Constraints

- `pytest -q` must pass. It is at **850**; each task adds tests. Never a net loss.
- Python is `.venv/Scripts/python.exe`; set `PYTHONPATH=.` when running scripts directly.
- **`bpy` is an optional dependency and must stay optional.** It is 659 MB against a 1.8 GB venv. Nothing in `planto3d/` outside the new module may import it at module level, and the whole test suite must pass on a machine without it.
- **Do not change `planto3d/preview.py`.** It is the fast path, the measurement scripts use it, and the spec says it stays.
- No number goes in a doc or commit message unless a script produced it in that session.
- Match existing code style: docstrings and comments explain *why*; every named constant carries a comment giving the reasoning behind its value. `MAX_SCALE_DISAGREEMENT` in `planto3d/pipeline.py` is the house standard.
- Commit messages: short, declarative, describing the *effect*. No conventional-commit prefixes beyond an occasional `docs:`.
- Commit with explicit paths only. Never `git add -A` or `git add .` — `demo_plans/` is intentionally untracked.

---

## What was measured before writing this

Run this session, on this machine, against `bpy` 5.2.1 LTS installed into the project venv.

| Finding | Measurement |
| --- | --- |
| `bpy` on Python 3.13 | installs cleanly — `bpy-5.2.1` plus six small deps |
| Headless render works | yes, without a display or a GPU |
| Engines actually available | `BLENDER_EEVEE`, `BLENDER_WORKBENCH`, `CYCLES` — **no `BLENDER_EEVEE_NEXT`** in this build |
| **Cycles, 640×480, 32 samples** | **7.7 s** |
| Cycles, 640×480, 64 samples | 9.5 s |
| **EEVEE, 640×480** | **57.3 s** |
| Order dependence | none — Cycles first or second gives the same times |
| The pipeline's own `.glb` imports | 13 mesh objects, **13 materials, names intact** |
| Material names that survive | boundary, coping, floor, frame, glass, ground, plinth, railing, roof, stone, timber, wall, wet |
| `bpy` install size | 659 MB |

Two of those decide the design.

**Cycles is the fast engine here, not EEVEE.** That is the opposite of the usual advice and it is measured, twice, in both orders. EEVEE is a rasteriser that needs a GL context; headless with no GPU it falls back to software and costs 57 s. Cycles is a CPU path tracer and does the same frame in 7.7 s. Cycles is also the better-looking of the two. There is no trade here — take Cycles.

**Quality is nearly free.** Doubling samples from 32 to 64 costs 1.8 s. The sample count is not where the budget goes.

**The material names survive the glTF round-trip**, which is what makes a physically-based mapping possible at all: `glass` can become genuine transmission, `railing` a metal, `wet` a tiled floor, without guessing which object is which.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `planto3d/blender_render.py` (create) | The whole Blender path: import the `.glb`, build the world and lights, map materials, place the camera, render. The only file in `planto3d/` that imports `bpy`. |
| `pyproject.toml` (modify) | A `render` extra carrying `bpy`, so the 659 MB is opt-in. |
| `scripts/render_blender.py` (create) | Command-line entry point, mirroring `scripts/run_pipeline.py`'s shape. |
| `tests/test_blender_render.py` (create) | The parts that can be tested without `bpy` installed — the material table, the view geometry, the skip behaviour when the dependency is absent. |

---

## Task 1: Render one view, headless

The smallest thing that proves the whole approach: a `.glb` in, a PNG out, no display, no GPU, no generative step. Everything else builds on this.

**Files:**
- Create: `planto3d/blender_render.py`
- Modify: `pyproject.toml`
- Test: `tests/test_blender_render.py` (create)

**Interfaces:**
- Consumes: the `.glb` written by `planto3d.materials.export_scene`; `planto3d.preview.VIEWS` for the standard view angles (a dict of name to `(azimuth_deg, elevation_deg)`, already defined at `preview.py:628`).
- Produces:
  - `blender_render.available() -> bool` — whether `bpy` can be imported, so callers and tests can skip cleanly.
  - `blender_render.render_view(model_path: Path, output_path: Path, azimuth: float, elevation: float, resolution: tuple[int, int] = (1200, 900), samples: int = DEFAULT_SAMPLES) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_blender_render.py`:

```python
"""The Blender path, tested where it can be tested without Blender.

bpy is a 659 MB optional dependency. The suite must pass on a machine
that does not have it, so everything here either avoids importing it or
skips explicitly -- never silently.
"""

import pytest

from planto3d import blender_render


def test_availability_is_reported_not_guessed():
    # Callers need to know whether the Blender path can run at all, and
    # a bare ImportError deep inside a render is a poor way to find out.
    assert isinstance(blender_render.available(), bool)


def test_the_module_imports_without_bpy_installed():
    # Importing planto3d.blender_render must never require bpy -- the
    # pipeline imports planto3d modules freely and most machines running
    # it will not have Blender.
    assert hasattr(blender_render, "render_view")
    assert hasattr(blender_render, "DEFAULT_SAMPLES")


def test_the_sample_default_is_the_measured_one():
    # 32 samples renders 640x480 in 7.7s; 64 costs 9.5s. Quality is nearly
    # free here, so the default sits above the minimum rather than at it.
    assert blender_render.DEFAULT_SAMPLES >= 32


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_a_model_renders_to_a_real_image(tmp_path):
    from planto3d.geometry_types import FloorPlan, Room, Wall
    from planto3d.materials import build_scene, export_scene

    outline = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    walls = [
        Wall(start=outline[i], end=outline[(i + 1) % 4], thickness=6.0)
        for i in range(4)
    ]
    plan = FloorPlan(
        walls=walls,
        rooms=[Room(polygon=outline, label="BEDROOM")],
        openings=[],
        footprint=outline,
    )
    model = export_scene(build_scene([plan], scale=20.0), tmp_path / "house.glb")

    out = blender_render.render_view(
        model, tmp_path / "aerial.png", azimuth=38.0, elevation=45.0,
        resolution=(320, 240), samples=16,
    )
    assert out.is_file()
    # A blank frame is also a file. Anything real is larger than this.
    assert out.stat().st_size > 5_000
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: FAIL — `No module named 'planto3d.blender_render'`.

- [ ] **Step 3: Write the module**

Create `planto3d/blender_render.py`:

```python
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
    offset = mathutils.Vector(
        (
            distance * math.cos(el) * math.sin(az),
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: PASS. `bpy` is installed in this venv, so the render test runs
rather than skipping — confirm it did by running with `-v` and checking
the test is not marked `s`.

- [ ] **Step 5: Add the optional dependency**

In `pyproject.toml`'s existing `[project.optional-dependencies]` block,
add alongside the `ml` and `dev` extras:

```toml
render = ["bpy>=5.2,<6"]
```

Add a comment above it, in the file's own voice, noting that this pulls
659 MB and that the Blender path is optional for exactly that reason.

- [ ] **Step 6: Confirm the suite still passes without needing Blender**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 850 plus your new tests. Then confirm the module is importable
without `bpy` being importable, which is the property that keeps the
dependency optional:

```
PYTHONPATH=. .venv/Scripts/python.exe -c "
import sys
sys.modules['bpy'] = None          # simulate bpy being absent
import importlib, planto3d.blender_render as br
importlib.reload(br)
print('imports without bpy:', hasattr(br, 'render_view'))
"
```

If that raises, `bpy` is being imported at module level somewhere and must
be moved inside a function.

- [ ] **Step 7: Commit**

```bash
git add planto3d/blender_render.py tests/test_blender_render.py pyproject.toml
git commit -m "Render a measured model with Blender, headless"
```

---

## Task 2: Give each surface a material that means something

The `.glb` carries thirteen named materials — `glass`, `railing`, `stone`, `timber`, `roof`, `wet` and the rest — and glTF import brings the names through intact. Right now they arrive as flat imported colours. This is where the render stops looking like a massing study.

The point is not decoration. Every material here is justified by something the pipeline decided: `glass` is a surface `extrude.py` built as glazing, `railing` one it built as a balustrade. Nothing is invented, which is exactly what separates this from the diffusion pass.

**Files:**
- Modify: `planto3d/blender_render.py`
- Test: `tests/test_blender_render.py`

**Interfaces:**
- Consumes: the material names present on the imported objects.
- Produces: `blender_render.SURFACE_SHADERS: dict[str, dict]` — surface name to the Principled BSDF settings it should take — and `_apply_materials(objects) -> int`, returning how many materials it recognised, so a silent miss is visible.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_blender_render.py`:

```python
def test_every_exported_surface_has_a_shader():
    # These are the material names planto3d/materials.py actually writes
    # into the glb -- confirmed by importing one and reading them back.
    # A surface with no entry falls back to a default and quietly looks
    # like plaster, which on glass or a railing is very obvious.
    exported = {
        "boundary", "coping", "floor", "frame", "glass", "ground",
        "plinth", "railing", "roof", "stone", "timber", "wall", "wet",
    }
    assert exported <= set(blender_render.SURFACE_SHADERS)


def test_glass_is_actually_transmissive():
    # The one surface where a wrong material is unmistakable.
    glass = blender_render.SURFACE_SHADERS["glass"]
    assert glass["transmission"] > 0.5
    assert glass["roughness"] < 0.2


def test_railings_and_frames_read_as_metal():
    for surface in ("railing", "frame"):
        assert blender_render.SURFACE_SHADERS[surface]["metallic"] > 0.5


def test_masonry_is_rough_and_not_metal():
    for surface in ("wall", "stone", "plinth", "coping"):
        shader = blender_render.SURFACE_SHADERS[surface]
        assert shader["roughness"] > 0.5
        assert shader["metallic"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: FAIL — `module 'planto3d.blender_render' has no attribute 'SURFACE_SHADERS'`.

- [ ] **Step 3: Add the shader table and apply it**

Add to `planto3d/blender_render.py`:

```python
# What each exported surface is made of, as Principled BSDF settings.
#
# The names are the ones planto3d/materials.py writes into the glb and
# that glTF import brings back intact -- verified by importing a model
# and reading the material names off it. Every entry here is justified by
# something the pipeline decided from the drawing: "glass" is a surface
# extrude.py built as glazing, "railing" one it built as a balustrade.
# Nothing is invented, which is the whole difference between this and the
# diffusion pass.
#
# Colour is deliberately absent: the glb already carries the palette the
# user chose through design.py, and overriding it here would silently
# discard that choice. Only the physical character is set.
SURFACE_SHADERS: dict[str, dict] = {
    "glass":    {"metallic": 0.0, "roughness": 0.05, "transmission": 0.95},
    "railing":  {"metallic": 0.9, "roughness": 0.35, "transmission": 0.0},
    "frame":    {"metallic": 0.8, "roughness": 0.40, "transmission": 0.0},
    "wall":     {"metallic": 0.0, "roughness": 0.85, "transmission": 0.0},
    "stone":    {"metallic": 0.0, "roughness": 0.90, "transmission": 0.0},
    "plinth":   {"metallic": 0.0, "roughness": 0.90, "transmission": 0.0},
    "coping":   {"metallic": 0.0, "roughness": 0.80, "transmission": 0.0},
    "boundary": {"metallic": 0.0, "roughness": 0.90, "transmission": 0.0},
    "roof":     {"metallic": 0.0, "roughness": 0.75, "transmission": 0.0},
    # Polished indoors, so it catches the light a wall does not.
    "floor":    {"metallic": 0.0, "roughness": 0.35, "transmission": 0.0},
    # Tiled wet areas -- glossier again than a room floor.
    "wet":      {"metallic": 0.0, "roughness": 0.20, "transmission": 0.0},
    "timber":   {"metallic": 0.0, "roughness": 0.55, "transmission": 0.0},
    "ground":   {"metallic": 0.0, "roughness": 0.95, "transmission": 0.0},
}

# Anything the table does not name. Plaster-like, and deliberately dull
# so an unmapped surface looks wrong rather than plausible.
DEFAULT_SHADER = {"metallic": 0.0, "roughness": 0.80, "transmission": 0.0}

# Our key -> the socket Blender actually calls it. Blender 5.2 renamed
# "Transmission" to "Transmission Weight", and the obvious shortcut of
# title-casing our own key silently misses it: inputs.get() returns None,
# the setting is skipped, and glass renders as opaque plaster with no
# error anywhere. Indexing by an explicit name raises a KeyError instead,
# which is the failure we want on a version that renames a socket again.
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
        recognised += settings is not None
        settings = settings or DEFAULT_SHADER

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
```

Call it from `render_view`, immediately after the meshes are collected
and before the camera is placed:

```python
    _apply_materials(meshes)
```

**Socket names were the risk here, and it was real — I probed this build
before writing the plan.** Blender 5.2 has no `Transmission` socket; it is
`Transmission Weight`. Title-casing our own key would have missed it,
`inputs.get()` would have returned `None`, and glass would have rendered
as opaque plaster with no error. Hence the explicit `SOCKET_NAMES` map and
indexing by `[]` rather than `.get()` — a future rename then raises rather
than silently doing nothing.

Confirm the names on whatever build you are running before committing:

```
PYTHONPATH=. .venv/Scripts/python.exe -c "
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
m = bpy.data.materials.new('probe'); m.use_nodes = True
p = m.node_tree.nodes.get('Principled BSDF')
print([s.name for s in p.inputs])
"
```

Expected on Blender 5.2: `Metallic` and `Roughness` present,
`Transmission` **absent**, `Transmission Weight` present. If your build
differs, correct `SOCKET_NAMES` and say so in the comment — never fall
back to a lookup that can silently skip.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: PASS.

- [ ] **Step 5: Prove the materials reach the image**

Render the same view before and after the material pass and confirm the
image changed. Build a model first:

```
PYTHONPATH=. .venv/Scripts/python.exe -c "
import warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from pathlib import Path
from planto3d.pipeline import run
from planto3d.segment import load_segmenter
r = run(Path('data/bridge/11001.gif'), Path('.blender_check'),
        segmenter=load_segmenter(Path('models/unet_cubicasa.pt')))
print(r.model_path)
"
PYTHONPATH=. .venv/Scripts/python.exe -c "
from pathlib import Path
from planto3d import blender_render as br
out = br.render_view(Path('.blender_check/house.glb'), Path('.blender_check/aerial.png'),
                     azimuth=38.0, elevation=45.0, resolution=(800, 600))
print('wrote', out, out.stat().st_size, 'bytes')
"
```

Report how many materials were recognised — the log line says so — and
paste it. It must be 13 of 13. Anything less means a name in
`materials.py` has no entry here.

- [ ] **Step 6: Commit**

```bash
git add planto3d/blender_render.py tests/test_blender_render.py
git commit -m "Give each exported surface a material that means something"
```

---

## Task 3: The six standard views, from the command line

`preview.render_views` renders top, front, back, left, right and aerial,
and every caller in the project expects that shape. Mirroring it makes the
Blender path a drop-in alternative that can be compared frame for frame
against the fast one.

**Files:**
- Modify: `planto3d/blender_render.py`
- Create: `scripts/render_blender.py`
- Test: `tests/test_blender_render.py`

**Interfaces:**
- Consumes: `planto3d.preview.VIEWS` — `{"top": (0.0, 90.0), "front": (0.0, 0.0), "back": (180.0, 0.0), "left": (270.0, 0.0), "right": (90.0, 0.0), "aerial": (38.0, 45.0)}`, defined at `preview.py:628`.
- Produces: `blender_render.render_views(model_path, output_dir, views=None, resolution=(1200, 900), prefix="blender", samples=DEFAULT_SAMPLES) -> dict[str, Path]` — deliberately the same signature shape as `preview.render_views`, minus its `lighting` parameter, which belongs to the rasterizer's own light model.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_blender_render.py`:

```python
def test_the_view_set_matches_the_rasterizer():
    # The two renderers must answer to the same view names, or comparing
    # them frame for frame means renaming files by hand.
    from planto3d.preview import VIEWS

    assert set(blender_render.STANDARD_VIEWS) == set(VIEWS)
    for name, angles in VIEWS.items():
        assert blender_render.STANDARD_VIEWS[name] == angles
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: FAIL — no attribute `STANDARD_VIEWS`.

- [ ] **Step 3: Add the view set and the loop**

In `planto3d/blender_render.py`:

```python
from planto3d.preview import VIEWS as STANDARD_VIEWS  # noqa: F401
```

Import it rather than restating the angles: two copies of the same six
numbers will drift, and the whole point is that the two renderers can be
pointed at identical angles.

Then:

```python
def render_views(
    model_path: Path,
    output_dir: Path,
    views: dict[str, tuple[float, float]] | None = None,
    resolution: tuple[int, int] = (1200, 900),
    prefix: str = "blender",
    samples: int = DEFAULT_SAMPLES,
) -> dict[str, Path]:
    """Render every standard view, the same six ``preview.py`` produces.

    Named to mirror ``preview.render_views`` so the two can be swapped
    and compared frame for frame. The prefix differs by default so both
    sets can sit in one directory without overwriting each other.
    """
    output_dir = Path(output_dir)
    rendered = {}
    for name, (azimuth, elevation) in (views or STANDARD_VIEWS).items():
        rendered[name] = render_view(
            Path(model_path),
            output_dir / f"{prefix}-{name}.png",
            azimuth=azimuth,
            elevation=elevation,
            resolution=resolution,
            samples=samples,
        )
    return rendered
```

Note this re-imports the `.glb` per view, where `preview.render_views`
loads once and reuses. That is deliberate for now: `read_factory_settings`
is what keeps each frame from inheriting the last one's camera and lights,
and at roughly 8 seconds a frame the import is not the cost. Say so in the
docstring so the next reader does not "fix" it.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: PASS.

- [ ] **Step 5: Write the CLI**

Create `scripts/render_blender.py`, modelled on `scripts/run_pipeline.py`
— read that file first and follow its argparse shape and its `main()`
structure:

```python
"""Render a built model with Blender, deterministically.

The pipeline's own rasterizer (planto3d/preview.py) is fast and plain.
This is the slow, pretty one: real materials, soft shadows and global
illumination, with no generative step -- so unlike the diffusion pass,
every surface in the image is one the drawing supports.

Needs the render extra:  pip install -e ".[render]"   (659 MB)

    python scripts/render_blender.py house.glb output
"""

import argparse
import logging
from pathlib import Path

from planto3d import blender_render


def main(model: str, output_dir: str, resolution: int, samples: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not blender_render.available():
        raise SystemExit(
            'Blender is not installed. Run: pip install -e ".[render]"  '
            "(659 MB), or use scripts/run_pipeline.py for the fast renderer."
        )

    height = int(resolution * 3 / 4)
    rendered = blender_render.render_views(
        Path(model), Path(output_dir),
        resolution=(resolution, height), samples=samples,
    )
    for name, path in sorted(rendered.items()):
        print(f"{name:8} {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="a .glb written by the pipeline")
    parser.add_argument("output_dir")
    parser.add_argument("--resolution", type=int, default=1200, help="width in pixels")
    parser.add_argument(
        "--samples", type=int, default=blender_render.DEFAULT_SAMPLES,
        help="Cycles samples; 32 is the measured floor, higher costs little",
    )
    arguments = parser.parse_args()
    main(arguments.model, arguments.output_dir, arguments.resolution, arguments.samples)
```

- [ ] **Step 6: Render all six and time it**

```
PYTHONPATH=. .venv/Scripts/python.exe scripts/render_blender.py .blender_check/house.glb .blender_check/views --resolution 800
```

Expected: six PNGs. Time the whole run and report it — at roughly 8 s a
frame it should land near a minute. Paste the output and the timing.

- [ ] **Step 7: Full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add planto3d/blender_render.py scripts/render_blender.py tests/test_blender_render.py
git commit -m "Render the six standard views through Blender"
```

---

## Task 4: Prove it is deterministic, and record what it costs

The whole claim of this workstream is that the image is evidence. An
image that differs run to run is not evidence, and one nobody has timed
cannot be planned around.

**Files:**
- Modify: `docs/AUDIT.md`
- Modify: `README.md`

**Interfaces:** none. Measurement and documentation.

- [ ] **Step 1: Measure determinism**

Render the same view twice into different files and compare the bytes:

```
PYTHONPATH=. .venv/Scripts/python.exe -c "
import hashlib
from pathlib import Path
from planto3d import blender_render as br
out = Path('.blender_check')
a = br.render_view(out/'house.glb', out/'det-a.png', 38.0, 45.0, (400,300), samples=16)
b = br.render_view(out/'house.glb', out/'det-b.png', 38.0, 45.0, (400,300), samples=16)
ha, hb = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b))
print('a', ha[:16]); print('b', hb[:16]); print('IDENTICAL:', ha == hb)
"
```

Report the answer honestly either way. Cycles is a sampled path tracer
and may or may not be bit-identical across runs depending on how it seeds
and threads. **If it is not identical, that is a finding, not a failure**
— say so, and say whether the difference is visible or only in the last
bits. Do not chase bit-identity by lowering the sample count until it
matches.

- [ ] **Step 2: Measure the real cost**

Time the six-view run at the default 1200 px and at 800 px, and record
both. Use the figures your own run produces, not the ones in this plan.

- [ ] **Step 3: Record it in `docs/AUDIT.md`**

Add a section in the document's own voice covering:

- that a deterministic renderer now exists, what it is for, and that it is
  the answer to the diffusion pass being an impression rather than a
  measurement
- **Cycles against EEVEE: 7.7 s versus 57.3 s** for a 640×480 frame,
  measured twice in both orders, and why — EEVEE needs a GL context and
  falls back to software headless, Cycles is a CPU path tracer
- that doubling samples 32 → 64 costs 1.8 s, so quality is nearly free
- the six-view timings from Step 2
- the determinism result from Step 1, whatever it turned out to be
- that `bpy` is 659 MB and therefore an optional extra
- that all 13 exported materials map to a physical surface, and that
  colour is left to the palette the user chose rather than overridden

- [ ] **Step 4: Document it in `README.md`**

Add a short subsection near the existing usage examples, in the same
voice, covering the `render` extra, the command, and one sentence on when
to reach for it rather than the fast renderer.

- [ ] **Step 5: Commit**

```bash
git add docs/AUDIT.md README.md
git commit -m "docs: what the Blender path costs and whether it repeats"
```

---

---

## Task 5: Put the building somewhere, and light it properly

*Added after Task 2, on looking at an actual render rather than a diff. Task 1's
`_add_lighting` is deliberately minimal — one sun and a flat ambient world — and
its own comment defers the real work to "the next task". Task 2 was materials,
Task 3 is views and the CLI, Task 4 is documentation: **no task covered the
environment.** The result is a correctly-shaded building floating in a grey void,
which does not meet the spec's "looks like an architectural visualisation".*

*Execute this before Task 3.*

Three things are missing and each is visible in the current output: there is no
sky, there is no ground for the building to stand on or cast a shadow onto, and
the light rig is a single untinted sun.

The third matters most for correctness, not just looks. `planto3d/style.py`
already defines a `Lighting` dataclass — sun, sky, bounce and fill colours with
matching strengths — and `LIGHTING_PRESETS` maps the user's time-of-day choice
onto it: `midday`, `golden hour`, `dusk`. `preview.py` honours all of that. The
Blender path currently ignores it, so choosing "sunset" changes the rasterizer's
output and not this one — the same class of disagreement the photoreal prompt
had before it was given the `Design`.

**Files:**
- Modify: `planto3d/blender_render.py`
- Test: `tests/test_blender_render.py`

**Interfaces:**
- Consumes: `planto3d.style.Lighting` (fields `sun`, `sky`, `bounce`, `fill` as 0-255 RGB triples; `ambient_strength`, `key_strength`, `fill_strength`, `exposure`) and `planto3d.style.LIGHTING_PRESETS`.
- Produces:
  - `render_view(..., lighting: Lighting | None = None)` — keyword-optional and last, defaulting to `Lighting()`, so every existing caller keeps working.
  - `_add_world(scene, lighting)` — a sky gradient rather than flat grey.
  - `_add_ground(centre, radius, lighting)` — a plane for the building to stand on.
  - `_add_lighting(scene, centre, radius, lighting)` — the existing function, extended to take the rig from `Lighting` rather than hardcoding one sun.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_blender_render.py`:

```python
def test_lighting_is_optional_and_defaults_to_the_shared_preset():
    # Every existing caller passes no lighting and must keep working.
    import inspect

    from planto3d.style import Lighting

    signature = inspect.signature(blender_render.render_view)
    parameter = signature.parameters["lighting"]
    assert parameter.default is None
    # Keyword-optional and last, so positional callers are unaffected.
    assert list(signature.parameters)[-1] == "lighting"


@pytest.mark.skipif(not blender_render.available(), reason="bpy not installed")
def test_the_time_of_day_changes_the_image():
    # style.py already maps the user's choice onto a Lighting preset and
    # preview.py honours it. If this renderer ignores it, the two outputs
    # disagree about what hour it is -- the same defect the photoreal
    # prompt had before it was given the design.
    import tempfile
    from pathlib import Path

    from planto3d.style import LIGHTING_PRESETS

    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir)
        model = _tiny_model(out)
        rendered = {}
        for name in ("midday", "dusk"):
            path = blender_render.render_view(
                model, out / f"{name}.png", azimuth=38.0, elevation=45.0,
                resolution=(160, 120), samples=16,
                lighting=LIGHTING_PRESETS[name],
            )
            rendered[name] = path.read_bytes()
        assert rendered["midday"] != rendered["dusk"]
```

`_tiny_model` is a helper — factor one out of the existing render test rather
than duplicating the model-building code, and use it in both.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: FAIL — `render_view` has no `lighting` parameter.

- [ ] **Step 3: Build the world and the ground**

In `planto3d/blender_render.py`:

```python
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

    The sky colour is the one style.py already chose for this hour, so the
    Blender render and the rasterizer agree about what time it is.
    """
    import bpy

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    tree = world.node_tree
    background = tree.nodes["Background"]
    background.inputs[0].default_value = _to_linear(lighting.sky)
    background.inputs[1].default_value = lighting.ambient_strength
    scene.world = world


def _add_ground(centre, radius, lighting) -> None:
    """A plane for the building to stand on and cast a shadow onto.

    Without it the model floats: Cycles has nothing to catch the contact
    shadow, which is most of what makes a render read as photographed
    rather than assembled.
    """
    import bpy
    import mathutils

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
```

- [ ] **Step 4: Rebuild the light rig from `Lighting`**

Replace `_add_lighting`'s body. It currently adds one untinted sun at fixed
energy; it must now take its colours and strengths from the `Lighting` it is
given:

```python
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
```

with the two energies as named constants carrying their reasoning:

```python
# style.py's strengths are multipliers tuned for its own rasterizer, not
# watts. These convert them into something Cycles can use. Both were set by
# rendering and looking: below these the model goes muddy, above them the
# filmic curve rolls the lit faces into white.
KEY_ENERGY = 4.0
FILL_ENERGY = 60.0

# The sun's angular diameter in radians, which is what softens the shadow
# edge. The real sun subtends about 0.53 degrees; this is deliberately
# wider, because a hard-edged shadow is the most obvious tell of a
# synthetic render and a little softness reads better at these sizes.
SUN_ANGLE = 0.06
```

Wire `_add_world` and `_add_ground` into `render_view` alongside the existing
`_add_lighting` call, and thread `lighting` through — `lighting = lighting or Lighting()`
at the top of `render_view`, importing `Lighting` inside the function so `bpy`
and `style` stay out of module scope in the same way.

Also apply the exposure `style.py` already chose, so the two renderers agree
about brightness as well as colour:

```python
    scene.view_settings.exposure = math.log2(max(lighting.exposure, 1e-3))
```

Blender's exposure is in stops while `style.py`'s is a linear multiplier, which
is why it is a log rather than a direct assignment. Say so in a comment.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blender_render.py -q`
Expected: PASS, including the new time-of-day test.

- [ ] **Step 6: Look at it**

This task exists because the render looked wrong, so the check is to look again.
Build a real plan and render the same aerial view at two hours:

```
PYTHONPATH=. .venv/Scripts/python.exe -c "
import warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from pathlib import Path
from planto3d.pipeline import run
from planto3d.segment import load_segmenter
from planto3d import blender_render as br
from planto3d.style import LIGHTING_PRESETS
r = run(Path('data/bridge/11001.gif'), Path('.env_check'),
        segmenter=load_segmenter(Path('models/unet_cubicasa.pt')))
for hour in ('midday', 'golden hour', 'dusk'):
    br.render_view(r.model_path, Path(f'.env_check/{hour}.png'),
                   azimuth=38.0, elevation=45.0, resolution=(900, 675),
                   samples=96, lighting=LIGHTING_PRESETS[hour])
print('rendered')
"
```

Report what changed: the building should now sit on ground, cast a contact
shadow, and stand against a sky whose colour differs between the three hours.
State plainly whether it does. If it does not, say so rather than reporting the
step as done — this is the one task whose success is visual.

- [ ] **Step 7: Full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add planto3d/blender_render.py tests/test_blender_render.py
git commit -m "Give the render a sky, a ground, and the hour the user chose"
```

## Self-Review

**Spec coverage.** The spec's workstream 2 asks for `planto3d/blender_render.py` driven headlessly by `bpy`, fed from `extrude.py`'s geometry and `materials.py`'s surfaces, with no generative step, and says the existing rasterizer stays untouched. Task 1 builds the path, Task 2 the materials, Task 3 the view set and CLI, Task 4 the evidence. `preview.py` is modified by no task, and the Global Constraints forbid it.

The spec names the schedule risk as `bpy` being awkward to install headless. That was settled before this plan was written — it installs on Python 3.13 and renders without a display or GPU — so the fallback the spec offers (polishing `preview.py` instead) is not needed and no task depends on it.

**Placeholder scan.** No TBD or TODO. Task 2's socket-name probe is a real command with real expected output, not a "check the API" instruction. Task 4's determinism step deliberately does not assert the answer, because the answer is not known in advance — that is a measurement, not a placeholder.

**Type consistency.** `render_view(model_path, output_path, azimuth, elevation, resolution, samples)` is defined in Task 1 and called by Task 3's `render_views` with matching keywords. `SURFACE_SHADERS` and `_apply_materials` are defined in Task 2 and used only there. `STANDARD_VIEWS` is imported from `preview.VIEWS` in Task 3 and asserted equal to it in the same task. `DEFAULT_SAMPLES` is defined in Task 1 and referenced in Tasks 1, 3 and the CLI.

**A risk this plan closes deliberately.** Blender renames Principled BSDF
sockets between versions, and this build proves it: there is no
`Transmission`, only `Transmission Weight`. The plan therefore indexes
sockets by an explicit name map and with `[]` rather than `.get()`, so a
future rename raises a `KeyError` on the first render instead of quietly
producing opaque glass. That converts the failure mode from silent to
loud, which is the most that can be done without pinning the Blender
version outright.
