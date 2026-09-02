# Scale Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the median scale error from 17.7% toward the spec's 10% gate, without fitting a constant to the one corpus that has ground truth.

**Architecture:** Three independent routes to a better number, in descending order of how much they can be trusted. An exact user-supplied measurement (cannot be wrong, works on any plan). Per-convention element constants chosen by a signal already in the drawing (shifts the centre where the tradition is detectable). And a measurement script that attributes error to its source, so every later change is provable rather than asserted.

**Tech Stack:** Python 3.11+, `.venv`, pytest, existing `planto3d.calibrate` / `planto3d.pipeline`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-02-plan-accurate-3d-design.md` — this plan implements its workstream 1.

## Global Constraints

- `pytest -q` must pass. It is at **835**; each task adds tests. Never a net loss.
- Python is `.venv/Scripts/python.exe`; set `PYTHONPATH=.` when running scripts directly.
- No number goes in a doc or commit message unless a script produced it in that session.
- **Do not change `TYPICAL_WALL_FT` or `TYPICAL_DOOR_FT` globally.** The standing project rule is that the wall-thickness constant must not be fitted to one dataset, and CubiCasa is the only corpus with metric ground truth. Per-convention values selected by a detected signal are permitted; a new global default is not.
- Match existing code style: docstrings and comments explain *why*; every named constant carries a comment giving the reasoning behind its value. `MAX_SCALE_DISAGREEMENT` in `planto3d/pipeline.py` is the house standard.
- Commit messages: short, declarative, describing the *effect*. No conventional-commit prefixes beyond an occasional `docs:`.
- Commit with explicit paths only. Never `git add -A` or `git add .` — `demo_plans/` is intentionally untracked.

---

## What was measured before writing this

All produced this session, over the 30 ground-truthed CubiCasa sheets at
`C:/Users/Rahul Soni/AppData/Local/Temp/claude/cubicasa_batch`.

| Finding | Measurement |
| --- | --- |
| Error as shipped | **17.7% median**, 20/30 within a fifth |
| Which source actually fires | **15 walls, 15 doors** — an even split |
| Printed dimensions on CubiCasa | **never fire** — these rasters carry almost no text |
| Changing the wall constant alone (0.75 → 0.633) | **9.9% median, 23/30** — the best of four combinations |
| Changing the door constant alone (2.5 → 2.25) | **20.1% median, 15/30** — *worse* than shipped |
| Changing both | 12.9% median, 18/30 |
| Real wall thickness implied by ground truth | min 0.478, p25 0.598, **median 0.648**, p75 0.803, max 1.176 ft |

Three things follow, and they shape every task below.

**The door assumption is not the problem.** The spec predicted a gain from
correcting it to 2'3"; measured, that change makes the result worse. Only
the wall constant is mis-centred — 0.75 shipped against 0.648 implied.

**A constant has a floor.** The interquartile spread of real wall thickness
is 0.205 ft on a median of 0.648 — about ±16%. Real buildings differ. No
choice of constant removes that, so constant-tuning alone cannot reach the
gate reliably and the exact routes matter more.

**Half the population uses each source**, which is why a wall-only change
helped so much: it fixed 15 sheets and left the other 15 untouched.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `scripts/scale_accuracy.py` (modify) | Break the error down by scale source, so a change can be attributed rather than asserted. |
| `planto3d/calibrate.py` (modify) | `scale_from_known_room` — an exact scale from a room whose real size the user states. Per-convention element constants and the detector that picks them. |
| `planto3d/pipeline.py` (modify) | Thread a caller-supplied scale override through `extract`, ahead of every inferred source. |
| `scripts/correct_and_build.py` (modify) | `--scale-room FLOOR:ROOM=WxH` so the override is reachable from the command line. |
| `tests/test_scale_override.py` (create) | The exact route: parsing, arithmetic, precedence over inferred sources. |
| `tests/test_conventions.py` (create) | The detector: which tradition a drawing belongs to, and refusing to guess. |

---

## Task 1: Report scale error by source

Nothing later in this plan can be judged without this. The population is an
even 15/15 split between walls and doors, and the two move in opposite
directions — a change that helps one and hurts the other shows up as
almost nothing in a single median. Every subsequent task is measured with
this script.

**Files:**
- Modify: `scripts/scale_accuracy.py`

**Interfaces:**
- Consumes: `planto3d.pipeline.run`, `planto3d.cubicasa.ground_truth_scale`, and the module's own `true_scale(svg_path, image_path) -> float | None` at line 40.
- Produces: no importable API. A printed breakdown, one row per `scale_source`, each with count, median absolute error, and the count within 20%.

- [ ] **Step 1: Read the script as it stands**

Read `scripts/scale_accuracy.py` in full. It already collects a `Counter`
of sources and a list of errors; this task pairs them so the error can be
grouped rather than pooled. Keep the existing overall line — it is the
number quoted in `docs/AUDIT.md` and must stay comparable.

- [ ] **Step 2: Group the errors by source**

Where the script currently appends to a flat list of errors, append to a
`dict[str, list[float]]` keyed by `result.scale_source`, keeping the flat
list as well so the overall median is unchanged.

After the existing overall report, add:

```python
    # Sources are reported separately because they fail in opposite
    # directions and a pooled median hides it: measured over 30 sheets the
    # split is an even 15 walls to 15 doors, and correcting the wall
    # constant alone moved the pooled figure from 17.7% to 9.9% while
    # correcting the door constant alone moved it to 20.1%. A change that
    # helps one source and harms the other nets out to nothing here.
    print()
    print(f"{'source':12}{'plans':>7}{'median err':>12}{'within 20%':>12}")
    for source in sorted(by_source):
        errors = by_source[source]
        within = sum(1 for error in errors if error <= TOLERANCE)
        print(
            f"{source:12}{len(errors):>7}"
            f"{statistics.median(errors):>11.1%}"
            f"{within:>8}/{len(errors)}"
        )
```

Use the script's existing tolerance constant for `TOLERANCE` — read the
file and use whatever it already calls the 20% threshold rather than
introducing a second name for the same number.

- [ ] **Step 3: Run it and record the baseline**

```
PYTHONPATH=. .venv/Scripts/python.exe scripts/scale_accuracy.py "C:/Users/Rahul Soni/AppData/Local/Temp/claude/cubicasa_batch" --checkpoint models/unet_cubicasa.pt --limit 30
```

Expected: the overall figure stays **17.7% median, 20/30 within a fifth**
— unchanged, because this task only regroups what was already computed.
The new table should show roughly 15 plans under `walls` and 15 under
`doors`. Paste the whole output into your report; it is the baseline every
later task is compared against.

If the overall figure moved, you changed the computation rather than the
reporting. Stop and report that.

- [ ] **Step 4: Commit**

```bash
git add scripts/scale_accuracy.py
git commit -m "Report scale error per source, not just pooled"
```

---

## Task 2: An exact scale from a room the user measures

The only route to scale that cannot be wrong, and the only one that works
on a drawing from a tradition nobody has tested. A user reads one room's
real size off their own plan — or off the building — and everything else
follows. It needs no detection, no corpus, and no assumption about doors
or walls.

It also sidesteps the constant's floor entirely: the ±16% spread of real
wall thicknesses does not apply when the size is stated rather than
inferred.

**Files:**
- Modify: `planto3d/calibrate.py`
- Modify: `planto3d/pipeline.py`
- Modify: `scripts/correct_and_build.py`
- Test: `tests/test_scale_override.py` (create)

**Interfaces:**
- Consumes: `planto3d.geometry_types.Room` and its `polygon`; `planto3d.calibrate._polygon_area_px(room)` which already exists at line 184.
- Produces:
  - `calibrate.scale_from_known_room(room: Room, width_ft: float, height_ft: float) -> float | None` — pixels per foot from a room whose real size is known, or `None` when the room's pixel area is degenerate.
  - `pipeline.extract(..., scale_override: float | None = None)` and the same parameter on `run`, taking precedence over every inferred source and setting `scale_source` to `"stated"`.
  - CLI flag `--scale-room FLOOR:ROOM=WxH` on `scripts/correct_and_build.py`, where W and H are feet.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scale_override.py`:

```python
"""A scale the user states outright, rather than one inferred.

Every inferred route rests on an assumption about a standard element --
a 2'6" door, a 9" wall -- and those assumptions are what the residual
error is made of: measured over 30 ground-truthed sheets, the real wall
thickness runs from 0.478 to 1.176 ft around a median of 0.648, so no
constant fits every building. A stated measurement has no such floor.
"""

import pytest

from planto3d.calibrate import scale_from_known_room
from planto3d.geometry_types import Room


def _room(width_px: float, height_px: float) -> Room:
    return Room(
        polygon=[(0.0, 0.0), (width_px, 0.0), (width_px, height_px), (0.0, height_px)]
    )


def test_a_square_room_gives_its_own_scale():
    # 200 px across a room the user says is 10 ft: 20 px per foot.
    assert scale_from_known_room(_room(200.0, 200.0), 10.0, 10.0) == pytest.approx(20.0)


def test_scale_comes_from_area_not_one_edge():
    # 200x100 px stated as 10x5 ft is 20 px/ft either way; taking one edge
    # would also give 20 here, so use a case where the room is drawn out of
    # proportion to what was stated and the area still settles it.
    # 400x100 px (40000) stated as 10x10 ft (100) -> sqrt(400) = 20.
    assert scale_from_known_room(_room(400.0, 100.0), 10.0, 10.0) == pytest.approx(20.0)


def test_a_degenerate_room_yields_nothing():
    # A zero-area polygon cannot state a scale, and returning a huge or
    # zero number here would resize the whole building.
    assert scale_from_known_room(_room(0.0, 0.0), 10.0, 10.0) is None


def test_a_zero_size_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        scale_from_known_room(_room(200.0, 200.0), 0.0, 10.0)


def test_a_negative_size_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        scale_from_known_room(_room(200.0, 200.0), 10.0, -4.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scale_override.py -q`
Expected: FAIL — `cannot import name 'scale_from_known_room'`.

- [ ] **Step 3: Implement it in `planto3d/calibrate.py`**

Add beside the other `scale_from_*` functions:

```python
def scale_from_known_room(room, width_ft: float, height_ft: float) -> float | None:
    """Pixels per foot, from a room whose real size the user states.

    The only route to scale with no assumption in it. Every other route
    rests on a standard element -- a 2'6" door, a 9" wall -- and the
    residual error is largely those standards not holding: measured
    against ground truth over 30 sheets, real wall thickness runs 0.478
    to 1.176 ft around a median of 0.648, so no constant fits every
    building. A stated size has no such spread.

    Taken from area rather than an edge because a room is rarely drawn
    as the clean rectangle its printed size implies -- a bay, a wardrobe
    recess or a chamfered corner all make one edge disagree with the
    stated width while the area stays close.
    """
    if width_ft <= 0 or height_ft <= 0:
        raise ValueError(
            f"a room's stated size must be positive, got {width_ft} x {height_ft} ft"
        )

    area_px = abs(_polygon_area_px(room))
    if area_px <= 0:
        return None
    return math.sqrt(area_px / (width_ft * height_ft))
```

Check `math` is already imported at the top of `calibrate.py`; add it if not.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scale_override.py -q`
Expected: PASS.

- [ ] **Step 5: Thread the override through the pipeline**

In `planto3d/pipeline.py`, add `scale_override: float | None = None` as the
last parameter of both `extract` and `run`, documenting it in each
docstring as: "``scale_override`` is pixels per foot stated by the caller.
It takes precedence over every inferred source, because a stated size has
no assumption in it to be wrong."

In `extract`, immediately after the block that settles `scale` and
`scale_source` from the printed and reference estimates — that is, after
the `if scale is None:` fallback to `assumed_scale` — add:

```python
    # Ahead of everything inferred. A caller who has measured a room knows
    # something no estimate can recover.
    if scale_override is not None:
        if scale_override <= 0:
            raise ValueError(
                f"scale_override must be positive, got {scale_override}"
            )
        scale, scale_source = scale_override, "stated"
        logger.info("scale %.2f px/ft stated by the caller", scale)
```

Forward it in `run`: `extract(source, output_dir, segmenter, crop, split=split, scale_override=scale_override)`.

`"stated"` must also be treated as a measured rather than assumed source.
Find `PRINTED_SCALE_SOURCES` in `pipeline.py` and add `"stated"` to it, so
`scale_assumed` reports `False` and the summary does not warn that the
size is inferred when the user supplied it.

- [ ] **Step 6: Add the CLI flag**

In `scripts/correct_and_build.py`, add beside the other flags:

```python
    parser.add_argument(
        "--scale-room",
        metavar="FLOOR:ROOM=WxH",
        help="set the scale from one room's real size in feet, e.g. 1:5=12x10",
    )
```

Add a parser beside `parse_correction`, reusing its FLOOR:ROOM convention
so a user learns one syntax rather than two:

```python
def parse_scale_room(text: str) -> tuple[tuple[int, int], float, float]:
    """Turn ``FLOOR:ROOM=WxH`` into the room to measure and its real size."""
    try:
        where, size = text.split("=", 1)
        floor_text, room_text = where.split(":", 1)
        width_text, height_text = size.lower().split("x", 1)
        floor, room = int(floor_text), int(room_text)
        width, height = float(width_text), float(height_text)
    except ValueError:
        raise ValueError(
            f"could not read {text!r}; expected FLOOR:ROOM=WxH in feet, "
            "for example 1:5=12x10"
        ) from None

    if floor < 1:
        raise ValueError(f"floor {floor} is not valid; floors are numbered from 1")
    if room < 0:
        raise ValueError(f"room {room} is not valid; rooms are indexed from 0")
    return (floor - 1, room), width, height
```

The flag needs the extracted rooms before it can measure one, so it cannot
be passed to `extract` directly. In `main`, after `extract` returns and
after `show(result)`, when `--scale-room` was given: look up the room,
compute the scale with `scale_from_known_room`, and re-run `extract` with
`scale_override` set. Re-running is the honest way to do this — the scale
gates room filtering (`MIN_ROOM_SQFT`) and the labelled-region placement
inside `extract`, so patching `result.scale` afterwards would leave those
stages computed against the old number.

Print what happened, since re-reading the sheet is visible in the log:

```python
        print(
            f"\nroom {floor_number}:{room_index} stated as {width}x{height} ft "
            f"-> {stated:.2f} px/ft (was {result.scale:.2f}); re-reading the sheet"
        )
```

Document the flag in the module docstring beside the existing examples.

- [ ] **Step 7: Prove it end to end**

`data/bridge/11001.gif` builds in under a minute and its scale is inferred
from doors, so the override has something to replace.

```
PYTHONPATH=. .venv/Scripts/python.exe scripts/correct_and_build.py data/bridge/11001.gif .scale_check --checkpoint models/unet_cubicasa.pt --list
```

Pick a room from the listing, then:

```
PYTHONPATH=. .venv/Scripts/python.exe scripts/correct_and_build.py data/bridge/11001.gif .scale_check --checkpoint models/unet_cubicasa.pt --scale-room 1:0=12x10
```

Expected: the run reports the stated scale replacing the inferred one, the
summary no longer warns that the size is inferred, and a model is built.
Paste both outputs into your report. Remove `.scale_check` afterwards.

- [ ] **Step 8: Full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add planto3d/calibrate.py planto3d/pipeline.py scripts/correct_and_build.py tests/test_scale_override.py
git commit -m "Let a measured room settle the scale, ahead of every guess"
```

---

## Task 3: Element sizes by drafting tradition

The wall constant is mis-centred — 0.75 ft shipped against 0.648 implied
by ground truth — and correcting it alone takes the pooled error from
17.7% to 9.9%. But changing the global default would fit it to the one
corpus that can be measured, which the project's standing rule forbids and
which would silently degrade the Indian and Spanish plans that cannot be
checked.

The way through is to change it only where the drawing says which
tradition it belongs to. CubiCasa is Finnish residential, and
`planto3d/features.py` already recognises Finnish and Swedish room names —
`PARVEKE`, `KYLPYHUONE`, `KEITTIO`, `BALKONG`. That is a signal already on
the page, already parsed, and specific to exactly the population whose
constant we can justify changing.

Everything undetected keeps today's values unchanged.

**Files:**
- Modify: `planto3d/calibrate.py`
- Modify: `planto3d/pipeline.py`
- Test: `tests/test_conventions.py` (create)

**Interfaces:**
- Consumes: `planto3d.calibrate.TextBox` and its `.text`; `planto3d.calibrate.TYPICAL_DOOR_FT` (2.5) and `TYPICAL_WALL_FT` (0.75) as the defaults.
- Produces:
  - `calibrate.CONVENTIONS: dict[str, tuple[float, float]]` — convention name to `(door_ft, wall_ft)`.
  - `calibrate.detect_convention(text_boxes: list[TextBox]) -> str | None` — the tradition a drawing belongs to, or `None` when nothing says.
  - `calibrate.element_sizes(text_boxes) -> tuple[float, float]` — the `(door_ft, wall_ft)` to use, falling back to the shipped defaults.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_conventions.py`:

```python
"""Which drafting tradition a drawing belongs to, where it says so.

Scale rests on assumed element sizes, and those differ by tradition.
Measured against CubiCasa's own annotations over 30 sheets, the implied
real wall thickness has a median of 0.648 ft against the 0.75 the code
assumes -- and correcting only that takes the pooled error from 17.7% to
9.9%. Correcting the door constant instead makes it worse, 20.1%.

So the wall value is changed, the door value is not, and both only where
the drawing identifies its tradition. Everything unrecognised keeps the
shipped defaults, because a constant fitted to the one corpus with
ground truth would silently degrade every corpus without it.
"""

import pytest

from planto3d.calibrate import (
    TYPICAL_DOOR_FT,
    TYPICAL_WALL_FT,
    detect_convention,
    element_sizes,
)


class _Box:
    """The only part of a TextBox this reads."""

    def __init__(self, text: str) -> None:
        self.text = text


def _boxes(*words: str) -> list:
    return [_Box(word) for word in words]


@pytest.mark.parametrize(
    "label", ["PARVEKE", "KYLPYHUONE", "KEITTIO", "OLOHUONE", "BALKONG"]
)
def test_finnish_and_swedish_names_identify_the_nordic_tradition(label):
    assert detect_convention(_boxes(label, "3200")) == "nordic"


def test_english_names_are_not_claimed_for_any_tradition():
    # English is the default the shipped constants already serve; claiming
    # it as a tradition would change behaviour for the plans that work.
    assert detect_convention(_boxes("BEDROOM", "KITCHEN", "BALCONY")) is None


def test_nothing_readable_identifies_nothing():
    # The common case on a scan OCR cannot read. It must not guess.
    assert detect_convention([]) is None
    assert detect_convention(_boxes("", "5 GE", "~ a")) is None


def test_one_stray_word_does_not_decide_a_sheet():
    # A single match is a misread waiting to happen; a tradition is a
    # property of the whole drawing, so it takes corroboration.
    assert detect_convention(_boxes("PARVEKE", "BEDROOM", "KITCHEN")) is None


def test_the_nordic_tradition_uses_its_measured_wall_thickness():
    door_ft, wall_ft = element_sizes(_boxes("PARVEKE", "KEITTIO"))
    assert wall_ft == pytest.approx(0.648)
    # The door constant is deliberately unchanged: correcting it measured
    # worse, 20.1% against 17.7% shipped.
    assert door_ft == pytest.approx(TYPICAL_DOOR_FT)


def test_an_unrecognised_drawing_keeps_the_shipped_defaults():
    door_ft, wall_ft = element_sizes(_boxes("BEDROOM", "KITCHEN"))
    assert (door_ft, wall_ft) == (TYPICAL_DOOR_FT, TYPICAL_WALL_FT)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_conventions.py -q`
Expected: FAIL — `cannot import name 'detect_convention'`.

- [ ] **Step 3: Implement the detector**

Add to `planto3d/calibrate.py`, near the other scale constants:

```python
# Room names that place a drawing in a drafting tradition. Only the Nordic
# set is listed, because it is the only tradition whose element sizes can
# be checked: CubiCasa is Finnish residential and carries metric ground
# truth. These words are already recognised by planto3d/features.py, which
# is where they came from.
CONVENTION_KEYWORDS = {
    "nordic": (
        "PARVEKE",
        "TERASSI",
        "KATTOTERASSI",
        "BALKONG",
        "TERRASS",
        "KYLPYHUONE",
        "PESUHUONE",
        "KODINHOITOHUONE",
        "KEITTIO",
        "OLOHUONE",
        "MAKUUHUONE",
        "ETEINEN",
    ),
}

# How many of those words must appear before a sheet is claimed for a
# tradition. One is a misread waiting to happen -- OCR turns "BEDROOM 2"
# into all sorts of things -- and a tradition is a property of the whole
# drawing rather than of one label.
MIN_CONVENTION_HITS = 2

# Element sizes per tradition, as (door_ft, wall_ft).
#
# Nordic's wall thickness is the measured median implied by CubiCasa's own
# ground truth over 30 sheets -- 0.648 ft against the 0.75 assumed
# worldwide. Substituting it alone moves the pooled error from 17.7% to
# 9.9% and the sheets within a fifth from 20/30 to 23/30.
#
# The door stays at the shipped 2.5 ft deliberately. Correcting it to
# CubiCasa's own 2'3" was measured and made things worse -- 20.1% median,
# 15/30 -- because the detector measures an opening span rather than the
# leaf the annotation records.
CONVENTIONS: dict[str, tuple[float, float]] = {
    "nordic": (TYPICAL_DOOR_FT, 0.648),
}


def detect_convention(text_boxes: list) -> str | None:
    """Which drafting tradition a drawing announces, or None if it does not.

    Deliberately conservative. Claiming a tradition changes how big the
    finished building is, so an unrecognised drawing keeps the defaults
    rather than being assigned a best guess.
    """
    words = {
        word
        for box in text_boxes
        for word in "".join(
            character if character.isalnum() else " "
            for character in getattr(box, "text", "").upper()
        ).split()
    }

    for convention, keywords in CONVENTIONS_BY_KEYWORDS():
        hits = len(words & set(keywords))
        if hits >= MIN_CONVENTION_HITS:
            logger.info("drawing reads as %s (%d matching name(s))", convention, hits)
            return convention
    return None


def element_sizes(text_boxes: list) -> tuple[float, float]:
    """The (door_ft, wall_ft) to measure this drawing with."""
    convention = detect_convention(text_boxes)
    if convention is None:
        return TYPICAL_DOOR_FT, TYPICAL_WALL_FT
    return CONVENTIONS[convention]
```

Replace the placeholder `CONVENTIONS_BY_KEYWORDS()` with a plain iteration
over `CONVENTION_KEYWORDS.items()` — it is written above only to make the
loop's intent obvious, and a helper for one dict is not worth defining.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_conventions.py -q`
Expected: PASS.

- [ ] **Step 5: Use it in the pipeline**

In `planto3d/pipeline.py`'s `extract`, the reference scale is currently
computed with the module defaults:

```python
    from_doors = scale_from_doors(...)
    from_walls = (scale_from_gauge(gauge) if gauge else scale_from_walls(...))
```

The text boxes are pooled a few lines further down as `boxes`. Move that
pooling above these two calls — it depends only on `floors`, so it can be
computed as soon as they exist — then select the sizes and pass them in:

```python
    boxes = [box for floor in floors for box in floor.text_boxes]

    # Element sizes differ by drafting tradition, and the drawing often
    # says which one it belongs to. An unrecognised sheet keeps the
    # defaults; see calibrate.CONVENTIONS for why only the wall value
    # moves.
    door_ft, wall_ft = element_sizes(boxes)

    from_doors = scale_from_doors(
        [opening for floor in floors for opening in floor.plan.openings],
        typical_width_ft=door_ft,
        gauge=gauge,
    )
    from_walls = (
        scale_from_gauge(gauge, typical_thickness_ft=wall_ft)
        if gauge
        else scale_from_walls(
            [wall for floor in floors for wall in floor.drawn_walls],
            typical_thickness_ft=wall_ft,
        )
    )
```

Delete the later duplicate assignment of `boxes` so it is pooled once.
Import `element_sizes` alongside the other `calibrate` imports at the top.

- [ ] **Step 6: Measure the effect**

```
PYTHONPATH=. .venv/Scripts/python.exe scripts/scale_accuracy.py "C:/Users/Rahul Soni/AppData/Local/Temp/claude/cubicasa_batch" --checkpoint models/unet_cubicasa.pt --limit 30
```

Expected: pooled median error falls from **17.7%** toward **9.9%**, and
the plans within a fifth rise from 20/30 toward 23/30. Task 1's per-source
table should show the improvement concentrated in the `walls` row, with
`doors` roughly unchanged.

The exact figure may differ from 9.9% because the detector fires only
where the labels are readable, and OCR does not read every sheet. **Report
what you actually measure.** If it does not improve at all, the detector
is not firing — check how many sheets log the "drawing reads as nordic"
line before assuming the constant is wrong.

Paste the full before/after output into your report.

- [ ] **Step 7: Full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add planto3d/calibrate.py planto3d/pipeline.py tests/test_conventions.py
git commit -m "Measure a drawing with its own tradition's element sizes"
```

---

## Task 4: Record what this cost and bought

The project's rule is that every published figure names the script that
produced it, and that ideas which did not work are written down so they
are not tried twice. This task is where the door-constant result gets
recorded — it is the more useful of the two findings, because it
contradicts what everyone assumes.

**Files:**
- Modify: `docs/AUDIT.md`

**Interfaces:** none. Documentation only.

- [ ] **Step 1: Add the measurements**

Add a section to `docs/AUDIT.md` in the voice of its neighbours. It must
carry, all produced by running `scripts/scale_accuracy.py` in the session
that does this work:

- the before and after pooled figures, and the per-source breakdown
- the four-way constant comparison, including that **correcting the door
  constant alone made the result worse** — 20.1% against 17.7% shipped —
  and why that is plausible: the detector measures an opening span while
  the annotation records the door leaf
- the implied real wall thickness distribution — min 0.478, p25 0.598,
  median 0.648, p75 0.803, max 1.176 ft — and the conclusion that follows:
  the interquartile spread is ±16% of the median, so **no constant can do
  better than that**, and the exact routes matter more than tuning
- that the Nordic wall value is measured from CubiCasa and applies only to
  drawings that identify themselves, leaving every other tradition on the
  shipped defaults and unvalidated

Update the gaps table's scale row with the new figure.

- [ ] **Step 2: Check the numbers against the run**

Re-read what you wrote against the actual script output from Task 3
Step 6. Every figure must match a line you can point to. If you cannot
point to it, remove it.

- [ ] **Step 3: Commit**

```bash
git add docs/AUDIT.md
git commit -m "docs: what element sizes by tradition bought, and what it did not"
```

---

## Self-Review

**Spec coverage.** The spec's workstream 1 asks for convention detection
switching constants (Task 3), a direct override from one known measurement
(Task 2), and extending rather than replacing the existing printed-scale
gate (Task 3 leaves `estimate_scale`, `scale_from_areas` and `corroborated`
untouched). Task 1 is not named in the spec but is a prerequisite: the
spec's gate is a number, and without per-source attribution a change that
helps one half of the population and harms the other is invisible.

The spec's Gate 1 target of <10% median is addressed but not guaranteed —
Task 3's measured ceiling on CubiCasa is 9.9%, and only where the detector
fires. Task 3 Step 6 says to report the real figure rather than the hoped
one.

**Placeholder scan.** One deliberate placeholder in Task 3 Step 3 —
`CONVENTIONS_BY_KEYWORDS()` — is called out in the step immediately below
it with the instruction to replace it with a plain dict iteration. Every
other step carries real code.

**Type consistency.** `scale_from_known_room(room, width_ft, height_ft)`
is defined in Task 2 and used only there. `element_sizes(text_boxes) ->
(door_ft, wall_ft)` is defined in Task 3 and consumed in Task 3 Step 5
with matching arity. `scale_override` is added to `extract` and `run` in
Task 2 Step 5 and forwarded with the keyword name used in the same step.
`"stated"` is registered in `PRINTED_SCALE_SOURCES` in the same task that
introduces it.
