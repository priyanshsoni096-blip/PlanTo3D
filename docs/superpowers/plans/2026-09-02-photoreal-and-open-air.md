# Photoreal Prompt and Open-Air Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the photoreal pass inventing a house the user did not ask for, and make correcting a wrongly-sealed balcony a one-time cost per drawing instead of a cost per run.

**Architecture:** Two independent changes. `planto3d/photoreal.py`'s `build_prompt` gains the `Design` the rest of the pipeline already honours, so the diffusion image stops contradicting the model beside it. `scripts/correct_and_build.py` gains a corrections file, so a plan annotated once stays annotated. Neither touches the segmenter, and neither needs a GPU to verify.

**Tech Stack:** Python 3.11+, `.venv`, pytest. No new dependencies.

**Spec:** `C:\Users\Rahul Soni\Downloads\PlanTo3D_Accuracy_and_Deploy_Plan.md` — this plan addresses that spec's Root Cause A (open-to-air spaces sealed because nothing labelled them) and Root Cause B (the photoreal pass as an impression), narrowed to what can be fixed without retraining. Read alongside `docs/AUDIT.md` for the measurement record.

## Global Constraints

- `pytest -q` must pass. It is at **802** now; each task adds tests. Never a net loss.
- Python is `.venv/Scripts/python.exe`; set `PYTHONPATH=.` when running scripts directly.
- No number goes in a doc or commit message unless a script produced it in that session.
- Match the existing code style: docstrings and comments explain *why*, not just *what*; every named constant carries a comment giving the reasoning behind its value. See `MAX_PROMPT_TOKENS` in `planto3d/photoreal.py` for the house standard.
- Commit messages: short, declarative, describing the *effect*. No conventional-commit prefixes beyond an occasional `docs:`.
- Commit with explicit paths only. Never `git add -A` or `git add .` — `demo_plans/` is intentionally untracked.
- **Do not touch the segmenter, its checkpoint, or the training code.** Window accuracy is a data problem needing annotation and a retrain; it is deliberately out of scope here and needs the user's go-ahead first.

---

## What was measured before writing this

Run this session, on the current head:

| Finding | Measurement |
| --- | --- |
| `build_prompt` ignores every design choice | `build_prompt(2, [...])` returns the identical string for all 4 styles and all 3 times — always "modern luxury residence at dusk" |
| Open-to-air spaces are rarely detected | 26 of 171 rooms (15%) end up open to sky across 16 plans; 2 plans got **zero** |
| …and mostly not from labels | of those 26, only **5** came from a printed label; **21** from the segmenter's `outdoor` class |
| …because most rooms carry no evidence at all | **51 of 171 rooms (30%)** have neither a label nor a predicted category |
| Windows are the rarest thing on the page | window = **0.1019%** of annotated pixels, against door 0.48% and wall 6.46% |
| End-to-end, on 20 CubiCasa sheets | **5/20 (25%)** correct on every count; failures: size 7, walls 7, openings 6, rooms 2 |
| Railings are **not** broken | 22 of 24 open rooms are railed; the 2 unrailed are paving, correctly so |
| `site.railed_rooms` / `site.has_open_edge` are dead | defined and tested, called from nowhere in `planto3d/`; the live path is `features.get("open")` at `planto3d/extrude.py:1633` |

---

## File Structure

| File | Responsibility |
| --- | --- |
| `planto3d/photoreal.py` (modify) | `build_prompt` takes an optional `Design` and derives the subject, cladding and light from it instead of hardcoding one house. New `STYLE_SUBJECTS`, `TONE_CLADDING`, `TIME_LIGHT` tables. |
| `tests/test_photoreal.py` (modify) | Prove each choice changes the prompt, and that the token budget still holds. |
| `scripts/correct_and_build.py` (modify) | `--save-corrections` / `--corrections` so a plan annotated once stays annotated. |
| `tests/test_corrections_file.py` (create) | Round-trip the corrections file, including a malformed one. |
| `planto3d/site.py` (modify) | Delete the dead `railed_rooms` / `has_open_edge` / `OPEN_EDGE_KEYWORDS`. |
| `tests/test_windows_railings.py` (modify) | Drop the tests for the deleted functions; keep everything else. |
| `notebooks/run_on_colab.ipynb`, `notebooks/photoreal_on_colab.ipynb` (modify) | Pass `DESIGN` into `build_prompt`. |
| `planto3d/ingest.py` (modify) | `split_sheet` gains a `force` override, so a user can settle a sheet the detector reads wrongly. |
| `planto3d/pipeline.py` (modify) | `_split_into_storeys`, `extract` and `run` thread that override through. |
| `tests/test_sheet_split.py` (create) | Forcing a split, forcing a sheet whole, and refusing to invent a divide. |

---

## Task 1: Let the photoreal prompt describe the house that was asked for

Today `build_prompt(storeys, room_labels)` never sees the user's choices, so a "traditional" house at "night" is described to the diffusion model as *"modern luxury residence at dusk, warm honey-toned limestone cladding"*. The 3D model honours those choices through `Design.palette()` and `Design.lighting()`; only the photoreal pass ignores them, so the two outputs disagree about what building this is.

**Files:**
- Modify: `planto3d/photoreal.py` (the `BASE_PROMPT` constant and `build_prompt`)
- Test: `tests/test_photoreal.py`

**Interfaces:**
- Consumes: `planto3d.design.Design` (fields `style`, `colour`, `time`, `landscaping`, `creativity`), `planto3d.design.STYLES` = `modern, luxury, traditional, minimalist`, `TONES` = `light, dark, warm`, `TIMES` = `day, sunset, night`.
- Produces: `build_prompt(storeys: int, room_labels: list[str] | None = None, design: Design | None = None) -> str`. The `design` parameter is **keyword-optional and last**, so every existing caller keeps working unchanged and falls back to today's wording.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_photoreal.py`:

```python
from planto3d.design import STYLES, TIMES, TONES, Design
from planto3d.photoreal import MAX_PROMPT_TOKENS, _tokens, build_prompt


def _design(**overrides):
    base = dict(
        style="modern", colour="warm", time="day",
        landscaping="basic", creativity="balanced",
    )
    base.update(overrides)
    return Design(**base)


def test_every_style_changes_the_prompt():
    # A traditional house described as "modern" is the wrong building.
    prompts = {
        style: build_prompt(2, ["BALCONY"], design=_design(style=style))
        for style in STYLES
    }
    assert len(set(prompts.values())) == len(STYLES), prompts


def test_every_time_of_day_changes_the_prompt():
    prompts = {
        time: build_prompt(2, ["BALCONY"], design=_design(time=time))
        for time in TIMES
    }
    assert len(set(prompts.values())) == len(TIMES), prompts


def test_every_tone_changes_the_prompt():
    prompts = {
        tone: build_prompt(2, ["BALCONY"], design=_design(colour=tone))
        for tone in TONES
    }
    assert len(set(prompts.values())) == len(TONES), prompts


def test_night_is_not_described_as_dusk():
    night = build_prompt(2, None, design=_design(time="night"))
    assert "dusk" not in night
    assert "night" in night


def test_the_prompt_still_fits_the_token_budget():
    # CLIP reads 77 tokens and silently drops the rest, tail first -- which
    # is where the drawing-derived detail lives. Every combination must fit.
    labels = ["BALCONY", "TERRACE GARDEN", "PARKING", "SWIMMING POOL"]
    for style in STYLES:
        for tone in TONES:
            for time in TIMES:
                design = _design(style=style, colour=tone, time=time)
                prompt = build_prompt(3, labels, design=design)
                assert _tokens(prompt) <= MAX_PROMPT_TOKENS, (style, tone, time)


def test_no_design_keeps_the_previous_wording():
    # Every existing caller passes no design and must be unaffected.
    assert "modern luxury residence at dusk" in build_prompt(2, ["BALCONY"])
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_photoreal.py -q -k "style or time or tone or budget or wording"
```

Expected: FAIL — `build_prompt() got an unexpected keyword argument 'design'`.

- [ ] **Step 3: Add the vocabulary tables**

In `planto3d/photoreal.py`, directly above `BASE_PROMPT`:

```python
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
STYLE_SUBJECTS = {
    "modern": "modern residence, clean rectilinear massing",
    "luxury": "luxury residence, generous proportions and deep reveals",
    "traditional": "traditional residence, pitched roofs and punched windows",
    "minimalist": "minimalist residence, unadorned planes and hidden detail",
}

TONE_CLADDING = {
    "light": "pale limestone and white render",
    "dark": "charcoal brick and dark stained timber",
    "warm": "warm honey-toned limestone",
}

# Paired with the lighting the renderer uses for the same choice, so the
# render beside it and the diffusion image agree about the hour.
TIME_LIGHT = {
    "day": "clear midday daylight, crisp shadows",
    "sunset": "low golden sunset light, long shadows",
    "night": "night, warm amber interior lighting in every window",
}
```

- [ ] **Step 4: Teach `build_prompt` to use them**

Replace `build_prompt`'s signature and the line that builds `parts[0]`:

```python
def build_prompt(
    storeys: int,
    room_labels: list[str] | None = None,
    design=None,
) -> str:
    """Compose the prompt, naming what the plan actually contains.

    Grounding the description in extracted labels keeps the model from
    inventing features the house does not have -- a pool, a pitched roof --
    which is the usual way a stylization stops resembling its subject.

    ``design`` is the same ``planto3d.design.Design`` the renderer uses. Passing
    it makes the prompt describe the house the user asked for; omitting it
    keeps the original wording, so existing callers are unaffected.
    """
```

Then, where `parts` is first assembled, replace:

```python
    parts = [BASE_PROMPT.format(storeys=storeys)]
```

with:

```python
    if design is None:
        opening = BASE_PROMPT.format(storeys=storeys)
    else:
        # Subject, cladding and light, in that order and never dropped: a
        # building of the wrong material is wrong in every frame, and an
        # hour that contradicts the render beside it is worse than plain.
        opening = (
            f"professional architectural visualization of a {storeys}-storey "
            f"{STYLE_SUBJECTS.get(design.style, STYLE_SUBJECTS['modern'])}, "
            f"{TONE_CLADDING.get(design.colour, TONE_CLADDING['warm'])} cladding, "
            f"{TIME_LIGHT.get(design.time, TIME_LIGHT['day'])}"
        )

    parts = [opening]
```

Leave everything below unchanged — the site phrases, `STYLE_PHRASES`, `QUALITY_PHRASES` and the budget loop all still work, and the budget is computed from `parts[0]`, which is now the design-aware opening.

- [ ] **Step 5: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_photoreal.py -q
```

Expected: PASS, including the existing tests.

If `test_the_prompt_still_fits_the_token_budget` fails, the openings above are too long — shorten the offending `STYLE_SUBJECTS`/`TONE_CLADDING` entry rather than raising `MAX_PROMPT_TOKENS`. The 77 is CLIP's limit, not ours.

- [ ] **Step 6: Show the prompts actually differ**

```bash
PYTHONPATH=. .venv/Scripts/python.exe -c "
from planto3d.design import Design
from planto3d.photoreal import build_prompt
for s in ('modern','traditional'):
    for t in ('day','night'):
        d=Design(style=s,colour='warm',time=t,landscaping='basic',creativity='balanced')
        print(f'{s:12}{t:8}', build_prompt(2, ['BALCONY'], design=d)[:110])
"
```

Expected: four visibly different prompts. Paste the output into your report.

- [ ] **Step 7: Run the full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add planto3d/photoreal.py tests/test_photoreal.py
git commit -m "Describe the house the user asked for, not always a modern one at dusk"
```

---

## Task 2: Pass the design through from the notebooks

Task 1 makes `design` optional so nothing breaks; this task makes the two notebooks that generate images actually use it. Without this, Task 1 changes nothing a user sees.

**Files:**
- Modify: `notebooks/run_on_colab.ipynb` (the cell calling `build_prompt`)
- Modify: `notebooks/photoreal_on_colab.ipynb` (same)

**Interfaces:**
- Consumes: `build_prompt(storeys, room_labels, design=...)` from Task 1; both notebooks already have a `DESIGN` object in scope from their design-choice cell.

- [ ] **Step 1: Find the call sites**

```bash
PYTHONPATH=. .venv/Scripts/python.exe -c "
import json
for name in ('run_on_colab','photoreal_on_colab'):
    nb=json.load(open(f'notebooks/{name}.ipynb',encoding='utf-8'))
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code' and 'build_prompt(' in ''.join(c['source']):
            print(name,'cell',i)
"
```

- [ ] **Step 2: Edit each call**

In each cell found, change:

```python
prompt = build_prompt(len(result.floors), labels)
```

to:

```python
prompt = build_prompt(len(result.floors), labels, design=DESIGN)
```

Confirm `DESIGN` is genuinely in scope in that notebook — both define it in their design-choice cell. If a notebook does not define `DESIGN`, stop and report rather than inventing one.

- [ ] **Step 3: Verify the notebooks are still valid**

```bash
PYTHONPATH=. .venv/Scripts/python.exe -c "
import json
for name in ('run_on_colab','photoreal_on_colab'):
    nb=json.load(open(f'notebooks/{name}.ipynb',encoding='utf-8'))
    print(name,'OK, cells:',len(nb['cells']))
"
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/run_on_colab.ipynb notebooks/photoreal_on_colab.ipynb
git commit -m "Feed the design choices into the photoreal prompt"
```

---

## Task 3: Annotate a plan once, not once per run

This is the practical answer to open-to-air spaces being sealed. The model finds only 15% of them and 30% of rooms carry no evidence at all, so a human has to say which spaces are open. Today that means retyping every `--correct` flag on every run. A corrections file makes it a one-time cost per drawing.

**Files:**
- Modify: `scripts/correct_and_build.py`
- Test: `tests/test_corrections_file.py` (create)

**Interfaces:**
- Consumes: `planto3d.corrections.CATEGORY_LABELS`, `apply_room_corrections`, and `parse_correction(text) -> tuple[tuple[int,int], str]` which already exists in `scripts/correct_and_build.py`.
- Produces, in `scripts/correct_and_build.py`:
  - `corrections_to_lines(corrections: dict[tuple[int, int], str]) -> list[str]`
  - `corrections_from_lines(lines: list[str]) -> dict[tuple[int, int], str]`
  - two new CLI flags, `--save-corrections PATH` and `--corrections PATH`

The file format is the same `FLOOR:ROOM=CATEGORY` the `--correct` flag already takes, one per line, `#` for comments — so anything a user can type they can also save, and the file is readable without documentation.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corrections_file.py`:

```python
"""A plan annotated once should stay annotated.

The segmenter finds only about 15% of the spaces that are open to the air,
and 30% of rooms carry neither a label nor a predicted type, so a human has
to say which spaces are terraces. Making them retype that on every run is
how a correction feature goes unused.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from correct_and_build import (  # noqa: E402
    corrections_from_lines,
    corrections_to_lines,
)


def test_corrections_round_trip():
    original = {(0, 5): "open", (1, 2): "paving", (0, 11): "lawn"}
    assert corrections_from_lines(corrections_to_lines(original)) == original


def test_saved_lines_use_the_same_syntax_as_the_flag():
    # Whatever a user can type at --correct they can also read in the file.
    lines = corrections_to_lines({(0, 5): "open"})
    assert "1:5=open" in lines  # floors print from 1, rooms index from 0


def test_comments_and_blank_lines_are_ignored():
    lines = ["# ground floor", "", "1:5=open", "   ", "# done"]
    assert corrections_from_lines(lines) == {(0, 5): "open"}


def test_an_unknown_category_is_refused_not_guessed():
    with pytest.raises(ValueError, match="not a feature category"):
        corrections_from_lines(["1:5=outdoor"])


def test_a_malformed_line_names_itself():
    with pytest.raises(ValueError, match="oops"):
        corrections_from_lines(["oops"])
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_corrections_file.py -q
```

Expected: FAIL — `ImportError: cannot import name 'corrections_to_lines'`.

- [ ] **Step 3: Add the two functions**

In `scripts/correct_and_build.py`, directly below `parse_correction`:

```python
# The file format is the flag's own syntax, one per line. A user who can
# type a correction can read the file without being taught anything new,
# and the file can be edited by hand or produced by another tool.
CORRECTIONS_HEADER = (
    "# PlanTo3D room corrections. One FLOOR:ROOM=CATEGORY per line.\n"
    "# Floor numbers are the ones --list prints; rooms are indexed from 0.\n"
)


def corrections_to_lines(corrections: dict[tuple[int, int], str]) -> list[str]:
    """Render corrections as the same text ``--correct`` accepts."""
    return [
        f"{floor + 1}:{room}={category}"
        for (floor, room), category in sorted(corrections.items())
    ]


def corrections_from_lines(lines: list[str]) -> dict[tuple[int, int], str]:
    """Read corrections back, ignoring comments and blank lines.

    Reuses ``parse_correction`` so the file and the flag can never drift
    apart, and so a bad line is refused with the whole vocabulary rather
    than silently skipped -- a correction that quietly does nothing is
    worse than one that stops you.
    """
    corrections = {}
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        where, category = parse_correction(text)
        corrections[where] = category
    return corrections
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_corrections_file.py -q
```

Expected: PASS.

- [ ] **Step 5: Wire the two flags into the CLI**

In the `argparse` block, after `--correct`:

```python
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        metavar="PATH",
        help="read corrections from a file written by --save-corrections",
    )
    parser.add_argument(
        "--save-corrections",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the corrections given on this run to a file, to reuse later",
    )
```

Add both to `main`'s signature (`corrections_path: Path | None`, `save_path: Path | None`) and pass them through from `__main__`.

In `main`, replace the block that begins `if corrections:` with:

```python
    parsed = dict(parse_correction(text) for text in corrections)
    if corrections_path is not None:
        # The file first, the flags second, so a one-off --correct can
        # override a saved annotation without editing the file.
        from_file = corrections_from_lines(
            corrections_path.read_text(encoding="utf-8").splitlines()
        )
        print(f"\nread {len(from_file)} correction(s) from {corrections_path}")
        parsed = {**from_file, **parsed}

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            CORRECTIONS_HEADER + "\n".join(corrections_to_lines(parsed)) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(parsed)} correction(s) to {save_path}")

    if parsed:
        result = apply_room_corrections(copy.deepcopy(result), parsed)
        print(f"\napplied {len(parsed)} correction(s):")
        for (floor, room), category in sorted(parsed.items()):
            print(f"   floor {floor + 1} room {room} -> {category}")
        show(result)
```

- [ ] **Step 6: Prove the round trip end to end**

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/correct_and_build.py \
    data/bridge/11001.gif .cf_a --checkpoint models/unet_cubicasa.pt \
    --correct 1:11=open --correct 1:12=paving --save-corrections .cf.txt

cat .cf.txt

PYTHONPATH=. .venv/Scripts/python.exe scripts/correct_and_build.py \
    data/bridge/11001.gif .cf_b --checkpoint models/unet_cubicasa.pt \
    --corrections .cf.txt

PYTHONPATH=. .venv/Scripts/python.exe -c "
from pathlib import Path
a=Path('.cf_a/house.glb').read_bytes(); b=Path('.cf_b/house.glb').read_bytes()
print('same model from file as from flags:', a==b)
"
rm -rf .cf_a .cf_b .cf.txt
```

Expected: the two models are byte-identical. Paste the output into your report. If they differ, the file is not reproducing the flags and the task is not done.

- [ ] **Step 7: Document it**

In `scripts/correct_and_build.py`'s module docstring, after the two existing usage examples, add a third:

```
    # 3. save the corrections, and reuse them on every later run
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --correct 1:5=open --save-corrections plan-corrections.txt
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --corrections plan-corrections.txt
```

And in `README.md`, under "Correcting what the reader got wrong", after the two existing commands:

```markdown
Corrections can be saved and reused, so a plan is annotated once rather than
once per run:

```bash
python scripts/correct_and_build.py plan.pdf output --correct 1:5=open --save-corrections plan.txt
python scripts/correct_and_build.py plan.pdf output --corrections plan.txt
```
```

- [ ] **Step 8: Run the full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add scripts/correct_and_build.py tests/test_corrections_file.py README.md
git commit -m "Save a plan's corrections so it is annotated once, not once per run"
```

---

## Task 4: Delete the dead railing code

`site.railed_rooms` and `site.has_open_edge` are defined, exported and tested, but called from nowhere in `planto3d/`. Railings are actually built from `features.get("open", [])` at `planto3d/extrude.py:1633`. The dead pair is actively harmful: it reads like the live mechanism, and measuring against it during this plan's investigation produced a false result — "24 of 24 open rooms unrailed", when the real figure is 22 of 24 railed.

**Files:**
- Modify: `planto3d/site.py` (remove `OPEN_EDGE_KEYWORDS`, `has_open_edge`, `railed_rooms`)
- Modify: `tests/test_windows_railings.py` (remove only the tests for those two functions)
- Modify: `planto3d/corrections.py` (its docstring names `has_open_edge`)
- Modify: `planto3d/pipeline.py` (the `build` docstring names `site.has_open_edge`)
- Modify: `tests/test_corrections.py` (imports and asserts `has_open_edge`)

- [ ] **Step 1: Confirm it really is dead before deleting anything**

```bash
grep -rn "railed_rooms\|has_open_edge\|OPEN_EDGE_KEYWORDS" --include=*.py --include=*.ipynb . | grep -v "\.venv"
```

Expected: hits only in `planto3d/site.py` (the definitions), `tests/`, and docstrings in `planto3d/corrections.py` and `planto3d/pipeline.py`. **If anything in `planto3d/` actually calls either function, stop and report** — the premise is wrong and the task should not proceed.

- [ ] **Step 2: Delete the definitions**

Remove from `planto3d/site.py`: the `OPEN_EDGE_KEYWORDS` constant with its comment, and the whole `has_open_edge` and `railed_rooms` functions.

- [ ] **Step 3: Remove the tests that covered them**

In `tests/test_windows_railings.py`, delete the tests asserting on `has_open_edge` and `railed_rooms` (around lines 127-139) and drop them from the import at line 8. Leave every other test in the file alone.

In `tests/test_corrections.py`, remove `has_open_edge` from the `planto3d.site` import and delete `test_open_categories_are_also_railed_by_site_py`. Replace it with a test of the live mechanism:

```python
def test_the_open_category_is_what_earns_a_railing():
    # Railings are built from features.get("open") in extrude.py, so the
    # canonical "open" label has to classify to exactly that -- this is
    # the property the correction relies on for a balcony to get a rail.
    from planto3d.features import feature_for
    from planto3d.geometry_types import Room

    room = Room(
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        label=CATEGORY_LABELS["open"],
    )
    assert feature_for(room) == "open"
```

- [ ] **Step 4: Fix the two docstrings that name the deleted function**

In `planto3d/corrections.py`'s module docstring and `planto3d/pipeline.py`'s `build` docstring, replace mentions of `site.has_open_edge()` with `site.classify_cover()` — which is live and makes the same point about reading `room.label`.

- [ ] **Step 5: Run the full suite**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: PASS, with a count a few lower than 802 (the deleted tests). A failure here means something did depend on the deleted code.

- [ ] **Step 6: Prove railings still get built**

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/correct_and_build.py \
    data/bridge/11001.gif .rail_check --checkpoint models/unet_cubicasa.pt --correct 1:11=open
PYTHONPATH=. .venv/Scripts/python.exe -c "
import trimesh
s=trimesh.load('.rail_check/house.glb')
rail=sum(len(g.faces) for n,g in s.geometry.items() if n.split('_')[0]=='railing')
print('railing faces:', rail)
assert rail > 0, 'railings disappeared -- the deleted code was not dead'
"
rm -rf .rail_check
```

- [ ] **Step 7: Commit**

```bash
git add planto3d/site.py planto3d/corrections.py planto3d/pipeline.py tests/test_windows_railings.py tests/test_corrections.py
git commit -m "Delete the railing helpers nothing calls"
```

---

## Task 5: Let the user settle a sheet the splitter gets wrong

A sheet carrying two floors that is read as one reconstructs both plans as a
single flat building -- confidently, with plausible numbers. Measured against
CubiCasa's own annotations over 60 sheets: **58/60 exact, 100% precision, 86%
recall** -- 2 of 14 multi-plan sheets collapse. The two failures are different:

- **8583** -- a horizontal cut at row 412 *is* proposed by `_gutter_cuts`; the
  acceptance gates reject it (the thin piece holds 7.3% of the ink against
  `MIN_PLAN_INK_SHARE` 0.20, and encloses 7.0% against `MIN_ENCLOSED_SHARE`
  0.10). A forced split can simply use the cut that was already found.
- **11378** -- two units sharing a party wall. `_gutter_cuts` and
  `_boundary_cuts` both propose **nothing** on either axis, so there is no cut
  to force. Inventing a divide here risks cutting a real plan in half.

Loosening the two gates was swept jointly and **rejected**: at (0.07, 0.06)
recall rises to 93% but precision falls to 93%, and exact stays 58/60. It
trades one error for another, so the thresholds are left alone -- the user
chose control over a different guess.

So this task adds an override, not a tuning change. `--split N` uses the cuts
the detector already proposed, bypassing only the acceptance gates.
`--no-split` forces the sheet to stay whole. When no cut was proposed at all,
it says so plainly and points at the remedy the pipeline already supports:
crop the sheet yourself and upload one image per storey.

**Files:**
- Modify: `planto3d/ingest.py` (`split_sheet`)
- Modify: `planto3d/pipeline.py` (`_split_into_storeys`, `extract`, `run`)
- Modify: `scripts/correct_and_build.py` (two CLI flags)
- Test: `tests/test_sheet_split.py` (create)

**Interfaces:**
- Consumes: `planto3d.ingest._ink_mask`, `_gutter_cuts`, `_boundary_cuts`,
  `_pieces_look_like_plans`, and the constants `MIN_SPLIT_WIDTH`,
  `MIN_PIECE_FRACTION`, `MIN_BOUNDARY_CUTS` -- all already in `ingest.py`.
- Produces:
  - `split_sheet(image, force: int | None = None) -> list[np.ndarray]`.
    `force=None` keeps today's automatic behaviour exactly. `force=1` returns
    the image whole. `force=N` (N>=2) takes the N-1 most promising proposed
    cuts and skips `_pieces_look_like_plans`, raising `ValueError` when fewer
    than N-1 cuts were proposed on either axis.
  - `_split_into_storeys(page, output_dir, force: int | None = None)`
  - `extract(source, output_dir, segmenter=classical_mask, crop=True, split: int | None = None)`
  - `run(...)` gains the same `split` parameter, passed straight through.
  - CLI flags `--split N` and `--no-split` on `scripts/correct_and_build.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sheet_split.py`:

```python
"""Forcing a split on a sheet the detector reads wrongly.

Measured against CubiCasa's annotations, automatic splitting is right on
58 of 60 sheets and never splits a single plan wrongly. The two it misses
are a sheet whose thin floor the acceptance gates reject, and two units
sharing a party wall with no gutter at all. The first is recoverable by
using the cut that was already proposed; the second is not recoverable
without inventing a divide, so it must fail loudly rather than guess.
"""

import numpy as np
import pytest

from planto3d.ingest import split_sheet


def _two_plans_side_by_side() -> np.ndarray:
    """A white sheet with two ink blocks and a wide blank gutter between."""
    sheet = np.full((600, 1200, 3), 255, dtype=np.uint8)
    # Each "plan" is a hollow rectangle, so it encloses area like a real one.
    for left in (60, 700):
        sheet[80:520, left : left + 440] = 0
        sheet[120:480, left + 40 : left + 400] = 255
    return sheet


def _one_plan() -> np.ndarray:
    sheet = np.full((600, 1200, 3), 255, dtype=np.uint8)
    sheet[80:520, 100:1100] = 0
    sheet[120:480, 140:1060] = 255
    return sheet


def test_force_none_keeps_todays_behaviour():
    assert len(split_sheet(_two_plans_side_by_side(), force=None)) == 2


def test_force_one_keeps_the_sheet_whole():
    # The escape hatch for a sheet that is one plan but splits wrongly.
    assert len(split_sheet(_two_plans_side_by_side(), force=1)) == 1


def test_force_two_splits_a_sheet_the_gates_would_reject():
    # A thin strip of a second plan: real, but too little ink to pass
    # MIN_PLAN_INK_SHARE. Automatic reading leaves it whole; forcing splits it.
    sheet = np.full((600, 1200, 3), 255, dtype=np.uint8)
    sheet[80:520, 60:900] = 0
    sheet[120:480, 100:860] = 255
    sheet[300:340, 1000:1160] = 0  # the thin strip, past a wide gutter
    assert len(split_sheet(sheet, force=None)) == 1
    assert len(split_sheet(sheet, force=2)) == 2


def test_forcing_a_split_with_no_cut_available_says_so():
    # The party-wall case. There is no gutter, so there is nothing to force,
    # and guessing a divide would cut a real plan in half.
    with pytest.raises(ValueError, match="no dividing line"):
        split_sheet(_one_plan(), force=2)


def test_forcing_more_pieces_than_there_are_cuts_says_so():
    with pytest.raises(ValueError, match="no dividing line"):
        split_sheet(_two_plans_side_by_side(), force=5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sheet_split.py -q`

Expected: FAIL — `split_sheet() got an unexpected keyword argument 'force'`.

- [ ] **Step 3: Add `force` to `split_sheet`**

In `planto3d/ingest.py`, change the signature:

```python
def split_sheet(image: np.ndarray, force: int | None = None) -> list[np.ndarray]:
```

and add this paragraph to the end of its docstring, before the closing quotes:

```
    ``force`` overrides the decision. ``None`` reads the sheet automatically,
    which is right on 58 of 60 CubiCasa sheets and has never split a single
    plan wrongly. ``1`` keeps the sheet whole. ``N`` takes the N-1 cuts the
    detector proposed and skips the acceptance gates -- which is what
    recovers a sheet whose second floor is too thin to pass them. It cannot
    invent a divide: where no cut was proposed, forcing raises rather than
    cutting a real plan in half.
```

Immediately after the existing `if width < MIN_SPLIT_WIDTH: return [image]`
guard, add the whole-sheet case:

```python
    if force == 1:
        return [image]
```

Inside the `for axis in (0, 1):` loop, replace this block:

```python
        if not _pieces_look_like_plans(ink, divisions, axis):
            continue
```

with:

```python
        # A forced split uses the cuts that were found and skips the gates:
        # the user has looked at the sheet and the gates have not.
        if force is None:
            if not _pieces_look_like_plans(ink, divisions, axis):
                continue
        else:
            if len(divisions) < force - 1:
                continue
            divisions = divisions[: force - 1]
```

After the `for axis` loop ends, before the function returns the unsplit
image, add the loud failure:

```python
    if force is not None and force > 1:
        raise ValueError(
            f"asked for {force} plan(s) but found no dividing line in this "
            "sheet. Two plans sharing a wall leave no gutter to cut along. "
            "Crop the sheet yourself and pass one image per storey."
        )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sheet_split.py -q`

Expected: PASS. If `test_force_two_splits_a_sheet_the_gates_would_reject`
fails because the synthetic strip does not produce a gutter cut, adjust the
synthetic sheet's geometry until `_gutter_cuts` proposes one — do not weaken
the assertion, and do not change `MIN_PIECE_FRACTION` or any other constant
to make a test pass.

- [ ] **Step 5: Thread `split` through the pipeline**

In `planto3d/pipeline.py`, give `_split_into_storeys` the parameter:

```python
def _split_into_storeys(
    page: Path, output_dir: Path, force: int | None = None
) -> list[Path]:
```

and inside it change `pieces = split_sheet(image)` to
`pieces = split_sheet(image, force=force)`.

Add `split: int | None = None` as the last parameter of both `extract` and
`run`. Document it in both docstrings as: "``split`` overrides the sheet
splitter: 1 keeps the sheet whole, N forces N plans, None reads it
automatically." Then pass it through in `extract`:

```python
    if len(cropped) == 1:
        cropped = _split_into_storeys(cropped[0], pages_dir, force=split)
```

and forward it in `run`:

```python
    result = extract(source, output_dir, segmenter, crop, split=split)
```

- [ ] **Step 6: Add the CLI flags**

In `scripts/correct_and_build.py`'s `argparse` block, after `--no-crop`:

```python
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        metavar="N",
        help="force the sheet into N plans, overriding the splitter",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="force the sheet to be read as a single plan",
    )
```

Refuse the contradiction rather than silently preferring one, immediately
after `arguments = parser.parse_args()`:

```python
    if arguments.no_split and arguments.split is not None:
        parser.error("--split and --no-split contradict each other")
```

Add `split: int | None` to `main`'s signature, pass it to `extract`, and
resolve the flags at the call site:

```python
        split=1 if arguments.no_split else arguments.split,
```

Document both flags in the module docstring beside the existing usage
examples.

- [ ] **Step 7: Prove it on the sheet that actually fails**

The CubiCasa sample lives at a session temp path; `docs/PROJECT_GUIDE.md`
records that the corpus is not in this checkout. Write this to a scratch
file and run it with `PYTHONPATH=. .venv/Scripts/python.exe <file>`:

```python
from pathlib import Path

from planto3d.ingest import read_image, split_sheet

root = Path("C:/Users/Rahul Soni/AppData/Local/Temp/claude/cubicasa_batch")

hits = list(root.glob("*/8583/F1_scaled.png"))
if not hits:
    print("8583 not present; skipping (see docs/PROJECT_GUIDE.md)")
else:
    image = read_image(hits[0])
    print("8583 automatic:", len(split_sheet(image)), "piece(s)")
    print("8583 forced   :", len(split_sheet(image, force=2)), "piece(s)")
    assert len(split_sheet(image, force=2)) == 2

hits = list(root.glob("*/11378/F1_scaled.png"))
if hits:
    image = read_image(hits[0])
    try:
        split_sheet(image, force=2)
        print("11378: FAILED to raise -- a divide was invented")
    except ValueError as error:
        print("11378 refuses, correctly:", error)
```

Expected: 8583 goes 1 piece to 2; 11378 raises the "no dividing line"
message rather than inventing a cut. Paste the output into your report.

- [ ] **Step 8: Confirm automatic behaviour is unchanged**

The whole point of an override is that it changes nothing until it is used.

Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/split_accuracy.py "C:/Users/Rahul Soni/AppData/Local/Temp/claude/cubicasa_batch" --limit 60`

Expected: **58/60 exact, precision 100%, recall 86%** — identical to before
this task. Any change means `force=None` did not stay on the old path, and
the task is not done. Paste the output into your report.

- [ ] **Step 9: Run the full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add planto3d/ingest.py planto3d/pipeline.py scripts/correct_and_build.py tests/test_sheet_split.py
git commit -m "Let the user force a sheet split the detector reads wrongly"
```

---

## Not in this plan, deliberately

**Windows.** The measurement says this is a data problem, not a tuning one: window is **0.1019%** of annotated pixels, five times rarer than doors, and the same weights score IoU 0.239 on CVC-FP — which draws windows larger — against 0.089 on CubiCasa. Fixing it means annotating window-dense plans and retraining, which costs GPU time and needs the user's agreement before starting. It deserves its own plan. Four narrower fixes were already tried and rejected; see `docs/AUDIT.md`'s "tried and rejected" table before proposing anything.

**Scale.** The largest end-to-end failure (7 of 20 sheets), and a real target — but it is the spec's Phase 4 and a plan of its own.

**Automatic splitting of party-wall sheets.** Sheet 11378 puts two units either side of a shared wall, leaving no gutter for any threshold to find. Task 5 makes it fail loudly and points at cropping instead; detecting it automatically needs a different mechanism than the ink-profile splitter and is not attempted here.

**Detecting open-air spaces automatically.** The honest position is that 30% of rooms carry no evidence at all, so no amount of downstream logic can infer them. That is the segmenter's single coarse `outdoor` class, and improving it is the same retrain conversation as windows. Task 3 makes the human answer cheap and permanent instead.

**SDXL, multi-guide ControlNet, seed variation.** All plausible improvements to the photoreal pass, all requiring a GPU to evaluate, none verifiable from this machine. Worth doing after Task 1 and 2 land, so the prompt is right before the model changes underneath it.

---

## Self-Review

**Coverage of what was asked.** Windows → deliberately deferred with the measurement that justifies it. Open-air spaces sealed → Task 3 (the human fix, made permanent) plus the honest statement that automatic detection needs a retrain. Photoreal prompt guessing → Tasks 1 and 2, which fix a confirmed defect. More gaps → the measurement table above, and Task 4 fixes one found while writing this.

**Placeholder scan.** No TBD/TODO. Every code step carries the real code. Task 4's step 1 is a genuine precondition check with a stated abort, not a placeholder.

**Type consistency.** `build_prompt(storeys, room_labels, design=None)` is used identically in Tasks 1 and 2. `corrections_to_lines` / `corrections_from_lines` have the same signatures in Task 3's tests and implementation, and both reuse the existing `parse_correction`. `CATEGORY_LABELS` is imported in Task 4's replacement test and already imported in that file.
