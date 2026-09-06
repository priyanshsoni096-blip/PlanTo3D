# Open to Air Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide correctly which spaces have sky above them, on plans that
name nothing, and build them open rather than sealed.

**Architecture:** The mechanism already works — a room identified as open
propagates through `open_to_sky` to geometry, verified across three
drafting conventions. The detection does not. Three candidate evidence
sources are **swept and measured**, and only those that earn their place
ship. The sweep harness is Task 1 and is itself the deliverable that makes
the rest arguable from numbers rather than intuition.

**Tech Stack:** numpy, OpenCV, PyTorch (inference only — no retraining),
pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-plan-accurate-3d-design.md`,
workstream 3.

## Baseline, measured this session

Over 60 CubiCasa plans, 669 rooms (`scripts/output_scorecard.py` corpus):

| | count | share |
| --- | --- | --- |
| rooms ending open to sky | 93 | 13.9% |
| carrying a printed label | 23 | **3.4%** |
| carrying a predicted type | 490 | 73.2% |
| carrying **neither** | **171** | **25.6%** |

Scorecard: **10 of 30 correct on every count**; the `openings` check fails
**9 of 30**, second only to `size` at 10.

**The 3.4% label rate is the fact that shapes this whole workstream.**
`feature_for` prefers a printed name and falls back to the predicted type,
and `CATEGORY_FEATURES` maps exactly one predicted type to open air:
`"outdoor" -> "open"`. So on this corpus, open-to-sky detection is very
nearly *the segmenter's OUTDOOR class alone*. Anything that does not move
OUTDOOR moves nothing.

**The 25.6% is the ceiling.** Those rooms have no label and no predicted
type. No rule in this plan can decide them, and a plan that claims
otherwise is wrong. The realistic target is to convert part of the 73.2%
that is typed but typed *wrongly*, and to stop discarding evidence the
model already produced.

## Global Constraints

- **No retraining, no architecture change, no new corpus.** Inference-time
  changes only. The spec and the user's standing instruction both require
  agreement before any of those, and this plan does not seek it.
- **Every number written into a doc, comment or commit message must come
  from a script run in that session.** No estimates, no carried-forward
  figures.
- **Do not fit a constant to one dataset.** Any threshold this plan
  introduces must be swept, and the sweep table recorded, exactly as
  `WINDOW_PROBABILITY_FLOOR` was.
- **A candidate that does not measure well does not ship.** Recording it as
  measured-and-rejected in `docs/AUDIT.md` is a successful outcome for that
  task, not a failure.
- **`scripts/output_scorecard.py` must not regress.** It is the arbiter.
  A candidate that improves the openings check while lowering the overall
  count has not earned its place.
- **`preview.py` is unchanged.** Measurement scripts depend on it.
- Commit with explicit paths only. **NEVER `git add -A` or `git add .`** —
  `demo_plans/`, `demo_output/`, `.env_check/` and `.blender_check/` are
  intentionally untracked. `data/soni_residence/` is personal, git-ignored,
  and must never be committed or sent anywhere.
- Python is `.venv/Scripts/python.exe`; set `PYTHONPATH=.` for scripts.
- Corpus path: pass it as a **command-line argument**, never as a string
  literal inside `python -c`. Git Bash rewrites POSIX paths in arguments
  but not inside Python source, so a literal `"/tmp/claude/..."` resolves
  to a non-existent `C:\tmp\...` and silently scores zero plans. This cost
  a wasted run this session.

## File Structure

| File | Responsibility |
| --- | --- |
| `scripts/open_air_accuracy.py` | **new.** The sweep harness. Scores open-to-sky decisions against CubiCasa's OUTDOOR annotation over a corpus, and sweeps one candidate at a time. |
| `planto3d/segment.py` | gains `OUTDOOR_PROBABILITY_FLOOR` and `OUTDOOR_MAY_OVERRULE`, mirroring the existing window mechanism. |
| `planto3d/features.py` | gains neighbour propagation, if it earns its place. |
| `planto3d/extrude.py` | the geometry fix: gardens above ground level. |
| `tests/test_open_air.py` | **new.** Tests for the harness and any shipped candidate. |
| `tests/test_extrude.py` | extended for the geometry fix. |
| `docs/AUDIT.md` | the sweep tables and every rejected candidate. |

---

### Task 1: The sweep harness

Nothing here can be argued without it. Every later task is "run this, read
the table, decide".

**Files:**
- Create: `scripts/open_air_accuracy.py`
- Test: `tests/test_open_air.py`

**Interfaces:**
- Consumes: `planto3d.pipeline.extract`, `planto3d.segment.load_segmenter`,
  `planto3d.features.is_open_to_sky`, `planto3d.classes.OUTDOOR`.
- Produces:
  - `score_plan(image_path, svg_path, segmenter) -> PlanScore` where
    `PlanScore` is a dataclass with fields
    `plan: str, true_open_px: int, predicted_open_px: int, intersection_px: int, rooms: int, undecidable: int`.
  - `summarise(scores: list[PlanScore]) -> dict[str, float]` returning keys
    `"recall"`, `"precision"`, `"iou"`, `"plans"`, `"rooms"`, `"undecidable_share"`.

Ground truth is the OUTDOOR class in CubiCasa's `model.svg`, which
`training/` already parses — reuse that parser rather than writing a second
one. Find it with `grep -rn "svg" training/ | head`; if it yields nothing
usable, rasterise the SVG's OUTDOOR polygons directly and say so in the
docstring.

Score in **pixels, not rooms**. A room-level score hides the thing that
matters: a terrace found at half its true extent is a quarter of a roof
left on. `iou` is the headline.

- [ ] **Step 1: Write the failing test**

```python
def test_summarise_reports_iou_recall_and_precision():
    from scripts.open_air_accuracy import PlanScore, summarise

    scores = [
        PlanScore(plan="a", true_open_px=100, predicted_open_px=50,
                  intersection_px=40, rooms=10, undecidable=2),
        PlanScore(plan="b", true_open_px=100, predicted_open_px=200,
                  intersection_px=80, rooms=10, undecidable=3),
    ]
    result = summarise(scores)
    # Pooled over pixels, not averaged over plans: a large terrace and a
    # small one are not equally important to the roof.
    assert result["recall"] == 120 / 200
    assert result["precision"] == 120 / 250
    assert result["iou"] == 120 / (200 + 250 - 120)
    assert result["plans"] == 2
    assert result["undecidable_share"] == 5 / 20


def test_a_plan_with_no_outdoor_truth_does_not_divide_by_zero():
    from scripts.open_air_accuracy import PlanScore, summarise

    result = summarise([
        PlanScore(plan="a", true_open_px=0, predicted_open_px=0,
                  intersection_px=0, rooms=4, undecidable=1),
    ])
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0
    assert result["iou"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_open_air.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.open_air_accuracy'`.

- [ ] **Step 3: Implement the harness**

Write `scripts/open_air_accuracy.py` with the two functions above plus a
`main()` taking `root` and `--checkpoint`, printing a per-plan table and
the pooled summary. Follow `scripts/scale_accuracy.py` for the table
style — this project's scripts share a voice.

`summarise` guards every denominator with `max(total, 1)` and returns
`0.0` rather than raising when a corpus has no OUTDOOR truth at all.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_open_air.py -q`
Expected: PASS.

- [ ] **Step 5: Establish the baseline**

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/open_air_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
```

Record the pooled `iou`, `recall`, `precision` **in the task report**.
Every later task is measured against this run and no other.

- [ ] **Step 6: Commit**

```bash
git add scripts/open_air_accuracy.py tests/test_open_air.py
git commit -m "Score open-to-sky against the annotation, in pixels"
```

---

### Task 2: Candidate A — let OUTDOOR overrule the argmax

The single highest-value candidate, because `CATEGORY_FEATURES` makes
OUTDOOR very nearly the only route to open air on an unlabelled plan, and
because **the mechanism already exists in this file and is already
justified**.

`Segmenter.predict` takes a hard `argmax` over the softmax, then applies
one exception: `WINDOW_PROBABILITY_FLOOR = 0.25` lets WINDOW win over
`WINDOW_MAY_OVERRULE = (WALL, BACKGROUND)` when it clears the bar. The
reasoning recorded there is that a window pixel sits ringed by wall and
loses the average.

A terrace loses the same way — it is an unfurnished rectangle and ROOM is
the model's fallback for exactly that. A room whose OUTDOOR probability is
0.45 against ROOM's 0.50 currently becomes generic ROOM, `category` is set
to `""`, and `feature_for` returns `None`: the model's opinion is thrown
away by a hair.

**This is the "low-confidence types discarded by an all-or-nothing
threshold" candidate the spec names, and the argmax is that threshold.**

**Files:**
- Modify: `planto3d/segment.py:61-70` (beside the window constants)
- Modify: `planto3d/segment.py:140-145` (the `np.where` that applies them)
- Test: `tests/test_segment.py`

**Interfaces:**
- Produces: `OUTDOOR_PROBABILITY_FLOOR: float`, `OUTDOOR_MAY_OVERRULE: tuple[int, ...]`.

- [ ] **Step 1: Write the failing test**

```python
def test_outdoor_overrules_room_but_never_wall_or_a_named_type():
    # A terrace is an unfurnished rectangle and ROOM is the model's
    # fallback for exactly that, so OUTDOOR loses the argmax by a hair
    # and its opinion is discarded. It may take ROOM. It must not take
    # WALL -- that would dissolve the building -- nor a confidently
    # named type like BEDROOM, which is a different room, not an
    # unsure one.
    from planto3d import segment
    from planto3d.classes import BEDROOM, OUTDOOR, ROOM, WALL

    assert ROOM in segment.OUTDOOR_MAY_OVERRULE
    assert WALL not in segment.OUTDOOR_MAY_OVERRULE
    assert BEDROOM not in segment.OUTDOOR_MAY_OVERRULE
    assert OUTDOOR not in segment.OUTDOOR_MAY_OVERRULE
    # A floor above 0.5 could never fire (it would have won the argmax
    # against a single rival), and one at or below 0 fires everywhere.
    assert 0.0 < segment.OUTDOOR_PROBABILITY_FLOOR < 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_segment.py -q`
Expected: FAIL — `AttributeError: module 'planto3d.segment' has no attribute 'OUTDOOR_MAY_OVERRULE'`.

- [ ] **Step 3: Sweep the floor before choosing it**

**Do not pick a number and then measure it.** Sweep, exactly as the window
floor was swept. For each of `0.20, 0.25, 0.30, 0.35, 0.40, 0.45` run:

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/open_air_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
```

with the floor set to that value, and record `iou`, `recall`, `precision`
for each. Then run `scripts/output_scorecard.py` and
`scripts/wall_accuracy.py` at your best two candidates — a floor that wins
on open air and costs wall agreement has not earned its place.

Report the whole table, including the values that lost.

- [ ] **Step 4: Implement**

Add beside the window constants, with the sweep table in the comment:

```python
# Outdoor areas lose the argmax the same way windows do, and for a
# related reason: a terrace is an unfurnished rectangle, and ROOM is
# what the model falls back to for an unfurnished rectangle. A room at
# P(outdoor)=0.45 against P(room)=0.50 is the model saying "probably a
# terrace" and being overruled by a hair -- and because CATEGORY_FEATURES
# maps outdoor alone to open air, that hair decides whether a roof gets
# built over it.
#
# Swept over <N> plans, measured with scripts/open_air_accuracy.py:
#
#   floor    IoU     recall   precision
#   <fill in every row actually run, marking the chosen one>
#
OUTDOOR_PROBABILITY_FLOOR = <the swept value>

# ...and only over ROOM. Not WALL: a floor there would dissolve the
# building's structure into terrace. Not a named type either -- BEDROOM
# beating OUTDOOR is the model distinguishing two rooms, not hesitating.
# ROOM is the one class that means "enclosed area, no further opinion",
# which is the only case this floor is arguing about.
OUTDOOR_MAY_OVERRULE = (ROOM,)
```

Extend the existing `np.where` rather than adding a second pass, so the
two floors compose in one place:

```python
            outdoor = probabilities[OUTDOOR].cpu().numpy()
```

then after the window `np.where`:

```python
        predicted = np.where(
            (outdoor >= OUTDOOR_PROBABILITY_FLOOR)
            & np.isin(predicted, OUTDOOR_MAY_OVERRULE),
            OUTDOOR,
            predicted,
        ).astype(np.uint8)
```

Order matters and is deliberate: windows resolve first, so this cannot
turn a recovered window into terrace. Say so in a comment.

- [ ] **Step 5: Run the tests and the scorecard**

```bash
.venv/Scripts/python.exe -m pytest -q
PYTHONPATH=. .venv/Scripts/python.exe scripts/output_scorecard.py <corpus> --checkpoint models/unet_cubicasa.pt
```

Expected: suite green. Scorecard **must not fall** below 10 of 30.

- [ ] **Step 6: Commit, or record the rejection**

If it earned its place:

```bash
git add planto3d/segment.py tests/test_segment.py docs/AUDIT.md
git commit -m "Let a nearly-outdoor room be outdoor, as windows already may"
```

If it did not: revert the code, write the sweep table into `docs/AUDIT.md`
under the settled list with **"Do not retry this"**, and commit that. A
measured rejection is this task's success condition too.

---

### Task 3: Candidate B — parapet-height enclosure on a top storey

The spec's first candidate: *a room enclosed by parapet-height walls on a
top storey is open whatever it is called.*

Sequenced after Candidate A because it is strictly weaker evidence and
should be measured against a corpus A has already improved — otherwise it
takes credit for rooms A would have found anyway.

**Files:**
- Modify: `planto3d/extrude.py` (near `open_to_sky`, line 419)
- Test: `tests/test_extrude.py`

**Interfaces:**
- Produces: `enclosed_by_parapets(room, floor, scale) -> bool`.

- [ ] **Step 1: Establish whether the evidence exists at all**

Before writing anything, measure. Wall heights in this pipeline are a
built quantity, not a read one — `PARAPET_HEIGHT_FT` is applied *by*
`_open_air_walls` **because** a room was already known to be open. If
nothing upstream distinguishes a parapet from a wall on the drawing, this
candidate is circular and cannot work.

Read `_open_air_walls` and `PARAPET_HEIGHT_FT` in `planto3d/extrude.py`
and answer in the task report, with the line numbers: **is parapet height
an input or an output?**

If it is an output, **stop and record the candidate as impossible rather
than implementing it.** That is a legitimate result and it saves the whole
task. Write it into `docs/AUDIT.md` and move to Task 4.

- [ ] **Step 2: If it is an input, write the failing test**

```python
def test_a_top_storey_room_ringed_by_low_walls_reads_as_open():
    # The spec's rule: on the top storey, an enclosure that never rises
    # to full height is a parapet, and what a parapet surrounds is a
    # terrace whatever the drawing calls it.
    from planto3d.extrude import enclosed_by_parapets

    room = _room_ringed_by_walls(height_ft=3.5)
    assert enclosed_by_parapets(room, _top_floor(), scale=30.0) is True

    room = _room_ringed_by_walls(height_ft=9.0)
    assert enclosed_by_parapets(room, _top_floor(), scale=30.0) is False
```

Build `_room_ringed_by_walls` and `_top_floor` from the synthetic harness
already in `tests/test_extrude.py` — find it with
`grep -n "def _" tests/test_extrude.py | head -20` and reuse, do not
duplicate.

- [ ] **Step 3: Run to verify it fails, implement, run again**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extrude.py -q`

- [ ] **Step 4: Measure it, on top of Candidate A**

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/open_air_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
PYTHONPATH=. .venv/Scripts/python.exe scripts/output_scorecard.py <corpus> --checkpoint models/unet_cubicasa.pt
```

Report the delta **against Task 2's result**, not against Task 1's.

- [ ] **Step 5: Commit, or record the rejection**

Same rule as Task 2. Ship only what the numbers support.

---

### Task 4: Candidate C — neighbour propagation

The spec's third candidate: *a room whose neighbours are all outdoor
probably is too.*

Measured last and held to the highest bar, because it is the one that can
cascade: a wrong `open` propagates to its neighbours and takes a roof off
a bedroom. Precision matters more than recall here, and the harness
reports both.

**Files:**
- Modify: `planto3d/features.py` (beside `is_open_to_sky`, line 1061)
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `propagate_open(rooms: list, scale: float) -> list[bool]` —
  index-aligned with `rooms`, `True` where the room should be treated as
  open. Pure and side-effect free, so it can be tested without geometry.

- [ ] **Step 1: Write the failing test**

```python
def test_a_room_surrounded_entirely_by_open_rooms_becomes_open():
    from planto3d.features import propagate_open

    rooms = [_open_room(0), _open_room(1), _undecided_room(2)]
    # room 2 touches only rooms 0 and 1, both open
    assert propagate_open(rooms, scale=30.0)[2] is True


def test_one_open_neighbour_is_not_enough():
    # The cascade risk is the whole reason this is measured last. A
    # bedroom beside a balcony is a bedroom.
    from planto3d.features import propagate_open

    rooms = [_open_room(0), _indoor_room(1), _undecided_room(2)]
    assert propagate_open(rooms, scale=30.0)[2] is False


def test_propagation_does_not_chain():
    # Two passes would let one wrong terrace unroof a whole floor.
    # A newly-opened room is not evidence for its own neighbours.
    from planto3d.features import propagate_open

    rooms = [_open_room(0), _undecided_room(1), _undecided_room(2)]
    result = propagate_open(rooms, scale=30.0)
    assert result[2] is False
```

- [ ] **Step 2: Run to verify it fails, implement, run again**

Single pass only, computed from the input state — never iterate to a fixed
point. The third test guards exactly that and must stay.

- [ ] **Step 3: Measure, on top of whatever shipped**

Both scripts, delta against the previous task's numbers. **If precision
falls, it does not ship** regardless of what recall does: a roof wrongly
removed is more visible than one wrongly left on, and the scorecard's
openings check counts both.

- [ ] **Step 4: Commit, or record the rejection**

---

### Task 5: The geometry piece — gardens above ground level

Separable from the detection work and bounded: this is about what happens
*after* a room is correctly identified. The spec calls it out specifically
— "including gardens above ground level, not only at grade".

**Files:**
- Modify: `planto3d/extrude.py` (the per-floor landscaping path)
- Test: `tests/test_extrude.py`

- [ ] **Step 1: Find and demonstrate the defect**

`open_to_sky` collects `floor.planting` and the `GROUND_COVERS` labelled
regions. Determine, and state in the report with line numbers, whether
planting on a **non-ground** storey is built as planting or silently
dropped. `_storey_base_ft` and the `for index, floor in enumerate(floors)`
loop around line 1546 are the places to look.

Write a test that fails against current behaviour before changing
anything. If planting on an upper storey already works, **say so and skip
the task** rather than inventing a change — the spec's claim would then be
stale, and recording that is worth more than a no-op commit.

- [ ] **Step 2: Fix it, minimally**

- [ ] **Step 3: Verify by looking**

This is a geometry change and its failure mode is visual. Build a plan
with an upper-storey garden and render it:

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/demo.py --plan demo_plans/2-TWO-STOREY-stairs.gif --output .blender_check/garden
```

State plainly whether the garden appears at the right level. A passing
test is not sufficient evidence for this one.

- [ ] **Step 4: Full suite, scorecard, commit**

---

### Task 6: Record what happened

**Files:**
- Modify: `docs/AUDIT.md`, `README.md`

- [ ] **Step 1: Write the results up**

For each of the three candidates: what was swept, the table, whether it
shipped, and why. **The rejected ones matter as much as the accepted
ones** — this project's audit is unusually good precisely because it
records dead ends, and four such entries already saved work this session.

- [ ] **Step 2: Update the headline figures**

Re-run and record: `output_scorecard.py`, `open_air_accuracy.py`, and the
test count. Every number from a run in that session.

- [ ] **Step 3: Update the spec's status**

`docs/superpowers/specs/2026-09-02-plan-accurate-3d-design.md`'s "Where it
stands, measured" table carries figures this workstream changes. Update
the ones that moved; leave the rest.

- [ ] **Step 4: Commit**

---

## Self-review

**Spec coverage.** Workstream 3 asks for two separable pieces —
*evidence* (Tasks 2, 3, 4) and *geometry* (Task 5) — and requires the
candidates be swept rather than adopted, which Tasks 2–4 each do with an
explicit rejection branch. The spec's three named candidates map to Tasks
2, 3 and 4 in that order. Scoring is on the openings check, which Task 1
instruments and every later task re-runs.

**One risk stated plainly.** The spec's framing implies three roughly
comparable candidates. The baseline says otherwise: with labels at 3.4%
and `CATEGORY_FEATURES` mapping only `outdoor` to open air, Candidate A is
likely to carry nearly all the improvement, and B may be structurally
impossible — which is why Task 3 opens by testing that rather than
assuming it. If A lands and B and C are both rejected, this workstream is
still complete and correctly executed.

**The ceiling is real.** 25.6% of rooms have neither label nor type.
Nothing here can decide them, and the honest outcome may be an openings
check that improves without reaching the spec's gate. Task 6 records the
number either way.
