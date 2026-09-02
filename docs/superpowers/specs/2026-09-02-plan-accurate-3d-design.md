# PlanTo3D — a plan-accurate 3D model, rendered so you can trust it

*Design spec, 2026-09-02. Supersedes the scope in
`PlanTo3D_Accuracy_and_Deploy_Plan.md` for everything after its Phase 2.*

## What this is for

One sentence: **any 2D floor plan in, a 3D model out that is correct
enough to build from, in a render good enough to show.**

Everything below serves that and nothing else. There is no interface —
the project is a pipeline and a set of command-line entry points. There
is no deployment story: the four Colab notebooks already let anyone run
it on a free GPU, and packaging a pipeline that is right on a quarter of
its inputs would be solving the wrong problem.

## What "accurate" means, exactly

A floor plan states some things and is silent about others, and no amount
of engineering recovers what was never drawn. Being explicit about the
line is the difference between a claim that survives scrutiny and one
that does not.

**Must be right — the drawing states it:**

- overall size in real feet
- wall positions and thicknesses
- door and window positions along those walls
- room layout, room count, and the storey split of a multi-plan sheet
- which spaces are open to the sky and which are roofed

**Assumed — the drawing is silent, so the model uses a stated default:**

- ceiling height (a parameter, default 9.0 ft)
- sill and head heights (`SILL_HEIGHT_FT` 3.0, `OPENING_HEAD_FT` 7.0)
- parapet, chimney, tower and tank heights
- roof pitch, and every material and finish

The honest claim this supports is **"a correct floor plan, in three
dimensions"** — not "a photograph of the finished house". An architect
reading the model gets the right building at the right size with the
right openings; the third dimension is a convention, clearly labelled,
and overridable.

Recovering the vertical dimension properly would mean reading a second
drawing — an elevation or a section — and matching it to the plan. That
is a second reader roughly as large as the one that exists. It is out of
scope, and saying so is not a weakness.

## Where it stands, measured

All figures produced by running the project's own scripts this session.

| | Measured | Instrument |
| --- | --- | --- |
| **Correct on every count, end to end** | **5 of 20 (25%)** | `output_scorecard.py` |
| — failures by check | size 7, walls 7, openings 6, rooms 2 | same |
| Scale error | 17.3% median; 33/48 within a fifth | `scale_accuracy.py` |
| Wall coverage / agreement | 96.6% / 92.2% | `wall_accuracy.py` |
| Sheet splitting | 58/60 exact, 100% precision, 86% recall | `split_accuracy.py` |
| Window IoU | 0.089 CubiCasa, **0.239 CVC-FP** | `class_accuracy.py` |
| Windows as a share of annotated pixels | **0.1019%** | `class_balance.py` |
| Rooms ending open to the sky | 26 of 171 (15%) across 16 plans | this session |
| Rooms with neither label nor predicted type | **51 of 171 (30%)** | this session |
| Tests | 835 passing | `pytest` |

Two of those deserve reading twice. **Scale is the largest single
failure** — 7 of 20 sheets — and it is not a vision problem: CubiCasa's
real door is 2'3" where the code assumes 2'6", and its real wall 7.6"
where the code assumes 9". That mismatch alone predicts −8.8% and −15.4%
against measured errors of −8.4% and −20.1%. The drawings are being read
correctly and then measured against the wrong standards.

And **30% of rooms carry no evidence at all** — no OCR label, no
predicted category. Those rooms cannot be open to the sky under any rule,
because nothing in the pipeline knows what they are. That is the ceiling
on the open-to-air problem, not the geometry.

## The four workstreams

### 1. Scale from the drawing's own conventions

The biggest win and the cheapest. Today `scale_from_doors` assumes a
2'6" door and `scale_from_gauge` a 9" wall, worldwide. Neither assumption
holds across drafting traditions, and the residual error is dominated by
that rather than by misreading.

The approach is to let the drawing say which tradition it belongs to and
switch constants accordingly. The signals are already on the page and
partly already parsed: printed units (`m²` versus `sq ft`), the
feet-and-inches notation style, and the language of the room labels.
`calibrate.py` already reads printed dimensions and areas and gates them
against the geometric estimate; this extends the gate rather than
replacing it.

Paired with a direct override — one known real measurement typed in,
which recalibrates everything — for the case where no convention is
detected. That override is the permanent floor under this problem: it
works on any plan from any tradition, needs no detection, and cannot be
wrong if the user reads their own drawing correctly.

### 2. A renderer that is beautiful *and* evidence

Today there are two renderers and each fails half the requirement.
`preview.py` is a hand-written rasterizer: accurate, deterministic,
plain. The diffusion pass is beautiful and invents — by construction, for
every plan, which is why the audit never scores it.

A third path, `planto3d/blender_render.py`, driven headless by Blender's
`bpy`, fed directly from `extrude.py`'s geometry and `materials.py`'s
seventeen surfaces. Real materials, soft shadows, global illumination,
and no generative step anywhere in it. The output is simultaneously the
deliverable and the evidence: every surface in the image is a surface the
drawing supports.

This is a genuinely new subsystem and the largest single piece of work
here. It is also a **debugging instrument** — a sealed balcony or a
missing window is obvious at a glance under real lighting and invisible
in flat grey massing. Everything sequenced after it gets easier to see.

The existing rasterizer stays. It is fast, dependency-light, and used by
the measurement scripts; nothing about it should change.

### 3. Open to the air

The mechanism works — a correction now propagates through to geometry,
verified across three drafting conventions. The detection does not:
15% of rooms end up open, and 30% of rooms have no evidence to decide
from.

Two separable pieces. **Geometry:** fix whatever `open_to_sky` and the
per-floor landscaping logic get wrong once a room *is* correctly
identified — including gardens above ground level, not only at grade.
This is bounded and testable against the synthetic harness that already
exists. **Evidence:** give the 30% of rooms that currently decide nothing
a way to be decided. The candidates, in order of how much they are worth
against what they cost: a room enclosed by parapet-height walls on a top
storey is open whatever it is called; a room the segmenter typed at low
confidence still carries a signal the current all-or-nothing threshold
discards; and a room whose neighbours are all outdoor probably is too.
Which of these earns its place is a measurement, not a guess, and the
plan for this workstream must sweep them rather than adopt them.

Scored on `output_scorecard.py`'s openings check, which fails on 6 of 20
sheets today.

### 4. Windows — a bounded experiment, not an open commitment

Windows are the weakest component and the evidence about why is
unusually clear. The same weights score 0.089 on CubiCasa and **0.239 on
CVC-FP** — 2.7× better on a corpus never trained on. Windows are 0.1019%
of CubiCasa's annotated pixels. The problem is the training data, not the
network.

It matters that **the cheap remedies are already spent.** `train.py`
already weights classes by inverse *square root* frequency with a ceiling
of 25, normalised to mean one, on top of Dice plus weighted
cross-entropy — and the reasoning for each of those choices is written
down. Higher training resolution was tried and measured to fail. Finding
windows as gaps in wall geometry was tried and measured to fail. There is
no configuration change left that has not been examined.

So the only untried lever is more and better-annotated windows, which is
manual work with an uncertain payoff. It is therefore scoped as a
**spike, not a commitment**: build annotation tooling, annotate 20–30
plans, retrain, measure with `window_detection_accuracy.py`, and decide
on the number. A negative result is a real result and gets recorded like
the others.

The annotation tool must pre-seed each plan with the model's current
predictions so the work is *correcting* boxes rather than drawing them
from nothing. That single decision is most of the difference between a
tolerable afternoon and an intolerable week.

## Sequencing, and why this order

**Scale → Renderer → Open-to-air → Windows spike.**

Scale first because it is the cheapest large win and banks real
improvement before anything risky starts — a bad schedule still leaves
the project meaningfully better than today.

The renderer second, not last. It delivers the second half of the goal,
and because it makes geometric faults visible at a glance, every accuracy
task after it becomes easier to diagnose. Leaving it until the end is the
common mistake: it is the newest subsystem, so it is the one that gets
squeezed, and the project finishes accurate but unshowable.

Open-to-air third, once there is a renderer good enough to see the
difference.

The windows spike last, deliberately. It is the only item whose outcome
is genuinely unknown, so it is the one that must be cuttable without
disturbing anything else.

Rough shape: **9–13 working days** — scale 2, renderer 4, open-to-air 3,
windows spike 2–3, plus the validation below.

## One spec, four plans

These four workstreams share a goal and a pair of acceptance gates, but
they touch different code and fail in different ways. Each gets its own
implementation plan, written when its turn comes rather than all now —
the renderer's plan in particular will be much better written after
scale has landed and the measurement instruments have been re-run.

Take them in the order above. Nothing here needs to be built twice if
the order is respected.

## Done means this

Two gates, both re-runnable, neither arguable.

**Gate 1 — population.** On the CubiCasa sample:

- `output_scorecard.py` reaches **14 of 20 or better** (from 5)
- `scale_accuracy.py` median error below **10%** (from 17.3%)
- `split_accuracy.py` does not regress from 58/60, 100% precision
- `wall_accuracy.py` coverage does not fall below 96%

**Gate 2 — reality.** The reference house in `data/soni_residence`,
whose plot and room dimensions can be physically checked, reconstructs
within **5%** on overall size and within **10%** on individual rooms,
with terraces, balconies and parking rendering open rather than sealed.
One building, but the only ground truth here that was not drawn by an
annotator.

Gate 1 proves it generalises. Gate 2 proves it is true of a real house.
Neither alone is sufficient.

## Deliberately not doing

- **A user interface**, of any kind. Command-line entry points only.
- **Deployment, containers, hosting.** The Colab notebooks are the
  distribution story and `pyproject.toml` already pins the dependencies.
- **Reading elevations or sections** to recover real heights. It is the
  correct fix for full 3D fidelity and it is a second project.
- **Acquiring a new annotated corpus.** Calendar time that improves a
  claim rather than the product. The Spanish and Portuguese vocabulary
  work is kept — it is a day, and the failing baseline test already
  exists.
- **Re-examining anything in the audit's settled list**: 768px training,
  windows-as-gaps, blending the two scale estimates, chamfering wall
  junctions, colour-based window detection, diagonal-wall recovery.
  Each was measured and each failed.
- **Tuning the diffusion pass toward accuracy.** It is an impression by
  construction. Workstream 2 is the answer to wanting a trustworthy
  image, not better prompting.

## Risks, honestly

**Blender is the schedule risk.** It is a new dependency, headless `bpy`
is awkward to install, and 4 days is an estimate not a measurement. If it
slips, the fallback is polishing `preview.py`'s lighting — visibly
better, materially cheaper, and a lower ceiling.

**The windows spike may return nothing.** That is why it is a spike. The
outcome to plan around is "measured, documented, unchanged".

**Convention detection could overfit to the corpora at hand.** The
project's own standing rule is not to fit a constant to one dataset. The
manual calibration override exists so the feature can be conservative:
detect only what is unambiguous, and let the user settle the rest.

**Gate 1's target is a judgement, not a derivation.** 14 of 20 is roughly
"most plans come out right". If scale and open-to-air both land and the
number reaches 11 or 12, that is a large improvement and stopping there
is a legitimate call to make on the evidence rather than a failure.
