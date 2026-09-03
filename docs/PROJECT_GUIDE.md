# PlanTo3D — Project Guide

*Written as a verified handoff document. Every number below was either
produced by running the named script against `models/unet_cubicasa.pt` on
2026-08-28, or is explicitly marked as unverified this run. Where
`docs/STATE.md` and `docs/AUDIT.md` disagreed with each other or with the
code, that is called out rather than silently resolved — see
[Doc drift found while writing this](#doc-drift-found-while-writing-this)
at the end.*

---

## What this project is

PlanTo3D converts a 2D architectural floor plan — a PDF, PNG, JPEG, or GIF —
into a measured 3D model (`.glb`) of the building it depicts, plus six
rendered views (aerial, top, front, back, left, right). A U-Net segmenter
reads the drawing into a per-pixel class mask; classical computer-vision
code turns that mask into wall, room, door and window geometry; the
geometry is calibrated to a real-world scale (from the drawing's own
printed dimensions where possible, otherwise from standard element sizes);
and the calibrated geometry is extruded into a stacked, materialled 3D
model. A separate, explicitly unmeasured diffusion pass can dress the same
model in a photorealistic style. The project's central discipline is that
every claim about how well it works is backed by a script that can be
re-run and disagreed with — `docs/AUDIT.md` is the record of that
discipline, and this document is its current snapshot plus a map of the
code that produces it.

---

## Architecture

### Pipeline diagram

```
 2D floor plan (PDF / PNG / JPEG / GIF)
        │
        ▼
 ┌─────────────────┐   planto3d/ingest.py
 │   INGESTION      │   rasterize_pdf · read_image · split_sheet · crop_pages
 │                  │   - reads a light-on-dark ("reversed") print the right way up
 └────────┬─────────┘   - splits a sheet holding several plans; requires 2+ agreeing
          │              boundary lines OR a real gutter, and every piece must enclose space
          ▼
 ┌─────────────────┐   planto3d/segment.py (Segmenter) or planto3d/classical.py
 │  SEGMENTATION    │   U-Net + ResNet34 encoder → per-pixel class index (0-10)
 │  (swappable)     │   - windows get a lower softmax bar than the class they're
 └────────┬─────────┘     drowned in (WINDOW_PROBABILITY_FLOOR), restricted to
          │                overrule only WALL/BACKGROUND so it can't steal a door
          ▼
 ┌─────────────────┐   planto3d/extract.py
 │    GEOMETRY      │   wall_gauge → extract_walls → close_envelope → extract_openings
 │   EXTRACTION     │   → extract_footprint → extract_rooms (squared, steps collapsed)
 └────────┬─────────┘   Every threshold is a ratio of the drawing's own wall
          │              thickness (the "gauge"), so it is resolution-free.
          ▼
 ┌─────────────────┐   planto3d/label_rooms.py + planto3d/features.py
 │  ROOM FUNCTION   │   Room type comes from the SEGMENTER (11-class scheme:
 │                  │   bedroom/kitchen/bath/storage/circulation/outdoor), not OCR.
 └────────┬─────────┘   A printed label, when present, overrides the prediction.
          │              460 keywords / 15 categories in features.py map a label
          │              or predicted type to what it means downstream (railed
          │              edge, tiled floor, planted, sunk into the ground, etc).
          ▼
 ┌─────────────────┐   planto3d/calibrate.py
 │  SCALE           │   Chain: printed dimensions/area (gated against geometry)
 │  CALIBRATION     │   → doors → wall gauge → drafting-ratio assumption.
 └────────┬─────────┘   Reports scale_confident: True only when a printed size
          │              was used, or the two geometric estimates (door width,
          │              wall gauge) agree within MAX_SCALE_DISAGREEMENT.
          ▼
 ┌─────────────────┐   planto3d/site.py + planto3d/extrude.py
 │  3D EXTRUSION    │   Walls, slabs, roof (flat/domed/pitched/glazed), plinth,
 │                  │   stairs, railings (only where a floor actually drops),
 └────────┬─────────┘   window frames, canopies, ramps, water/void handling.
          │              open_to_sky() cuts a hole in the slab above a balcony/
          │              terrace/pool only where the storey above doesn't stand.
          ▼
 ┌─────────────────┐   planto3d/materials.py + planto3d/design.py + planto3d/style.py
 │  MATERIALS &     │   5 user choices (style/colour/time/landscaping/creativity)
 │  DESIGN          │   compose into a per-surface colour + a Lighting preset.
 └────────┬─────────┘
          │
          ▼
 ┌─────────────────┐   planto3d/preview.py
 │  RENDER          │   CPU rasterizer (no GPU/display needed): linear-light
 │                  │   shading, ACES-style filmic tonemap, ambient occlusion,
 └────────┬─────────┘   2x supersampling. Six standard views.
          │
          ▼
   .glb + 6 view PNGs + detection overlay   ═══════════ MEASURED RESULT ═══════════
          │
          │  (architecturally separate; never feeds back into the above)
          ▼
 ┌─────────────────┐   planto3d/photoreal.py
 │  PHOTOREAL PASS  │   Depth-conditioned Stable Diffusion + ControlNet.
 │  (optional)      │   Prompt built from what the pipeline actually read off
 └─────────────────┘   the drawing. NEVER SCORED — there is no ground truth
                        for what a house should look like.  ═══ IMPRESSION ═══
```

### Stage-by-stage file mapping

| Stage | Primary file | Key entry points |
| --- | --- | --- |
| Ingestion | `planto3d/ingest.py` (745 lines) | `rasterize_pdf`, `read_image` (auto-corrects reversed prints), `split_sheet`, `crop_pages`, `detect_drawing_region` |
| Segmentation | `planto3d/segment.py` (164 lines) | `Segmenter` class, `load_segmenter(checkpoint_path)` — `None` falls back to the classical baseline |
| Segmentation (baseline) | `planto3d/classical.py` (253 lines) | `classical_mask`, `wall_mask`, `room_mask`, `window_mask`, `vegetation_regions`, `refine_windows` (colour heuristic — **only used by the baseline now**, see Rejected) |
| Geometry extraction | `planto3d/extract.py` (928 lines) | `wall_gauge`, `extract_walls`, `close_envelope`, `extract_openings`, `extract_footprint`, `extract_rooms`, `_rectilinear`, `_collapse_steps` |
| Geometry types | `planto3d/geometry_types.py` (188 lines) | `Wall`, `Room`, `Opening`, `FloorPlan` dataclasses |
| Class scheme | `planto3d/classes.py` (72 lines) | 11 class indices, `CLASS_NAMES`, `ROOM_CLASSES`, `WET_CLASSES` |
| Room labelling | `planto3d/label_rooms.py` (67 lines) | `assign_labels` — OCR label wins over predicted type when both exist |
| Feature vocabulary | `planto3d/features.py` (1068 lines) | `classify`, `finish_for`, `regions_from_labels`, `group_by_feature`, `is_open_to_sky` — 460 keywords / 15 categories |
| Scale calibration | `planto3d/calibrate.py` (637 lines) | `parse_dimension_text`, `parse_area_text`, `estimate_scale`, `scale_from_doors`, `scale_from_gauge`, `assumed_scale`, `corroborated`, `read_text_boxes` (now upscales sub-1200px sheets before OCR), `scale_from_known_room` (exact scale from a user-measured room, gates `--scale-room`), `detect_convention`, `element_sizes`, `CONVENTIONS` (per-tradition door/wall sizes; fires on 0/30 CubiCasa sheets, see `docs/AUDIT.md`) |
| Site / outdoor | `planto3d/site.py` (143 lines) | `classify_cover`, `outdoor_rooms`, `site_outline`, `boundary_walls` |
| 3D extrusion | `planto3d/extrude.py` (1931 lines — largest file) | `floors_to_parts` (top-level), `slab_mesh`, `_wall_parts`, `open_to_sky`, `_railing_parts` (`_guarded_edges`), `_stair_parts`, roof-form builders |
| Materials | `planto3d/materials.py` (182 lines) | `Surface` dataclass, `build_scene`, `export_scene` |
| Design choices | `planto3d/design.py` (236 lines) | `Design`, `Tone`, `Landscaping`, `apply_tone` |
| Lighting/palette | `planto3d/style.py` (261 lines) | `Lighting` (rebalanced defaults today), `Palette`, `parse_colour` |
| Rendering | `planto3d/preview.py` (669 lines) | `render`, `render_depth`, `render_glb`, `render_views` |
| Photoreal | `planto3d/photoreal.py` (401 lines) | `build_prompt`, `build_negative_prompt`, `edge_guide`, `build_guides` |
| Orchestration | `planto3d/pipeline.py` (688 lines) | Three entry points — `extract()` (geometry + labels, up to the correction pause point), `build()` (corrections applied, geometry to a model), `run()` (`extract()` then `build()` with no pause, for callers with no correction step); `PipelineResult`, `FloorResult` dataclasses |
| Room corrections | `planto3d/corrections.py` (67 lines) | `apply_room_corrections`, `CATEGORY_LABELS` — turns a user's UI override into a `Room.label` change, applied in place between `extract()` and `build()` |
| CubiCasa5K reader | `planto3d/cubicasa.py` (350 lines) | `svg_to_mask`, `sample_paths`, `ground_truth_scale`, `parse_feet` |
| CVC-FP reader | `planto3d/cvc_fp.py` (168 lines, **new today**) | `svg_to_mask`, `sample_paths`, `annotation_size` |
| Misc tooling | `planto3d/tools.py` (88 lines) | `poppler_bin_dir`, `tesseract_exe`, `configure_tesseract` |
| Training loop | `training/train.py` (295 lines) | `build_model`, `class_weights`, `build_loss`, `train`, `evaluate`, `main` |
| Training dataset | `training/dataset.py` (118 lines) | `CubiCasaDataset` |
| Augmentation | `training/augment.py` (222 lines) | `rotate`, `flip`, `rescale`, `exposure`, `compress`, `blur`, `unfill_walls` (**new, not yet trained in**), `augment` |
| Training metrics | `training/metrics.py` (66 lines) | `dice_score`, `iou_score`, `per_class_iou` |

### Key design decisions

**Why the segmenter is swappable.** `load_segmenter(checkpoint_path)` returns
either a `Segmenter` (the trained U-Net) or, when `checkpoint_path is None`,
`planto3d.classical.classical_mask` — a threshold-and-morphology baseline.
Both conform to the same "callable that takes an image and returns a class
mask" contract, so every downstream stage (extraction, calibration,
extrusion) is written against that contract and never imports torch
directly. This is why `planto3d/classes.py`'s docstring can say geometry
code can name classes "without importing the segmentation wrapper, which
pulls in torch" — it's a genuine dependency-isolation boundary, verified by
grep: torch is imported nowhere in `planto3d/` except inside two methods of
`segment.py`'s `Segmenter` class (deliberately deferred until a checkpoint
is actually loaded or run — that file's own top-of-file comment says so),
plus at module level in three files under `training/`.

**Why room type comes from the model, not OCR.** `planto3d/classes.py`'s own
docstring states the reason precisely and it is corroborated by a live
measurement made today (`batch_evaluate.py`, 30 CubiCasa plans): printed
room names were read on only **6 of 29** reconstructed plans; the predicted
type was available on **29 of 29**. The 11-class scheme (indices 0-4
unchanged from an earlier 5-class scheme, so old checkpoints still load) was
built specifically so room function does not depend on text existing on the
sheet at all.

**Why the photoreal pass is architecturally separated from the measured
pipeline.** This was a documented decision made explicitly today (commit
`b7f28cd`, "The photoreal image is an impression, and the audit said it was
measured"). The reason is structural, not stylistic: there is no ground
truth for what a house *should* look like, so nothing about the diffusion
output can be scored the way wall coverage or scale error can be. Before
this change, `docs/AUDIT.md`'s "Complete and measured" table listed the
photoreal pass with "runs on a T4" in the *measured* column — that was
identified as dishonest and removed. `notebooks/run_on_colab.ipynb` now
opens its photoreal section with "Nothing below is measured, and it is not
the output of the pipeline," and the two artefacts (the `.glb`+views vs the
photoreal PNG) are packaged and labelled separately in the notebook's final
zip step. The two failure modes hide each other in opposite directions — a
convincing photoreal image says nothing about whether the storey count is
right, and an ugly one says nothing about whether it's wrong — which is
why conflating them in one "does it work" judgement was actively
misleading.

**Data representation.** A class mask is a `np.int64` array (`(H, W)`,
values 0-10). Geometry is a list of dataclasses (`Wall(start, end,
thickness)`, `Room(polygon, category, label)`, `Opening(wall_id, position,
width)`) in the mask's own pixel coordinates — nothing is converted to real
units until `extrude.py`'s `_to_metres`, gated by the calibrated `scale`
(px/ft). Every geometric threshold in `extract.py` (minimum wall length,
merge gap, simplification tolerance, opening search distance) is expressed
as a *ratio* of `wall_gauge(mask)` — the drawing's own measured wall
thickness — rather than an absolute pixel count. This is why the pipeline
is resolution-independent: `docs/AUDIT.md` records scale error flat at
8-16% across a 4x range of input sizes, which was a deliberate rewrite
(commit-era "the pipeline no longer depends on the drawing's resolution")
verified by `tests/test_resolution.py` (11 tests).

**Training setup.** `training/train.py:build_model` wraps
`segmentation_models_pytorch.Unet` with a ResNet34 encoder,
ImageNet-pretrained. Loss is cross-entropy weighted by inverse
class-frequency (`class_weights`, ceiling 25.0) — necessary because a
window is 0.11% of a training image and an unweighted loss has no
incentive to predict it at all. `training/augment.py` applies six
image-level transforms (quarter-turn rotation only — the geometry stage
can't use anything else — flip, rescale-crop, exposure, JPEG compression,
blur) plus, as of today, `unfill_walls`, which redraws a filled wall as a
hollow outline using the ground-truth mask to find it. The installed
checkpoint (`models/unet_cubicasa.pt`) was **not** trained with
`unfill_walls` — see [Gaps](#gaps-ranked-worst-first) and [What has been
tried and rejected](#what-has-been-tried-and-rejected).

---

## What's achieved so far

Every number in this section was reproduced live on 2026-08-28 by running
the named script against `models/unet_cubicasa.pt` (11 classes, epoch 22,
val Dice 0.7757) over the CubiCasa5K sample at
`C:\Users\RAHULS~1\AppData\Local\Temp\claude\cubicasa_batch` (60 sheets)
unless marked otherwise. Where a figure differs from what's currently
written in `docs/AUDIT.md`, that's noted inline — but every one matched
exactly on this run.

### Per-class segmentation accuracy — `scripts/class_accuracy.py`, 30 sheets

Pixel-pooled IoU (the figure quoted throughout the project; a friendlier
per-sheet-median average also exists and is intentionally *not* the quoted
one — see the script's own docstring):

| class | share of page | IoU (pooled) | recall |
| --- | --- | --- | --- |
| background | 39.40% | 0.961 | 97.0% |
| outdoor | 5.77% | 0.883 | 95.9% |
| kitchen | 6.11% | 0.778 | 85.5% |
| circulation | 4.64% | 0.733 | 88.5% |
| room (untyped) | 19.34% | 0.717 | 76.9% |
| bedroom | 9.38% | 0.713 | 89.2% |
| **wall** | 8.50% | **0.697** | 86.1% |
| bath | 3.94% | 0.602 | 67.1% |
| storage | 4.25% | 0.570 | 77.4% |
| door | 0.62% | 0.560 | 85.2% |
| **window** | **0.11%** | **0.089** | 50.9% |

Median across classes: 0.713.

### Geometry — `scripts/wall_accuracy.py`, 30 sheets

| | median | mean |
| --- | --- | --- |
| Coverage — annotated wall that gets built | **96.6%** | 93.3% |
| Agreement — built wall that really is wall | **92.2%** | 88.1% |

0 of 30 plans below 70% coverage; 4 of 30 below 70% agreement.

### Scale — `scripts/scale_accuracy.py`, 48 plans

Median error **17.3%**, worst 44.6%, **33/48 within a fifth of true.**
Split by source: doors (n=30) 12.3% median, **−8.4% bias**; walls (n=18)
20.2% median, **−20.1% bias**.

The bias is now understood, not just measured (`docs/AUDIT.md`, "Most of
the scale error is not a mistake"): CubiCasa's *annotated ground truth*,
converted to real feet using its own recorded scale, shows a Finnish
interior door is a median 2.28 ft (2'3") against the code's assumed 2.50 ft
(2'6") — predicting a −8.8% bias against the measured −8.4%. A Finnish wall
is 0.63 ft (7.6") against an assumed 0.75 ft (9"), predicting −15.4%
against the measured −20.1%. **The door bias is ~97% explained by the
constant being wrong for this population, not by the vision failing to
read the door correctly.**

### Splitting a sheet with several plans — `scripts/split_accuracy.py`, 60 sheets

Exact floor count **58/60 (97%)**. Treating it as "does this sheet hold
more than one plan": precision **12/12 (100%)**, recall 12/14 (86%).

**Note:** `docs/AUDIT.md`'s own gaps table (row 3, line 131) still quotes
this as "57/60" — that was the number after the paired-boundary-cut fix but
*before* a later fix (the enclosure-share check, which stops a legend or
title block being read as an extra storey) that took it to 58/60 — see
[Doc drift](#doc-drift-found-while-writing-this).

### End-to-end scorecard — `scripts/output_scorecard.py`, 30 plans

This is the newest and, structurally, the most important measurement: every
other script above scores one *stage*. This one runs the full pipeline
per plan and asks how many plans are correct on **every** check at once
(built / right storey count / scale within a fifth / room count within 25%
/ wall coverage+agreement / opening count within 0.6x-1.5x of drawn),
scored against annotations, not by eye.

**10/30 (33%) pass every check.** Per-check failure counts, worst first:

| check | fails on |
| --- | --- |
| size (scale) | **10/30** |
| openings | 9/30 |
| walls | 8/30 |
| rooms | 4/30 |
| built | 0/30 |
| storeys | 0/30 |

This inverts the priority order that the per-stage numbers alone would
suggest. Window IoU (0.089) looks like the single worst number in the
project, but it costs the *finished output* only 9 plans of 30 (openings
check) — while scale, whose stage-level error (17.3%) looks more modest,
is the single largest cause of an end-to-end failure. **Nothing crashes
and nothing splits wrongly at this scale, so the entire 67% failure rate
is accuracy, not robustness.**

Three faults were found and fixed *in the scorecard itself* before its
first number could be trusted — documented in `docs/AUDIT.md` under "Score
the finished model, not each stage of it" and worth knowing before adding
a fourth check: (1) counting annotated rooms over the union of room
classes collapsed an open-plan floor into one region; (2) the openings
check was originally a floor ("at least as many as drawn"), which a
plan could pass by *over*-reporting, rewarding the exact failure mode it
should catch; (3) wall coverage was originally painted onto the *original
sheet's* annotation even for split sheets, which compares nothing once a
sheet is split into separately-framed pieces. All three fixed; the six
current thresholds (`SCALE_TOLERANCE=0.20`, `ROOM_COUNT_TOLERANCE=0.25`,
`MIN_WALL_COVERAGE=0.85`, `MIN_WALL_AGREEMENT=0.80`,
`MIN_OPENING_RATIO=0.6`, `MAX_OPENING_RATIO=1.5`) are explicitly flagged in
the script's own comment as first guesses, not derived quantities.

### A second drafting tradition — CVC-FP, 30 sheets

**No tracked script measures this yet** — `planto3d/cvc_fp.py` (the
loader) is committed and tested, but the measurement in `docs/AUDIT.md`
was produced by an ad-hoc script during a session, not by anything in
`scripts/`. Re-run live today (reusing the committed loader +
`wall_accuracy.score`) to confirm it still holds — **it matches exactly**:

| class | IoU | recall |
| --- | --- | --- |
| background | 0.950 | 95.1% |
| wall | 0.636 | 88.1% |
| room | 0.530 | 55.8% |
| door | 0.136 | 15.6% |
| window | **0.239** | 27.4% |
| outdoor | 0.003 | 4.8% |

Wall coverage 96.7%, agreement 77.5%, 30/30 sheets yield ≥8 walls.

**Walls generalise almost perfectly** (coverage 96.7% vs 96.6% on
CubiCasa). **Windows are nearly 3x better** on CVC-FP (0.239 vs 0.089),
which is strong evidence the window problem is substantially a property of
CubiCasa's training distribution (windows there are drawn at 0.11% of a
page; CVC-FP draws them at a much larger share) rather than of the network.
**The door collapse (0.560 → 0.136) is not a door failure** — verified by
measuring depth-perpendicular-to-wall: CubiCasa annotates the door *leaf*
(median 0.79 wall-thicknesses deep, aspect ratio 4.6 — a thin strip);
CVC-FP annotates the *swing arc* (median 3.19 deep, aspect 1.1 — nearly
square). Two annotation conventions meaning different things by "door";
the number says nothing about whether the model finds doors.

**What CVC-FP cannot tell you:** it carries no metric ground truth
anywhere in its 122 annotations, so it cannot judge scale — the single
largest end-to-end failure. It also has no room-type labels (everything is
generic "Room") and no floor grouping.

### A real-world third source — BRIDGE, 60 sheets (unverifiable by script)

`data/bridge/` holds a 60-plan sample of ~2,400 real Indian/US listing
plans. **This corpus has no structural ground truth at all** — its XML is
Pascal VOC bounding boxes for furniture symbols and free-text region
captions ("master bedroom has a double bed, sofa"), so nothing about wall,
room, door or window accuracy can be scored against it. Its value was
finding real bugs by contact with real drawings, which it did — verified
live today, not from memory:

- Every one of the 60 GIF files crashed before today's fix (`.gif` was not
  in `IMAGE_SUFFIXES`, so every file was routed to the PDF rasterizer,
  which failed with "unable to get page count"). **Re-run live: 15/15
  sampled plans now reconstruct with 0 crashes**; `.gif` confirmed present
  in `planto3d.pipeline.IMAGE_SUFFIXES`.
- `12'-6" x 13'-8"` (hyphen between feet and inches) and `14'` (bare feet,
  no inches) both failed to parse before today. Confirmed fixed:
  `parse_dimension_text` now handles both, verified against 9 hand-built
  cases including these exact strings.
- OCR has a resolution floor these sheets sit under (median width ~600px);
  `read_text_boxes` now upscales anything under 1200px on its long edge,
  capped at 2x (going to 3x measured no further gain). End-to-end effect
  on the 60-sheet sample: 7 plans now source scale from printed dimensions
  where 0 did before; plans reporting `scale_confident=True` rose from 19
  to 24.

### Tests — `pytest`, full suite

**835 passed, 0 failed, at the time this section was written.** This
matched `docs/AUDIT.md`'s figure of that day exactly. `README.md` and
`docs/AUDIT.md` both said 748 as of the last commit before that
(`3f674ca`, "docs: the test count again, 748") — both already correct at
the time of that audit; the count had grown to 835 with the
room-correction work and the photoreal/open-air design and
splitter-validation work that followed. It has since grown again, to
**850**, with the `scale-accuracy` branch's per-source error reporting,
`--scale-room` and per-tradition element sizes — `pytest -q` is the source
of truth, not this paragraph. `docs/STATE.md` line 104 still says
"384 tests" — see [Doc drift](#doc-drift-found-while-writing-this).

Composition (40 test files, 6,627 lines): the largest suites are
`test_roof_forms.py` (30 tests), `test_windows_railings.py` (17),
`test_style.py` (16), `test_stairs.py` (16), `test_site.py` (20). 6 test
files were touched by today's commits (`test_calibrate.py`,
`test_scale_fallback.py`, `test_open_to_sky.py`, `test_split_sheet.py`,
`test_augment.py`, `test_image_input.py`); the other 31 predate today's
session and were spot-checked (docstrings + counts) rather than re-read in
full, given the size of the suite.

### Other measurements re-run live

- **`scripts/convention_stress.py`, 15 sheets, 9 simulated conventions**:
  matched `docs/AUDIT.md` exactly (as-drawn 0.747 wall IoU; hatched walls
  0.688; **outline walls 0.534** — the worst simulated convention;
  reversed print 0.014 raw / 4-of-15 reconstruct, which is the
  *un-corrected* number since this script bypasses `read_image`'s fix by
  design, documented in the script itself).
- **`scripts/batch_evaluate.py`, 30 plans**: 29/30 meet the ≥8-wall/≥3-room
  "usable geometry" bar (`MIN_WALLS=8`, `MIN_ROOMS=3` in the script), 0/30
  crashed. This is a *different, stricter* bar than
  `output_scorecard.py`'s "built" check (which only requires a
  `model_path`), so "29/30 reconstructed" here and "0/30 built failures"
  in the scorecard are both true and not in conflict — just two different
  thresholds for two different questions.
- **Reversed-print fix, end-to-end**: verified live by writing a reversed
  copy of a real CubiCasa sheet to disk and running it through
  `read_image` → `run()`. Median grey came back at 252 (correctly
  un-reversed) and the plan reconstructed with 21 walls / 3 rooms — not
  reproduced from the audit's own figures, but independently re-derived.

---

## Gaps, ranked worst first

Ranked by the **end-to-end scorecard's per-check failure counts**
(`output_scorecard.py`, verified above), not by stage-level metrics — the
scorecard's whole purpose was to correct for the fact that stage metrics
and finished-output correctness don't rank the same way.

1. **Scale — 10/30 end-to-end failures, the single largest cause.**
   Stage-level: 17.3% median error, 33/48 within a fifth. **Mostly
   NOT general, and mostly NOT a vision problem** — see [Structurally
   unfixable vs. unfinished](#structurally-unfixable-vs-unfinished). It is
   a convention-mismatch problem (the code assumes 2'6" doors and 9"
   walls; CubiCasa's real numbers are 2'3" and 7.6"), which no amount of
   segmentation improvement touches. What *is* general and unfinished:
   reading a drawing's own printed dimensions/area, which today's BRIDGE
   work improved (OCR upscaling, hyphen/bare-feet parsing) but which is
   still gated conservatively against the (biased) geometric estimate.

2. **Openings — 9/30 end-to-end failures.** Window detection: 62.1%
   recall at 43.5% precision (measured via an in-session harness, not a
   tracked script — see below). **Largely a property of the training
   corpus, not the network**: CVC-FP's window IoU (0.239) is 2.7x
   CubiCasa's (0.089) with the *same weights*, and windows are 0.11% of a
   CubiCasa training image. General in the sense that any CubiCasa-trained
   checkpoint will show this; specific in the sense that a differently
   composed training set would likely close much of the gap without any
   architecture change.

3. **Walls — 8/30 end-to-end failures.** Stage-level coverage/agreement
   (96.6%/92.2%) look strong; the end-to-end check is stricter
   (`MIN_WALL_COVERAGE=0.85` AND `MIN_WALL_AGREEMENT=0.80` simultaneously,
   only judged on unsplit sheets). **General and largely already
   addressed** — CVC-FP shows coverage holds on an unseen convention
   (96.7%); the remaining 4/30 plans below 70% agreement on CubiCasa are
   concentrated on specific sheets (`12787`, `10711`, `11578`, `11855` —
   named in the wall_accuracy.py worst-list above), not spread evenly.

4. **Rooms — 4/30 end-to-end failures.** Least of the four measured
   checks. Room-type IoU (bath 0.602, storage 0.570) looked serious in
   isolation but the scorecard shows it barely touches the finished
   output — a room read as the wrong *type* is a wrong floor finish, not
   a structurally wrong building. **General**, low priority.

5. **Only 3½ drafting conventions ever tested** (CubiCasa, CVC-FP, one
   Indian villa control, one two-plan sheet — BRIDGE doesn't count toward
   this since it can't be scored). This is the ceiling on every number
   above: none of them can be claimed to hold on a convention not yet
   seen. **General, and the single biggest qualification on this entire
   document** — stated explicitly at the top of `docs/AUDIT.md`.

6. **`unfill_walls` augmentation exists but was never trained in.** The
   installed checkpoint has not seen it. A retrain *was* done today
   (epoch 20, val Dice 0.7726) and **was rejected** — see [Tried and
   rejected](#what-has-been-tried-and-rejected). This is "unfinished"
   only in the sense that a cheaper variant (lower `unfill` probability
   than the 0.3 that was tried) hasn't been tried; the full-strength
   version is a closed question, not an open one.

7. **No tracked script reproduces the CVC-FP or window-detection numbers.**
   Both were measured with in-session throwaway code. This is a real gap
   in the audit's own reproducibility discipline, worth closing before
   anyone relies on those figures without re-deriving them.

---

## What has been tried and rejected

Every one of these was measured, not guessed, and the measurement that
killed each idea is named so it is not re-attempted without new evidence.

| Idea | Measured result | Why it failed |
| --- | --- | --- |
| **Train at 768px instead of 512px** | Windows +0.016 IoU; room and bath *worse*; downstream flat-to-worse (fewer openings found, worse scale) | Resolution was not the constraint; more pixels per class doesn't help a class that's 0.11% of the image if the loss/architecture can't use them |
| **Find windows as gaps in extracted wall geometry** | Of 75 missed windows, 62 sit inside *solid* predicted wall (not a gap at all); ceiling estimated at +3.2% recall at best | The model doesn't leave a gap where a window is — it paints over it as wall. There's no gap-finding to do |
| **Combine the two scale estimates** (doors + wall gauge) into one number via averaging, min, or max | Every blend scored 24/30 within a fifth against 25/30 for doors alone | Averaging two biased-in-different-directions estimates doesn't beat picking the less-biased one; the *disagreement* between them is useful (as a confidence signal) but the *combination* isn't |
| **Chamfer/snap near-miss wall junctions** (a proposed fix for room-closure failures) | Explicitly measured and found not to help (see `docs/AUDIT.md`, "Chamfering open junctions was tried, and there were none to close") | — |
| **Colour-based window/planting detection** (`refine_windows` in `classical.py`) applied to the *trained model's* output | Actively harmful — measured to be **deleting correctly-detected windows** and replacing them with worse colour-heuristic detections on some plans | Removed from the trained-model code path entirely today (commit `2fea61a`) — verified: `pipeline.py` no longer imports or calls it, only a comment there explains why it once did. Remains in `classical.py` for the *classical baseline only* |
| **Retrain with `unfill_walls` augmentation at probability 0.3** | Epoch 20 checkpoint, val Dice 0.7726 (vs installed 0.7757) — **measured and set aside**, not installed | See `docs/AUDIT.md`, "The outline-wall augmentation worked, and cost more than it bought" — it improved outline-drawn walls specifically but the net effect on measured stages was negative or flat everywhere else tested |
| **Diagonal wall recovery** | Diagonal walls cost only 2.7% of total wall pixels across the sample tested; no plan lost more than 10% | Not worth the added complexity for the measured benefit — explicitly deprioritised, `docs/AUDIT.md` "Diagonal walls are not worth recovering" |

---

## Structurally unfixable vs. unfinished

This distinction matters because it changes what "closing a gap" even
means.

**Structurally unfixable by code alone — needs data or a different
question:**

- **Scale, for the ~20% of the gap explained by wall-gauge bias, and most
  of the gap for plans with no printed dimensions.** The measurement is
  precise: a Finnish wall really is 7.6", not the assumed 9". No amount of
  better wall-*segmentation* changes that the constant is wrong for this
  population. Fix (a), reading the drawing's own stated convention and
  switching constants accordingly, has now been **tried and shipped as a
  dormant mechanism**: `planto3d/calibrate.py` gained `CONVENTIONS`,
  `detect_convention` and `element_sizes`, but it fires on 0 of 30 CubiCasa
  sheets because those rasters carry almost no readable room-name text —
  the mechanism is correct and unit-tested, it simply has nothing to
  detect on this corpus. See `docs/AUDIT.md`, "Scale error, broken down by
  source, and four routes tried that did not close it" for the numbers,
  including why correcting the door constant made things *worse* rather
  than better. The remaining structural fix is (b), more training data
  from the target population, which changes what "trained on CubiCasa"
  even means.
- **The 3½-conventions ceiling.** This is a data-acquisition problem, not
  a code problem. No measurement methodology closes it; only more
  ground-truthed corpora from different traditions do.
- **CVC-FP's inability to validate scale or room type**, ever, on its own
  — it's a property of that corpus's annotation, not something this
  project's code can work around.

**Unfinished, and there is a known next step:**

- **Windows.** The corpus-composition hypothesis (CubiCasa under-represents
  windows) is now well evidenced by the CVC-FP comparison, but untested by
  actually rebalancing or supplementing CubiCasa's training data — that's
  a concrete, doable next step, distinct from "make the model bigger" (
  tried, rejected) or "post-process gaps" (tried, rejected).
- **A tracked script for the CVC-FP measurement and for window
  detection-as-opposed-to-IoU.** Both numbers exist and are correct
  (re-verified above) but only as throwaway code. Committing them is
  mechanical, not risky.
- **Gating the printed-dimension scale source less conservatively**, now
  that OCR reads more of BRIDGE-style sheets than it used to. This is a
  tuning question with a clear measurement path (`scale_accuracy.py`
  before/after), not a research question.
- **A lower-`unfill` retrain.** The full-strength version was tried and
  rejected; a milder one (say, `unfill=0.1`) hasn't been, and is a single
  parameter change plus a ~2.5-hour GPU run away from an answer.

---

## Future plan (priority order)

1. **Reproducibility debt first, before anything else.** Commit
   `scripts/cvc_fp_accuracy.py` (a thin wrapper around the exact
   measurement re-run above) and a window-detection script (recall/
   precision as computed today, not IoU) so the two most-cited numbers
   outside CubiCasa stop depending on someone re-deriving throwaway code.
   Near-zero risk, an afternoon of work, and it's the difference between
   an audit that can be trusted and one that has to be taken on faith for
   two of its headline claims.

2. **Fix the internal doc drift** (see next section) — update
   `docs/AUDIT.md`'s gap-table row 3 (57→58/60) and rows 5-6
   (storage/bath IoU: 0.525/0.586 → the current 0.570/0.602), and either
   refresh or clearly re-header `docs/STATE.md`'s stale sections (lines
   85-770 predate today's session almost entirely and contain
   recommendations — e.g. "improve `refine_windows`" — that the code has
   since done the opposite of). Cheap, and prevents someone acting on
   superseded advice.

3. ~~Read the drawing's own stated convention and switch scale constants
   accordingly.~~ **Done, on a `scale-accuracy` branch.** The mechanism
   (`CONVENTIONS`, `detect_convention`, `element_sizes` in
   `planto3d/calibrate.py`, keyed on room-label language) exists, is
   unit-tested, and is dormant — it fires on 0 of 30 CubiCasa sheets, since
   those rasters carry almost no readable text. No measured gain on this
   corpus follows from that; see `docs/AUDIT.md`, "Scale error, broken
   down by source, and four routes tried that did not close it" for the
   numbers and for why the door half of the fix was tried and rejected.
   The remaining value here is data-side (a corpus that actually carries
   the room-label text this mechanism reads), not code-side.

4. **Rebalance or supplement window training data**, now that the
   corpus-composition hypothesis has real cross-corpus evidence behind it.
   This is a genuine open experiment (unlike the four already-rejected
   window ideas above) — worth scoping as: does adding CVC-FP-style
   window-dense samples to the training mix move CubiCasa-native window
   IoU, without regressing everything the 768px/unfill retrains
   regressed?

5. **A milder `unfill_walls` retrain** (e.g. probability 0.1 rather than
   0.3), to see whether the outline-wall gain is separable from the
   regression on everything else. Cheap to test relative to its
   information value, since the full-strength result is already known.

6. **Acquire a fourth ground-truthed corpus** from a population whose
   scale constants are known to differ (the Indian villa control already
   shows +6% wall bias vs CubiCasa's −20%, i.e. real doors/walls in that
   population are *larger* than assumed, not smaller) — this is the only
   way to move past "3½ conventions" as a qualification on every number in
   the project, and it's a data-acquisition task rather than a code task.

---

## Repo layout reference

```
planto3d/            The pipeline package. See file mapping table above.
training/            Dataset, augmentation, loss/metrics, and the training loop
                      for the segmenter. Kept separate from planto3d/ so geometry
                      work doesn't require torch installed (pyproject.toml's
                      `ml` extra is optional).
scripts/              16 command-line entry points — see table below.
tests/                42 files, 850 tests, 6,749 lines. pytest, PYTHONPATH=. required.
notebooks/            5 Colab notebooks:
                        train_on_colab.ipynb   — trains the segmenter (27 cells)
                        run_on_colab.ipynb     — upload a plan, get a house (21 cells)
                        design_on_colab.ipynb  — the 5-choice design flow (19 cells)
                        photoreal_on_colab.ipynb — the diffusion pass alone (16 cells)
docs/
  AUDIT.md            The measurement record. Authoritative for numbers — every
                       figure names the script that produced it. 1,259 lines.
  STATE.md            A narrative handoff note. Its "Start here" section (top ~45
                       lines) is current-ish but already one session stale as of
                       this writing (says "738 tests", recommends a retrain that
                       has since happened and been rejected, and lists "a second
                       corpus" as unstarted when two now exist). Everything below
                       that header (lines ~85-770) predates today's session almost
                       entirely and should be read as history, not current status
                       — its own header says so.
  PROJECT_GUIDE.md    This file.
  images/pipeline.png The current README hero image (CubiCasa sheet → detection
                       overlay → model, nothing hand-corrected).
  superpowers/        The ORIGINAL day-1 design spec and implementation plan
                       (dated 2026-08-17). Historically useful for intent, but
                       describes an architecture since substantially changed —
                       e.g. it says room labels "come from OCR text, never from
                       the segmentation model's class output," which is the
                       exact opposite of the current, deliberate design. Do not
                       treat as current.
data/                 Gitignored. Not shipped with the repo.
                        cubicasa5k/      not present in this checkout; the
                                         verification run used a pre-extracted
                                         60-sheet sample at a session temp path
                        cvc_fp/          122 CVC-FP plans, present, used above
                        bridge/          60-plan BRIDGE sample, present, used above
                        soni_residence/  one personal reference PDF, present but
                                         excluded from git for privacy
models/                Gitignored except a README. unet_cubicasa.pt (93MB, 11
                       classes, epoch 22, val Dice 0.7757) present and used for
                       every measurement above.
pyproject.toml        Core deps: opencv-python, numpy, pillow, pdf2image,
                       pytesseract, trimesh. `ml` extra: torch,
                       segmentation-models-pytorch. `dev` extra: pytest.
                       Python >=3.11 (running 3.13.2 here).
README.md              User-facing overview. 346 lines. Verified consistent
                       with the numbers above at time of writing.
```

### Scripts reference

| Script | What it does |
| --- | --- |
| `batch_evaluate.py` | Run the whole pipeline over many plans and report where it breaks |
| `class_accuracy.py` | Per-class accuracy against CubiCasa's own annotations |
| `class_balance.py` | Measure how much of a drawing each class occupies |
| `convention_stress.py` | How much does a change of drafting convention cost the segmenter? |
| `crop_pages.py` | Rasterize the floor plan PDF and crop every page to the shared drawing area |
| `generalisation_test.py` | Measure what generalises beyond the drawings the pipeline was tuned on |
| `output_scorecard.py` | Is the finished model right? Not "is each stage right" — the end-to-end check |
| `random_plan.py` | Pick a plan at random and run the whole pipeline on it |
| `run_pipeline.py` | Build a 3D model from a floor plan (the main CLI entry point) |
| `scale_accuracy.py` | Score the inferred scale against the sizes CubiCasa recorded |
| `split_accuracy.py` | Score sheet splitting against the floor count CubiCasa records |
| `wall_accuracy.py` | Do the extracted walls land where the drawing puts its walls? |

---

## Doc drift found while writing this

Everything below was found by actually running code or grepping the repo,
not by comparing prose. Severity is about impact on someone trusting the
document, not about how hard it was to find.

1. **`docs/STATE.md` line 104: "384 tests."** Actual: 748 (verified live).
   This is inside a status table (`## Done and working`) that is
   otherwise entirely superseded — it also describes the 5-class→11-class
   room-type retrain as a pending next step, when the installed checkpoint
   has had it for a long time (verified: `class_accuracy.py` output above
   shows bedroom/kitchen/bath/storage/circulation/outdoor rows).
   **Severity: low on its own, high in context** — someone reading only
   this table would think room-type prediction is unfinished. It isn't.

2. **`docs/STATE.md`'s "three things asked for next" (lines 626-674) is
   fully superseded and, in one place, gives advice the codebase has since
   done the opposite of.** It recommends: "give `refine_windows` a way to
   tell a sheet that marks windows in colour from one that does not,
   rather than assuming." Today's session found `refine_windows` was
   *actively deleting correctly-detected windows* when applied to the
   trained model's output and **removed it from that code path entirely**
   (verified: `grep refine_windows planto3d/pipeline.py` returns nothing;
   it survives only in `classical.py` for the baseline). Someone following
   STATE.md's advice here would be improving code that was since deleted
   for being harmful. **Severity: high** — this is advice, not just a
   number, and it points the wrong way.

3. **`docs/STATE.md`'s own "Start here" header (top of the file, written
   only one day before this audit) is already one session stale.** It says
   "Tests: 738" (now 748), lists "a retrain is queued and prepared" as
   item 1 of "what is worth doing next" (the retrain has since happened
   *and been rejected* — see `docs/AUDIT.md` "The outline-wall
   augmentation worked, and cost more than it bought"), and lists "a
   second drafting corpus is the real unknown" as item 3 (two have since
   been acquired: CVC-FP and BRIDGE). **Severity: medium** — this is the
   *most* current part of STATE.md and it's still behind; a reader has no
   way to know that without cross-checking against AUDIT.md and the git
   log, which is exactly what this document now does for them.

4. **`docs/AUDIT.md`'s own gap-table (line 131) says splitting is
   "57/60"; the live-verified, current number is 58/60.** Two other places
   in the *same file* (lines 1103, 1147) already say 58/60 / the correct
   sequence of fixes. This is an internal inconsistency, not a
   STATE-vs-AUDIT one: the gap table simply wasn't touched by the later
   commit that fixed the legend/title-block-as-extra-storey bug.
   **Severity: low** — off by one sheet, but worth a two-line fix since
   the whole point of the gaps table is to be the reliable top-level
   summary.

5. **`docs/AUDIT.md`'s own gap-table (lines 133-134) quotes storage IoU
   0.525 and bath IoU 0.586.** Live re-verification via
   `class_accuracy.py` today gives **storage 0.570, bath 0.602** — and
   these newer numbers are *already* quoted correctly elsewhere in the
   same file (line 64). The gap-table values appear to be left over from
   before the window-probability-floor change (which had knock-on effects
   on other classes; the file's own "before/after" table for that change,
   lines 283 and 289, shows storage 0.525→0.544 and bath 0.586→0.523 —
   neither of which matches the current 0.570/0.602 either, meaning at
   least three different values for these two classes have been true at
   different points and only one — 0.570/0.602 — matches what the
   *currently installed* checkpoint actually produces). **Severity:
   medium** — a genuine numeric error in a table meant to be the
   project's summary of its own weaknesses, now corrected in this
   document.

6. **No contradiction, but a subtlety worth flagging so it isn't mistaken
   for one:** `batch_evaluate.py` reports "29/30 reconstructed" while
   `output_scorecard.py` reports "0/30 built failures" on overlapping
   samples. These are different, both-correct measurements —
   `batch_evaluate.py`'s "reconstructed" requires ≥8 walls AND ≥3 rooms
   (`MIN_WALLS`/`MIN_ROOMS` in that script); `output_scorecard.py`'s
   "built" only requires a `.glb` to have been produced at all. Verified
   by reading both scripts' source rather than assuming a conflict.

7. **CVC-FP's numbers in `docs/AUDIT.md` are correct but not
   independently re-runnable** by anyone else without rebuilding the
   measurement from scratch, since no file in `scripts/` does it. Not a
   drift (the numbers matched exactly on re-run), but a reproducibility
   gap worth closing — listed as priority 1 in the future plan above.
   **Update:** `scripts/cvc_fp_accuracy.py` now exists and closes this gap
   — running it does not reproduce every number it inherited (see the
   discrepancy notes added to `docs/AUDIT.md`), but the corpus is
   independently re-runnable now, which is what this entry was about.

No other numeric claim checked in this pass (per-class IoU, wall
coverage/agreement, scale error and bias, convention-stress table, feature
vocabulary count, test count in README/AUDIT, the reversed-print fix, the
BRIDGE `.gif`/dimension-parsing/OCR-upscaling fixes) showed any
discrepancy between what's documented and what the code currently does
when actually run.
