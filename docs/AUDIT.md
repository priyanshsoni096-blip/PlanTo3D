# Where every part of the project stands

An honest stage-by-stage audit. Every figure here is measured, and the
source of each measurement is named so it can be re-run and disagreed
with.

**What it is measured on.** 60 CubiCasa plans (Finnish apartments), 122
CVC-FP plans in four further styles, one Indian villa used as a control,
and one web-sized sheet carrying two plans. Call it **three and a half
drafting conventions**, and it remains the single biggest qualification on
everything below -- CVC-FP records no scale, so it cannot judge the
largest failure there is.

```bash
python scripts/output_scorecard.py <corpus> --checkpoint models/unet_cubicasa.pt
python scripts/class_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
python scripts/wall_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
python scripts/scale_accuracy.py <corpus> --checkpoint models/unet_cubicasa.pt
python scripts/split_accuracy.py <corpus>
python scripts/convention_stress.py <corpus> --checkpoint models/unet_cubicasa.pt
python scripts/batch_evaluate.py <corpus> --checkpoint models/unet_cubicasa.pt --limit 60
```

## Complete and measured

| Stage | State | Measured |
| --- | --- | --- |
| Ingestion — PDF, PNG, JPEG, folders | Complete | 60/60 read, 0 crashes |
| Multi-storey reconstruction | Complete | 60/60 build a model |
| Room type prediction | Complete | **60/60 plans**, every room typed |
| Wall extraction | Complete | Median 20 walls/plan |
| Room outlines square | Complete | **92.4%** square perimeter; 100% on a clean mask |
| Resolution independence | Complete | Scale error flat 8–16% across a 4× size range |
| Storey placement | Complete | Every feature builds on the storey it was drawn on |
| Open-to-sky handling | Complete | Balconies, terraces, pools, courtyards — one rule |
| Roof forms | Complete | Dome, pitched, glazed, tank, chimney, tower, canopy, ramp |
| Feature vocabulary | Complete | 460 keywords, 15 categories, no duplicates |
| Materials and design choices | Complete | 5 user choices, 12 style×tone combinations |
| Renderer | Complete | Tonal spread 76, saturation 35 — see the daylight section |
| Notebooks | Complete | `train_on_colab`, `run_on_colab` |
| Tests | **912 passing** | — |

## What the finished model gets right, end to end

Every other measurement here scores one stage. All of them can pass and
still leave a model nobody would accept, because a plan 80% right on six
separate things is not 80% of a house. `scripts/output_scorecard.py` runs
plans end to end and asks how many come out right on **every** count at
once, scored against the annotations rather than by eye.

Over 60 plans from CubiCasa's held-out **test** split, which the checkpoint
never saw: **27 of 60 (45%)**. The list is `data/cubicasa_test60.txt`; see
"The benchmark was mostly training data" below for why this replaced the
earlier sample.

| Check | Fails on | |
| --- | --- | --- |
| **size** — scale within a fifth of true | **16 of 60** | the largest single cause |
| **openings** — within 0.6x to 1.5x of those drawn | 11 of 60 | |
| **walls** — coverage ≥85% and agreement ≥80% | 10 of 60 | |
| **rooms** — count within 25% of annotated | 8 of 60 | |
| storeys — right number of them | 3 of 60 | |
| built — a model comes out at all | **0 of 60** | |

On the earlier sample the same script gave 10 of 30 (33%). That figure is
kept in the history below but should not be quoted.

## The benchmark was mostly training data

Every CubiCasa figure in this file up to 2026-09-13 was measured on a
60-sheet sample kept in a session temp folder,
`C:\Users\RAHULS~1\AppData\Local\Temp\claude\cubicasa_batch`. That folder was
cleared, and nothing recorded which sheets it held. Of the 32 sheet IDs that
can still be recovered from saved script output, checked against
`cubicasa5k/train.txt`, `val.txt` and `test.txt`:

| split | sheets | what it means for this checkpoint |
| --- | --- | --- |
| train | **25** | `training/train.py:201` trains on these |
| val | 6 | `training/train.py:203` scores epochs on these; the best is kept |
| test | 1 | never used |

So the old numbers were measured almost entirely on drawings the model had
learned from. The replacement is 60 sheets drawn from `test.txt` only — 20
per category, `random.Random(20260913)`, listed in
`data/cubicasa_test60.txt` so it can be rebuilt from the dataset archive
into the git-ignored `data/cubicasa5k/`. Re-measured on 2026-09-13:

| Measure | Old sample (mostly train) | **Held-out test, 60** | Script |
| --- | --- | --- | --- |
| Right on every check | 10 of 30 (33%) | **27 of 60 (45%)** | `output_scorecard.py --limit 60` |
| Scale, median error | 17.7% | **12.9%** (44/60 within a fifth) | `scale_accuracy.py --limit 60` |
| — door route | 10 plans, 9.8% | 38 plans, 8.7%, bias −1.9% | same |
| — wall route | 14 plans, 20.2% | 22 plans, 16.7%, bias −14.3% | same |
| Wall coverage / agreement, median | 96.6% / 92.2% | **97.6% / 93.8%** | `wall_accuracy.py --limit 60` |
| Sheets split exactly | 58/60, 86% recall | **57/60**, 100% precision, 73% recall | `split_accuracy.py --limit 60` |
| Open-air IoU / recall / precision | 80.5 / 96.3 / 83.0 (48 sheets) | **75.4 / 89.3 / 82.9** (52 sheets) | `open_air_accuracy.py` |

Two things moved the way contamination predicts: open-air recall fell 7
points while precision held, and split recall fell. Two moved the other way,
and the reason is the mix of sheets rather than the model: the held-out
sample happens to carry usable doors on 38 of 60 sheets, so more plans take
the accurate door route, and the scorecard and scale both improve. Walls are
as good on unseen drawings as on seen ones. Neither sample is large enough to
treat a few points as a trend; the held-out one is the one to quote.

## Why plans still fall back to the wall estimate

Size is the largest end-to-end failure on the held-out plans, 16 of 60, and
the wall route is where it comes from: 22 plans take it, at 16.7% median
error against the door route's 8.7%. The question was why those 22 do not
use doors. Measured per plan on the same 60 test sheets, comparing detected
openings against the doors CubiCasa's annotation draws:

| on the 22 wall-route plans | median |
| --- | --- |
| doors the annotation draws | **6.5** |
| doors the segmenter detects | **2** |
| detected doors inside the width band | 1.5 |

- **16 of 22 have fewer than 3 detected doors before any filter runs.** The
  width band decides only 6 plans, and every door it removes there is a
  sliver measuring 0.4 to 0.9 ft when the annotated doors on those sheets
  measure 2.0 to 3.3 ft. Those rejections are right.
- **The door constant is not the problem.** Annotated doors on these sheets
  run about 2.5 ft, which is the 2'6" assumed.
- **The minimum door count is not the problem.** Swept at 3, 2 and 1: median
  error 12.9%, 13.5%, 13.1%. Lowering it routes more plans through doors,
  but those plans' doors are the fragmentary ones and the door route itself
  worsens from 8.7% to 12.0%. It stays at 3, with the sweep in its comment.

The failure is **door recall in the segmenter**, on roughly a third of
sheets. Nothing downstream can recover a door that was never detected, so
the remedy is training — more or better-weighted door examples — and not
another rule. **Do not retry** widening the band, changing the door
constant, or lowering the minimum count.

## Window correction tooling, built; the retrain is not run

The spec's fourth workstream is a spike: correct 20–30 plans' window
annotations, retrain, measure. The tooling for it exists and is tested; the
correction and the retrain have not been done, because both need a person —
the boxes need correcting by hand, and a GPU retrain needs explicit
agreement before it starts.

- `scripts/export_windows.py` writes the model's WINDOW pixels as LabelMe
  rectangles beside each image. It refuses `test.txt` and
  `data/cubicasa_test60.txt`, and never overwrites an existing file.
- `training/dataset.py` applies a label file through
  `planto3d/window_labels.adjust_mask` **only if it is ticked reviewed**,
  returning removed windows to wall because CubiCasa paints windows over
  walls.

Smoke-tested on three training plans on 2026-09-13: 3 label files
written (plan 2564 with 20 predicted windows, 6165 with 18); a re-run wrote
0 and kept 3; the test list was
refused with exit 1; on plan 6044 the training mask held 78 window pixels at
512 px from the annotation, still 78 after an unreviewed edit, and 2 once
the edit was ticked reviewed. The corrections reach a Colab run as a zip:
`export_windows.py --bundle` packed the one reviewed file and left the two
unreviewed ones behind, and `window_labels.unpack_bundle` put it back beside
the same plan in a fresh folder, where the dataset found it reviewed.
Section 3b of `notebooks/train_on_colab.ipynb` unpacks the bundle from Drive
and stops if any held-out test plan carries a corrected label. The retrain itself is measured with
`scripts/window_detection_accuracy.py` and `scripts/class_accuracy.py` on
the held-out plans, and ships only if windows improve without walls or doors
regressing.

This reorders the work. The gaps table above ranks windows first and scale
second on stage metrics; end to end it is **scale first**. Room function,
which looked like a major problem at bath 0.602 and storage 0.570, costs
only 4 plans of 30 -- a room read as the wrong kind is a wrong floor
finish, not a wrong building.

Nothing crashes and nothing is split wrongly, which is worth saying: the
33% is entirely accuracy, not robustness.

### Three faults in the scorecard itself, found and fixed before trusting it

The first version reported **0 of 12** and would have been bad news from a
bad ruler:

- Annotated rooms were counted over the *union* of the room classes, so an
  open-plan floor collapsed into one component and a four-room flat scored
  as two. Counted per class, as `extract_rooms` does, room failures fell
  from 7 in 12 to 1.
- The openings check was a floor rather than a band, so a plan passed by
  over-reporting -- which is the failure it should catch, given window
  precision is 43%.
- The wall check painted every storey's walls onto the original sheet's
  annotation. On a split sheet each piece has its own origin, so that
  compares nothing; it is now judged only on sheets that stayed whole.

The thresholds are still first guesses and are marked as such in the
script. They decide the headline, so they deserve more scepticism than the
figures they produce.

## The photoreal pass is not part of the result

It used to sit in the table above, under a heading that says "complete and
measured", with "runs on a T4" in the measurement column. Running is not a
measurement. It has been taken out, because leaving it there was the one
place this document was quietly dishonest.

**It has never been scored against anything, and it cannot be.** There is
no ground truth for what a house should look like. Everything else here is
checked against CubiCasa's annotations; a diffusion render has nothing to
be checked against.

That matters more than it sounds, because the two failure modes point in
opposite directions and each hides the other:

- A convincing image **does not** mean the geometry is right. Diffusion
  will happily dress a model with the wrong number of storeys and make it
  look like an architect's visualization.
- An awkward image **does not** mean the geometry is wrong. It is equally
  capable of making a correct model look strange.

So the two are now shipped and judged separately. The model, its six views
and the detection overlays are the **result**: measured, and traceable to
a line on the drawing. The photoreal image is an **impression**: unmeasured
and partly invented, useful for showing someone what a building could look
like and useless as evidence that the pipeline read the plan correctly.

`notebooks/run_on_colab.ipynb` says so at the top of both sections, and the
zip it produces separates them.

This is the risk that was flagged when the end-to-end scorecard was
proposed and it went unaddressed for a while: judging the pipeline by its
prettiest artefact meant the geometry was never really being looked at.

## A third renderer, and it is the answer to the paragraph above

`planto3d/blender_render.py` drives Blender Cycles headlessly through
`bpy`. It sits next to `preview.py` (fast, numpy, plain) and
`photoreal.py` (the diffusion pass above, which invents by construction)
as a third option, and it is the only one of the three where the picture
is also the proof: every surface it paints is one the drawing supports,
because there is no generative step anywhere in the path. That is the
direct answer to the previous section -- the photoreal image can never be
scored because nothing says what it should look like, and this renderer
exists so a plan can still get a convincing image that *is* checkable,
because it was built from the same geometry everything else in this
document is checked against.

**Cycles against EEVEE: 7.7 s versus 57.3 s**, for the same 640×480
frame, measured twice in both engine orders so neither run got the
warm-cache advantage. The gap looks backwards -- Cycles is usually the
slower, better-looking option -- but headless changes what each engine
actually is: EEVEE is a rasteriser that wants a GL context, and with no
display and no GPU it falls back to software; Cycles is a CPU path
tracer and never needed a context in the first place. There is no
trade-off to weigh here, because Cycles also renders the better image.

Samples are close to free: doubling 32 to 64 on that same frame costs
**1.8 s** (7.7 s to 9.5 s), so the default sits above the floor rather
than on it and there is little reason to render at the floor.

The real cost is the six-view set a model actually ships with, at the
resolution it actually ships at:

| Resolution | Samples | Six views |
| --- | --- | --- |
| 800×600 | 64 | **4 m 33 s** |
| 1200×900 (default) | 64 | **5 m 19 s** |

Both measured end to end with `blender_render.render_views` against the
same model, not extrapolated from the single-frame figures above.
Resolution alone does not explain the gap either way: 1200×900 has
2.25× the pixels of 800×600 but only costs 1.16× the time, because a
fixed per-frame overhead -- glTF re-import, scene teardown and rebuild,
material assignment -- is paid six times regardless of resolution and
dominates at these sizes more than the pixel count does.

**Determinism.** Rendered the same view twice, same model, same
arguments, into separate files, and hashed the output:

```
a 0d903750345e656c
b e624dc0943e81a32
IDENTICAL: False
```

The hashes disagree, but not because the picture changed. Every one of
the 20 differing bytes sits inside PNG `tEXt` metadata chunks Blender
writes into the file -- `tEXtDate`, `tEXtTime`, `RenderTime`, and two
Cycles timing chunks -- which carry the wall-clock time the frame was
rendered and necessarily differ between two runs two seconds apart. The
pixel data itself, decoded and compared channel by channel, is
byte-identical: mean absolute difference **0.0**, max absolute
difference **0**, across all 120,000 pixels. Cycles' own path tracing is
therefore fully deterministic here -- same seed, same thread count, same
result down to the last bit of the image -- and the only thing that
varies run to run is a timestamp nobody looks at. Two renders of the
same view are the same evidence twice.

**`bpy` is 659 MB**, which is why it lives behind its own `render` extra
(`pip install -e ".[render]"`) rather than in the base install --
`preview.py` needs nothing but numpy, and most runs of the pipeline
should not have to pull two-thirds of a gigabyte to get a picture out.

**All 24 surfaces `materials.py` can produce map to a physical shader**,
not the 13 one demo plan's `.glb` happened to contain -- see the
derivation in `blender_render.SURFACE_SHADERS`, built from
`materials.SURFACES` rather than hand-copied, so a surface added to one
file cannot silently fall back to plaster in the other. Colour is left
alone: the `.glb` already carries the palette the user chose through
`design.py`, and the shaders here set only metallic, roughness and
transmission, never a colour that would override it.

Two things this renderer does honestly rather than hides:

- At the standard `aerial` view (45° elevation, the angle `preview.py`
  also uses) **no sky is visible at all**. The camera pitch exceeds the
  lens's vertical half-FOV at that elevation, so the frame sits entirely
  below the horizon. Measured rather than reasoned: a bare world at
  elevation 45, dusk preset, comes back a uniform **(138, 99, 80)** at
  every ramp position tried, top row and mid-frame alike -- warm, flat,
  no gradient anywhere in it. The gradient itself is correct and does
  show at the elevation-0 views: same world at elevation 0 gives
  **(9, 32, 61)** at the top of frame against **(129, 98, 80)** a
  quarter of the way down. So this is a framing consequence of an angle
  shared with the fast renderer, not a bug in the sky, and no choice of
  `SKY_TOP_POSITION` can change it.
- The blue is only in the top of an elevation-0 frame, not overhead
  across it: at the shipped `SKY_TOP_POSITION = 0.22`, **12%** of the
  frame's rows read blue (mean B greater than mean R), the rest warming
  towards the horizon glow. That is what a dusk sky seen through a 30°
  vertical field of view pointed at a building actually looks like; the
  alternative -- pushing the constant lower to get more blue -- clips
  the transition into a flat band at the top of frame instead.
- Dusk renders close to monochrome. The model's own palette is beige and
  the dusk key light is warm, so the two multiply into a render with
  little colour separation. That is the light and the building agreeing
  with each other, not a defect -- but it is a real limitation of the
  preset, not just of this one house.

## Gaps, worst first

| # | Gap | Measured | Why it matters | General? |
| --- | --- | --- | --- | --- |
| 1 | **Windows weak** | Detection **62.1%** at 43.5% precision; IoU 0.089 | Façades sparser than the drawing, plus openings that are not there. Openings fail on 9 of 30 plans end to end | Partly — CVC-FP reads **0.239**, so it is largely a property of CubiCasa |
| 2 | **Scale, 17.3% median error** | 33/48 within a fifth; **doors 10.4% error / +0.9% bias, walls 20.2% error / −20.2% bias** (30 sheets) | Sets the whole building's size, and is the **largest end-to-end failure** at 10 of 30. The error is concentrated in the wall-derived half of the population — doors are already accurate | No — wall thickness genuinely varies (IQR ±16% of median); no single constant repairs it. See below |
| 3 | Sheet splitting misses | Recall **86%**, **58/60** exact | A missed split reconstructs several plans as one flat building, confidently. All five failures now diagnosed -- three distinct modes, below | **Yes** |
| 4 | **Only 2½ conventions tested** | Now **3½** — CVC-FP added, 122 sheets, 4 styles | Walls hold at 96.7% coverage on an unseen tradition; scale still untestable there | **Yes** |
| 5 | Storage rooms weak | IoU 0.570 | Storage reads as ordinary rooms | Yes |
| 6 | Bath rooms weak | IoU 0.602 | Wet floors missed | Yes |
| 7 | ~~Prompt truncated~~ **closed** | 68 tokens, site features preserved | — | — |
| 8 | OCR reads few names | 12/60 plans | Largely mitigated: types come from the model now | Inherent |
| 9 | ~~Diagonal walls~~ **not worth building** | Costs **2.7%** of wall pixels, no plan over 10% | See below | — |
| 10 | Classical baseline | No walls on 10/12 unseen | Scaffold for clean CAD only; the trained model is the contribution | Documented |
| 11 | Colour heuristics | Windows 5/12, planting 1/12 | One drafting office's convention | Documented |

> **Discrepancy note (row 1):** `scripts/window_detection_accuracy.py` now
> exists and measures **81.7% recall at 75.7% precision**, not the
> 62.1%/43.5% above -- the original measurement's method was never recorded
> and could not be recovered. The figure above is kept as the historical
> record rather than silently changed; treat the script's output as current.

## What rendering a random plan turned up

Everything above is scored against the annotations, which says how well
the drawing is read and nothing about what gets built from it. Rendering
a plan picked at random found four faults that no per-class score would
ever have shown, all of them general and all now fixed.

| Fault | What it did | Measured |
| --- | --- | --- |
| Room outlines kept their steps | Squaring makes edges axis-aligned but leaves 4-to-17-pixel notches, every one perfectly square and none of them real | 12.2 vertices per room to **6.9**, no area lost |
| The footprint was never squared | It carries the slabs, the roof and the parapet, so its jaggedness showed from every angle at once | 31 vertices to **10**; diagonal perimeter 6.8% to **none**; area moves 0.0% |
| Only the largest piece of a split slab was built | A terrace covering 5% of a storey removed half its roof, by spanning the building rather than sitting inside it | Every piece built now |
| Noise punched holes in roofs | An "open area" of 2.8 square feet, ten inches wide, cut a slot through a roof and the parapet lined both sides of it | **33 of 96** open regions were too small to stand in, on **10 of 40** plans |

The lesson worth keeping: a per-class IoU cannot see any of these. Three
of the four are in the geometry stage, downstream of anything the
segmenter is scored on, and the fourth is in ingestion upstream of it.
**Render a random plan and look at it** -- it is the cheapest test in the
project and it found more in one sitting than five GPU hours did.

## Most of the scale error is not a mistake

Both scale estimators read low -- doors by 8.4%, walls by 20.1% -- and two
independent methods biased the same way looks like a fault upstream of
both. It is not. CubiCasa records a true scale, so its own annotation can
be converted into feet and the question settled:

| | Measured from the annotation | Assumed by the code | Predicts a bias of | Actually measured |
| --- | --- | --- | --- | --- |
| Interior door | **2.28 ft (2'3")** | 2.50 ft | −8.8% | **−8.4%** |
| Wall thickness | **0.63 ft (7.6")** | 0.75 ft | −15.4% | **−20.1%** |

**The door bias is 97% explained by the constant alone.** A Finnish
interior door is 2'3", not the 2'6" the code assumes, and the pipeline is
reading it correctly. Walls are 77% explained, leaving about 4.7 points
that really is the extraction reading them thin -- which matches the 21%
narrowing already recorded above, damped by the gauge being measured on
the mask rather than on the segments.

This reframes gap #2 entirely. Scale is not 17.3% out because the vision
is weak; it is 17.3% out because **the pipeline does not know which
country the drawing came from**, and is applying one tradition's standards
to another's. The vision is better than that number suggests.

It also makes the trap below exact rather than cautionary. Fitting
`TYPICAL_DOOR_FT` to 2.28 would take the door bias to near zero on
CubiCasa and introduce an equal and opposite error on any plan drawn where
doors are 2'6". There is no constant that is right for both.

What follows from it:

- **No amount of better segmentation fixes most of this.** Only knowing
  the convention does.
- **The drawing usually states its own origin** -- areas printed in m²
  rather than sq ft, the language of the room labels, the paper size, the
  units on dimension strings. Reading that and selecting standards to
  match is the one route that does not involve fitting a constant to a
  dataset, and it is untried.
- Window widths were measured while the annotation was open: **3.88 ft
  median**, against the 3.0 ft one might assume, but with a quartile
  spread of 2.62 to 5.31 against doors' 1.94 to 2.66. A window is far too
  variable to calibrate on, which is worth knowing before anyone tries.

## The drawing can say whether to trust its own size

Since most of the scale error is a convention mismatch that no amount of
better vision detects, the useful thing left is to say **which** answers
to doubt. Doors and the wall gauge measure different things and are wrong
in different ways, so how far apart they sit says something neither says
alone. Both are now worked out even though only one is used, and the gap
is reported as `scale_agreement`, with `scale_confident` reading it.

Swept over the 30 plans carrying both estimates. They usually agree
closely -- median disagreement 0.062, 90th percentile 0.207 -- so the line
sits out in the tail:

| Threshold | Flagged | Error if confident | If flagged | Gap |
| --- | --- | --- | --- | --- |
| 0.10 | 11/30 | 10.8% | 17.6% | 6.8 |
| **0.14** | 6/30 | **11.1%** | **21.4%** | **10.3** |
| 0.18 | 5/30 | 11.5% | 17.6% | 6.1 |
| 0.26 | 2/30 | 12.3% | 17.5% | 5.2 |

As it ships, over 48 plans:

| | Plans | Median error | Within a fifth |
| --- | --- | --- | --- |
| Size reported confident | 24 | **11.1%** | **22/24** |
| Size flagged as doubtful | 24 | 20.2% | 11/24 |

Ninety-two per cent against forty-six. A plan is also flagged when only
one estimate could be worked out at all, which is a different reason to
doubt an answer and an equally good one.

**Combining the two was tried first and does not work.** It was the
obvious idea and it was wrong:

| Rule | Median error | Within a fifth |
| --- | --- | --- |
| **Doors alone, as shipped** | 12.3% | **25/30** |
| Walls alone | 9.3% | 24/30 |
| Arithmetic mean | 11.8% | 24/30 |
| Geometric mean | 11.6% | 24/30 |
| Larger of the two | 9.9% | 24/30 |

Every blend loses a plan. Walls alone looks tempting on the median and is
a trap twice over: these are the 30 plans where doors *also* work, which
is the easier half, and on the 18 where doors fail walls runs at 20.2%.

So the scale itself is unchanged. Only the confidence is new, and a caller
ignoring it gets the same numbers as before.

## The one that is a trap

**Do not re-tune the wall-thickness constant.** It looks like a bug and is
not:

| Population | Wall-scale bias | Implied wall |
| --- | --- | --- |
| CubiCasa, 24 plans | **−19.7%** | ~7.4 in |
| Indian villa (control) | **+6%** | ~9.5 in |

The 9-inch constant is right for one population and wrong for the other.
Fitting it to CubiCasa would trade Indian plans for Finnish ones. What
does generalise is preferring doors, which measure **+1.6% bias** — a door
is about 2'6" wherever it is drawn. Though see below: it is already
preferred as hard as it usefully can be.

## Training at 768 was tried, and did not pay

The prediction, recorded beforehand: a window arrives at the network 1.5
pixels wide at a 512 input and 2.2 at 768, so raising the resolution
should lift the model's weakest class substantially.

It was run: 24 epochs at 768, about five hours on a T4, validation Dice
0.7780 against 512's 0.7757. Scored per class over 40 plans:

| class | IoU 512 | IoU 768 | change |
| --- | --- | --- | --- |
| door | 0.567 | 0.619 | **+0.052** |
| bedroom | 0.765 | 0.807 | +0.042 |
| wall | 0.718 | 0.743 | +0.025 |
| storage | 0.525 | 0.544 | +0.019 |
| **window** | **0.096** | **0.113** | **+0.016** |
| kitchen | 0.713 | 0.689 | −0.024 |
| outdoor | 0.773 | 0.743 | −0.030 |
| circulation | 0.727 | 0.691 | −0.037 |
| room | 0.740 | 0.687 | **−0.053** |
| bath | 0.586 | 0.523 | **−0.063** |

The thin classes improved and the large room classes got worse, and the
two cancelled. Windows moved 17% in relative terms and remain five times
worse than anything else.

Downstream, which is what actually matters, it is no better and slightly
worse:

| | 512 | 768 |
| --- | --- | --- |
| Plans reconstructed | 20/20 | 20/20 |
| Median openings found | **11** | 9 |
| Scale within a fifth of true | **16/24** | 14/24 |
| Training cost | 2.5 h | 5 h |

**The 512 checkpoint is the one in use.** Resolution is not the binding
constraint on windows, and the five hours bought nothing. Recorded as a
prediction that failed.

### So what is the constraint?

Unknown, and worth saying so rather than guessing again. Three candidates,
none tested:

1. **The metric may be misleading.** IoU is brutal on a 4-pixel structure
   -- a one-pixel offset costs a quarter of it. Recall rose from 36.7% to
   41.2%, which is the more informative number and still poor.
2. **Class imbalance beyond what weighting can fix.** Windows are 0.10% of
   a drawing. Their loss weight is already the highest at 3.72.
3. **Semantic segmentation may be the wrong tool for them.** A window is
   not a region so much as an interruption in a wall, and might be better
   found by looking for gaps along already-extracted walls than by asking
   a per-pixel classifier for a class that thin.

The third is the most promising and the least explored.

## Wall extraction, measured against the annotations

Everything downstream rests on this. Measured by painting the built walls
back at their own thickness and comparing with the annotation, over 30
plans -- `scripts/wall_accuracy.py`:

| | Median | Plans below 70% |
| --- | --- | --- |
| **Coverage** — annotated wall that gets built | **96.6%** | **0 of 30** |
| **Agreement** — built wall that really is wall | **92.2%** | 4 of 30 |

Both were 0.8 higher (97.4% and 93.0%) before windows were given a lower
bar than the wall they sit in; see below for that trade.

Agreement was 88.4% until the ceiling on wall thickness came down from
four times the drawing's own wall to 2.5 -- see `MAX_THICKNESS_RATIO`. It
cost 1.6 points of coverage and bought 4.6 of agreement, and the wall and
opening counts did not move, so what it removed was a handful of grossly
over-thick runs per plan rather than real wall.

Three hypotheses for the remainder were tested and rejected:

- **Not envelope closing.** Sample 10711 has *zero* invented walls and
  still sat at 62%, so the disagreement is in the walls read off the
  drawing.
- **Not thickness in general.** Extracted walls come out *thinner* than
  annotated ones at a median ratio of 0.79, so a wall painted at its own
  thickness lands inside the real one rather than spilling outside it. It
  is the few extreme runs that cost, not the typical one.
- **Not open junctions.** See below.

What is left, on the four plans still under 70%, is **sheets that were
never split**. Plan 11855 is two plans side by side and scores 56%: its
worst invented walls are a title-block rule along the foot of the sheet
and runs fused across the gap between the two drawings. That is gap #3,
not a wall-extraction fault.

## Chamfering open junctions was tried, and there were none to close

The prediction: 3DPlanNet (Park & Kim, *Electronics* 2021, 10, 2729)
lists four node/edge generation rules, and three of them already exist
here in `_merge_collinear`. The fourth, "chamfer" -- extending two walls
whose endpoints nearly meet into a proper right-angle junction -- did
not, and open junctions looked like an obvious cause of rooms failing to
close.

It was built and swept from 0.5 to 4 wall-thicknesses of reach. Coverage
and agreement did not move at all until 4, where agreement got *worse*.

The reason is measurable, and is why the rule does not transfer. Across
12 plans there are 79 junctions where two perpendicular walls do not
meet, and their gaps are not near-misses:

| Gap between wall and wall, in wall-thicknesses | |
| --- | --- |
| 10th percentile | 3.2 |
| median | **4.8** |
| 90th percentile | 5.7 |
| **within 1.5 thicknesses** | **3 of 79** |

A junction here is either already crossing or a real doorway. Walls are
extracted as runs whose endpoints sit on their own bounding-box
centreline and are then merged across gaps of up to 3.75 thicknesses, so
anything that could be chamfered has already been joined. Reaching to 4
starts bridging doors, which is why agreement fell.

The rule is sound in its own setting and redundant in this one. Recorded
as a prediction that failed, and the code removed rather than left in
place firing on 3 junctions in 79.

## Scale from what the drawing prints, gated

A drawing that states its own size is better evidence than any inference
from door widths. Two forms are read: dimension pairs (`13'0" x 10'0"`)
and printed areas (`2130 SQ.FT.`, `125 m2`), the latter being the
relation 3DPlanNet calibrates on, where it reaches 97%.

**Both are now gated.** Previously a dimension reading won outright. OCR
on a busy sheet drops a foot mark, transposes a digit, or takes a door
tag for a room size, and one bad read resized the whole building. A
printed figure is now used only when it agrees with the geometric
estimate within 40% -- looser than that estimate's own ~20% error, so it
cannot reject a correct reading, but tight enough to catch the failures,
which are wrong by multiples rather than by a third.

The area path was checked for structural bias, which it would inherit
squared. Over 30 plans the extracted room polygons enclose **0.986** of
the annotated room area at the median, 29 of 30 within 10%, so a scale
taken from an area carries only a **-0.7%** bias. The method is sound.

Matching a label to a room was not. A point on a plan sits inside several
outlines, because the segmenter emits one polygon per class and a sheet
can carry ninety of them; taking whichever came first in class order
charged the reference sheet's `2130 SQ.FT.` to a fragment a third of the
terrace's size and put the scale out by 36%. Charging each label to the
**largest** outline containing it took that to -14.5%, where the gate
then prefers the dimension pairs anyway.

**On the corpus this changes nothing**, and that is the point: CubiCasa's
rasters print almost no text -- OCR returned 0 to 3 boxes per plan and no
usable dimension pair on 8 of 8 -- because its dimensions live in the
annotation rather than on the drawing. Scale accuracy is unchanged at
17.7% median, 16 of 24 within a fifth.

So this is **opportunistic and unvalidated beyond one drafting
convention**. It helps sheets that print their sizes, it is checked
before it is believed, and it cannot make an unlabelled plan worse. It is
not evidence that scale is solved; gap #4 is what would settle that.

## Windows: the gap search cannot work, but a lower bar can

Windows are the weakest class by a wide margin. This audit previously
called finding them as **gaps in already-extracted wall geometry** "the
most promising and the least explored". It was measured, and it is
blocked.

### Why the gap search cannot work

A gap search can only find a window where the predicted wall mask is
actually interrupted. Of the 75 windows the segmenter misses across 28
plans:

| Where the missed window sits | Count | |
| --- | --- | --- |
| Inside **solid** predicted wall | **62** | **83%** |
| In a gap in the predicted wall | 6 | 8% |
| Not near any extracted wall | 7 | 9% |

Asked what it predicts at those points, the segmenter says **wall 85% of
the time**. It does not leave a hole where a window is; it paints the
window as wall. The ceiling for a gap search is therefore 6 windows,
taking recall from 60.5% to at most **63.7%** -- before counting the
false positives it would add to a precision already at 36%. Recorded as
a prediction that failed, and not implemented.

### What the numbers actually are

Scored as detection rather than per-pixel overlap, which is the more
useful question for a 4-pixel strip: does an opening land on the window
the drawing shows, within a wall thickness and a half. Over 190
annotated windows on 28 plans, the segmenter-only baseline is **60.5%
recall at 36.3% precision** -- far better recall than the IoU of 0.096
suggests, and far worse precision. 317 detections for 190 windows: the
problem was never only that windows are missed.

### The bar was in the wrong place

Taking the most likely class at every pixel is right when classes are
comparable in size. Windows are 0.10% of a drawing and arrive at the
network barely a pixel wide, so a window pixel sits ringed by wall and
wall wins the average. Giving windows a lower bar, with no retraining:

| Rule | Recall | Precision | F1 |
| --- | --- | --- | --- |
| argmax | 58.4% | 39.6% | 0.472 |
| P(window) >= 0.35 | 57.9% | 40.3% | 0.475 |
| P(window) >= 0.30 | 62.6% | 46.7% | **0.535** |
| **P(window) >= 0.25** | **63.2%** | **45.3%** | 0.527 |
| P(window) >= 0.20 | 60.5% | 47.3% | 0.531 |

Better on both counts at once, which is not the usual shape of this
trade: forcing the confident pixels through also tidies the fragments
either side of them, so fewer spurious openings survive as well as more
real ones. 0.20 to 0.30 all behave alike and 0.35 upwards is
indistinguishable from argmax, so `WINDOW_PROBABILITY_FLOOR` sits at
0.25, mid-band.

One thing this is **not**: evidence that the model sees these windows and
narrowly loses the argmax. At a missed window the mean wall probability
is 0.68 against 0.028 for window, and window ranks fourth or worse half
the time. The model is confidently wrong there. The floor recovers the
minority it is unsure about, not the majority it gets wrong.

### It may not overrule a door

The floor is allowed to overturn **wall and background only**
(`WINDOW_MAY_OVERRULE`). The argument for it -- a window pixel sits
ringed by wall and wall wins the average -- says nothing about a door,
and doors are what the scale estimate rests on. Letting windows overrule
doors would trade the building's size for its glazing. Restricting it
costs nothing and gains a little:

| Rule | Window recall | Window precision |
| --- | --- | --- |
| argmax | 58.4% | 39.6% |
| Floor over any class | 63.2% | 45.3% |
| **Floor over wall and background** | **63.2%** | **45.8%** |

### It is a trade, and the cost is in the scale

Measured end to end over 48 plans, with each of today's two behaviour
changes switched on alone:

| Configuration | Scale median | Within a fifth | Plans scaled from doors |
| --- | --- | --- | --- |
| Before today | 17.6% | **34/48** | 30 |
| Window floor only | 17.7% | 30/48 | 29 |
| Splitting fixes only | 17.6% | 32/48 | 29 |
| Both, as shipped | 18.1% | 28/48 | 28 |

The mechanism is second-order and worth stating plainly: carving window
out of wall changes the wall segments, some doors then sit too far from
any wall to bind to one, and a plan with too few doors falls back to the
wall gauge, which carries a -20% bias. It is not that windows overrule
doors -- they are forbidden to -- it is that they move the walls doors
attach to.

That opposition turned out to be mostly someone else's fault. Removing
`refine_windows` (below) gave the scale back, so the standing position is:

| | Before | Window floor | ...and without `refine_windows` |
| --- | --- | --- | --- |
| Window recall | 60.5% | 62.1% | **62.1%** |
| Window precision | 36.3% | 43.5% | **43.5%** |
| Wall coverage | 97.4% | 96.6% | 96.6% |
| Wall agreement | 93.0% | 92.2% | 92.2% |
| Scale within a fifth, of 48 | 34 | 28 | **33** |

One plan of forty-eight, for six points of window precision.

**Kept**, on the grounds that a model with the wrong windows in the wrong
places is wrong in a way anyone can see, while absolute size is already
declared an estimate to callers whenever it is not measured. But it is a
judgement between two real costs rather than a free win, the whole of it
is above, and reverting it is one constant.

## Two proposed scale sources, both examined and neither added

Two OCR-independent scale sources were proposed for the fallback chain.
Both were investigated and neither is added, for different reasons, both
measured.

### PDF page size cannot contribute to scale

The proposal was to read a PDF's physical page size and combine it with
`WORKING_DPI` for an exact pixel-to-foot mapping. It cannot work, and the
reason is arithmetic rather than empirical:

    pixels per foot = dpi x 12 / drafting ratio

Page size does not appear. A drawing at 1:150 rendered at 400 dpi is 32
px/ft on A4, A1 or a napkin -- a bigger sheet holds more building, at the
same scale. What page size can recover is *dpi*, when a raster's
resolution is unknown. Two measurements say that case does not arise
here:

| | |
| --- | --- |
| Reference PDF page size | 612 x 792 pt (US Letter) |
| Rasterized at `WORKING_DPI` = 400 | 3400 x 4400 px, exactly as predicted |
| CubiCasa rasters carrying a dpi tag | **0 of 60** |

For a PDF the pipeline rasterizes itself, measured dpi is identical to
assumed dpi by construction. For a bare image there is no tag to read.
The source is inert on both corpora and is not implemented.

### Printed areas are legible on one sheet in sixty

The proposal was to take one printed total area against the shoelace area
of the extracted footprint. Two things defeat it.

**The text cannot be read.** Across 60 CubiCasa sheets, exactly **1**
yields a printed area OCR can parse. The areas are there -- 12787 prints
`44.5 M2`, 11378 prints `99,0 m2`, 8583 prints `101,0 m2`, all legible to
the eye. They are below OCR's resolution floor: these rasters are 514 to
1900 px wide where the reference sheet is 2275 px for a *single* floor.
Five preprocessing variants were tried and none recovered a single
dimension pair or area from twelve sheets:

| Preprocessing | Lines | Dimension pairs | Areas |
| --- | --- | --- | --- |
| Current, cutoff 60 | 8 | 0 | 0 |
| Otsu | 41 | 0 | 0 |
| Cutoff 128 | 29 | 0 | 0 |
| Cutoff 160 | 30 | 0 | 0 |
| Upscale 2x + Otsu | 41 | 0 | 0 |

Upscaling interpolates pixels; it does not create letterforms that were
never sampled.

**And the "total" is not identifiable.** On the reference sheet, where
areas *are* readable, neither printed figure is a building total --
`2130 SQ.FT.` is the terrace garden and `600 SQ.FT.` the deck. Taking the
largest as the total, against the footprint:

| Printed figure | As a building total | Error against the true 26.28 px/ft |
| --- | --- | --- |
| `600 SQ.FT.` | 67.85 px/ft | **+158%** |
| `2130 SQ.FT.` | 36.01 px/ft | **+37%** |

The per-room form already in `calibrate.scale_from_areas` -- each label
charged to the region it names rather than assumed to be the whole
building -- gives **-14.5%** on the same sheet and is correctly overruled
by the dimension pairs. The total-against-footprint form is worse in
every case measured and is not implemented.

`scripts/scale_accuracy.py` is unchanged at **17.7% median error, 16 of
24 within a fifth**, because nothing was added to the chain.

### One thing this did settle: leave `INK_CUTOFF` alone

The hard cutoff at greyscale 60 in `isolate_ink` looks like a constant
fitted to one drawing set, and its own docstring says it was measured on
the reference sheet. Tested against four alternatives on that sheet at
full resolution, it is the best of them at what actually matters:

| Preprocessing | Lines | Dimension pairs | Areas |
| --- | --- | --- | --- |
| **Current, cutoff 60** | 105 | **14** | **2** |
| Cutoff 128 | 118 | 9 | 1 |
| Otsu | 73 | 9 | 0 |
| Upscale 2x + Otsu | 154 | 15 | 0 |
| Cutoff 160 | 69 | 7 | 0 |

Alternatives find more raw text and less usable text. Recorded so it is
not re-tuned on a word count.

Also worth recording, because it is an operational fact rather than a
code one: the same sheet rendered at 150 dpi yields **0** text lines
where at 400 dpi it yields **105**. Resolution, not preprocessing, is
what decides whether a drawing's own text is available at all.

## Daylight was washing the building out

The renders read as one flat wash. Measured on what actually reaches the
pixels rather than argued about: half the picture sat above tone 189, the
75th to 99th percentiles were crushed between 213 and 221, and median
saturation was **23 of 255** -- close to grey, from a material palette
spanning 145 levels of luminance.

The cause was ambient light. It lights every surface whatever way it
faces, so it is the one control that flattens a building rather than
lighting it, and at 0.58 with a key of 0.78 and a fill on top, more light
arrived than the filmic curve had room for.

| Ambient | Exposure | Tonal spread | Saturation |
| --- | --- | --- | --- |
| 0.58 | 1.05 | 35 | 28 |
| 0.42 | 0.90 | 71 | 34 |
| **0.35** | **0.90** | **75** | **35** |
| 0.28 | 0.82 | 80 | 41 |

Both numbers keep improving as the two come down, so where to stop is a
judgement rather than a peak: at 0.35 and 0.90 the form reads and nothing
is crushed, and going further trades a murkier shaded wall for a few more
points. Raising the key instead does **not** work -- it pushes the lit
faces back into the roll-off and the spread falls again, to 34.

Checked on fourteen models built from fourteen different drawings, not on
the one it was chosen from:

| | Spread | Saturation | Blown | Crushed |
| --- | --- | --- | --- | --- |
| As shipped | 68 | 30 | 0.0% | 0.0% |
| Rebalanced | **76** | **35** | 0.0% | 0.0% |

Thirteen of fourteen gained tonal spread and fourteen of fourteen held or
gained saturation.

Only "midday" used these defaults; the other times of day set their own
and were scored the same way afterwards. They are left alone, because
each is now doing something deliberate rather than failing:

| Preset | Spread | Saturation | Median |
| --- | --- | --- | --- |
| midday | 77 | 35 | 180 |
| golden hour | 67 | 34 | 193 |
| overcast | 66 | 28 | 171 |
| dusk | 59 | **74** | 148 |

Overcast is flat on purpose -- its own note says nothing should be
flattered by a good raking light -- and dusk trades spread for colour,
which is what dusk does.

## Resolution is not the constraint, and that is now settled

Four separate attempts to buy accuracy with pixels, none of which paid.
Written together because the idea keeps returning in a new costume.

| Attempt | Result |
| --- | --- |
| **Train at 768** | Windows +0.016 IoU, room and bath worse, downstream flat to worse. 5 GPU hours. |
| **Find windows as gaps in the wall** | 62 of 75 missed windows sit inside *solid* predicted wall. Ceiling +3.2% recall. |
| **Segment high-resolution tiles** | Window F1 0.543 to 0.409 at 2x2 and 0.319 at 3x3. Precision halves. |
| **Infer at a larger input** | No configuration dominates; 640 buys 0.8 of wall agreement and costs two plans their scale. |

The tiling result is worth keeping for what it says about Task 5. Cutting
a sheet into tiles and segmenting each at the same input size gives the
network a bigger picture of a smaller area -- exactly what the tiled
object-detection precedent proposes -- and it makes windows sharply
**worse**, because a tile is out of distribution twice over: the walls
arrive two or three times thicker than any it trained on, and a corner of
a plan with no enclosing wall is not something it has seen. That does not
refute Task 5, whose detector would be *trained* on tiles. It does mean
Task 5 cannot be tested cheaply with the weights we have, so it is a
full training commitment rather than a quick experiment.

Inference size was worth trying because a U-Net is fully convolutional
and it costs nothing but time. It is measured here because the natural
next thought after "training at 768 failed" is "then infer at 768", and
someone should be able to see that it was tried:

| Inference input | Wall coverage | Wall agreement | Scale median | Within a fifth |
| --- | --- | --- | --- | --- |
| **512, as trained** | 96.5% | 90.2% | **17.7%** | **16/24** |
| 640 | **97.1%** | 91.0% | 18.7% | 14/24 |
| 768 | 94.9% | **92.0%** | 17.7% | 15/24 |

What the same experiments did turn up is that the model's quality tracks
the wall thickness it is *shown* after the square resize:

| Wall gauge at the network's input | Plans | Median wall IoU |
| --- | --- | --- |
| Under 7 px | 9 | 0.697 |
| 7 to 12 px | 12 | 0.772 |
| Over 12 px | 4 | **0.827** |

A sheet arriving at 2.2 px scores 0.442. Large sheets shrink most and are
read worst. That is a real weakness and it is **not** fixed by feeding
the network more pixels, as the table above shows -- the two ideas look
alike and are not. It is a training-distribution problem: the augmentation
already rescales, and whether it rescales enough is a question for the
next run rather than for inference.

## Colour-read windows were making the model worse

`refine_windows` ran on every mask the segmenter produced. Where colour
found enough glazing strips to be trusted it **deleted the model's
windows entirely** and used the coloured ones instead; below that
threshold it merged the two. Its premise, in its own docstring, was that
"colour is far more reliable than the model on sheets that mark glazing
in colour".

It had never been scored against the annotations. Over 28 plans, as
detection:

| | Recall | Precision | F1 |
| --- | --- | --- | --- |
| The model alone | **62.1%** | **43.5%** | **0.512** |
| After `refine_windows` | 41.6% | 26.7% | 0.325 |

And it fails hardest exactly where it claims to help:

| What it did to the sheet | Plans | F1 model | F1 after |
| --- | --- | --- | --- |
| Colour trusted, model wiped | 12 | 0.509 | **0.051** |
| Colour merged in | 7 | 0.406 | 0.397 |
| No colour found | 9 | 0.600 | 0.600 |

On the twelve sheets where it decided colour knew better, it took window
detection to an F1 of **0.05** -- ten times worse than the model it
overruled. It hurt the colourful subset by 0.240 and the high-quality one
by 0.107, so this is not a matter of picking the right sheets.

The premise was probably true once. It was written against a model
scoring 0.12 IoU on windows; that model has since been retrained on 11
classes with class weighting, and windows now have a probability floor.
The heuristic was never re-measured against the model that replaced the
one it was beating.

It no longer touches the trained model's output. The function stays in
`classical`, where the baseline still uses it and where the traps list
already says colour-driven windows are one drawing office's convention.

**It was also the scale regression.** Deleting the model's windows
changed the wall mask, which moved the wall segments, which orphaned
doors -- and doors carry the scale estimate. Removing it took plans
calibrated within a fifth from 28 of 48 back to **33**, with 30 plans
scaled from doors rather than 28, and the worst error from 56.1% to
44.6%. The window floor was blamed for that and was mostly innocent.

## Open to the air, and how much of the gain was the instrument

The workstream set out to decide correctly which spaces have sky above
them. It found one defect in the build and two in the score that was
meant to judge it, and the headline number moved mostly for the second
reason. Read without that distinction it says something untrue.

Measured this session, `scripts/open_air_accuracy.py` over
`/tmp/claude/cubicasa_batch` with `models/unet_cubicasa.pt`:

| | IoU | recall | precision | sheets scored |
| --- | --- | --- | --- | --- |
| where it started | 48.2% | 93.0% | 50.0% | 60 |
| **where it stands** | **80.5%** | **96.3%** | **83.0%** | **48** |

**The two rows do not score the same corpus, and that is half the story.**
Three changes moved the number; one of them changed what gets built:

| change | what it moved | after |
| --- | --- | --- |
| scoring what `extrude.open_to_sky` merges, not only what `features.is_open_to_sky` accepts | **the instrument** | 48.2%, 60 sheets — the honest baseline. The first run of the harness reported 74.5% |
| `MAX_PLANTING_SHEET_SHARE` in `planto3d/classical.py` | **the build** | 72.5%, 60 sheets |
| declining to score sheets `ingest.split_sheet` would have cut | **the instrument** | 80.5%, **48 sheets** |

`git diff --stat <the planting commit>..HEAD -- planto3d/` is empty. One
commit in this workstream touched production code; the rest touched the
harness, the tests and this file. **48.2 to 80.5 is not 32 points of
better roofs.** The build improved by the 24.3 points the planting bound
bought, over an unchanged 60 sheets. The rest is a score that had been
flattering itself in one direction — never looking at the colour route,
which is where the whole defect was — and punishing itself in another, by
judging twelve multi-plan sheets in a configuration nothing ships.

The first correction made the number worse and was still the right change.
A harness that cannot see the source a candidate would move is not an
instrument, and this one had already been used to report 74.5%.

Wherever 80.5% is quoted, the 48 belongs beside it. The 12 sheets set
aside score 65.1% / 86.2% / 72.7% and are not fixed, only excluded from a
measurement that could not frame them.

### The workstream was scored against the wrong check

Both the spec and the plan said this work would be judged by
`output_scorecard.py`'s `openings` check. It will not be, and cannot be.
That check is `result.opening_count / drawn` bounded to 0.6-1.5
(`scripts/output_scorecard.py:138`) -- it counts **doors and windows**
against the annotation's, and has nothing to do with which spaces have
sky above them. It stands at 9 of 30 before this workstream and 9 of 30
after, exactly as it must.

The scorecard is unchanged at **10 of 30** for the same reason. Nothing
it measures is what this workstream changed. `scripts/open_air_accuracy.py`
exists because no instrument in the project could see this defect --
`output_scorecard.py` says so itself in its own docstring, listing "a
balcony sealed under a slab" among the things it cannot judge.

The lesson is worth more than the correction: a workstream that names its
acceptance metric without checking what the metric computes can run to
completion and prove nothing. This one built its own instrument and is
measurable; the nomination in the spec was simply wrong and is corrected
there.

### What the numbers here do not cover

**The corpus's provenance is unrecorded.** These 60 CubiCasa sheets may
overlap the split `models/unet_cubicasa.pt` was trained on -- nothing in
this repository records which sheets trained the checkpoint. If they do,
the 96.3% recall is partly memorised and the true figure on unseen plans
is lower. This qualifies every number in this section, and it is not
fixable without the training manifest.

**The remaining false positives need retraining, not a rule.** After the
planting bound and the split exclusion, 17.0% of what is still wrongly
opened is covered exterior structure the segmenter calls outdoor and
CubiCasa does not: a BILTAK carport, an AVOKUISTI porch, a glazed
balcony. CubiCasa maps `CarPort` and `Garage` to STORAGE
(`planto3d/cubicasa.py:97`), so the model is being trained toward one
answer and judged against another. No rule was invented for this, and
inventing one would be fitting to the sheets that happen to show it.

### Green ink does not mean planting, and it was taking roofs off

`vegetation_regions` states its assumption in its own first line --
*"Architectural sheets are otherwise greyscale, so saturated colour is a
reliable signal"* -- and then never checked it. `pipeline.py` calls it on
every sheet, and `extrude.open_to_sky` merges its output with the other
two open-air sources, so a green region it invents is a hole in a roof
slab.

**Scored on its own over the 60-plan CubiCasa corpus**, with the other two
sources switched off, the colour route is:

| | IoU | recall | precision |
| --- | --- | --- | --- |
| every source (as shipped) | 48.2% | 93.0% | 50.0% |
| the colour route alone | 4.1% | 7.0% | **8.8%** |
| the colour route deleted | 74.6% | 91.4% | 80.3% |

Deleting it outright is worth **26.4 IoU points** for 1.6 points of
recall. It was not deleted: on the reference sheet the same code finds
the TERRACE GARDEN correctly, twelve regions over twelve pages, and on a
plan drawn in colour it is the only source that finds a garden at all.
The defect is the missing precondition, not the detector.

**What the green actually is on the sheets where it fails.** Looked at,
not guessed. On CubiCasa `high_quality/11855` the green is an estate
agency's brand colour applied to the unit's *walls* -- the Aktia logo is
in the same green at the foot of the sheet. On `high_quality_architectural/6266`
it is a highlighter tracing one apartment's exterior wall. The closing
step joins that outline into a single region enclosing the whole floor
plan, and `open_to_sky` then opens all of it.

#### Four candidate preconditions were measured; only one separates

The obvious gate -- *is this sheet in colour at all* -- does not work, and
the measurement is worth keeping because it is the one that looks most
convincing on a corpus average. Fraction of pixels with HSV saturation
above 0.15, per sheet rather than pooled:

| corpus | mean saturation | fraction > 0.15 |
| --- | --- | --- |
| CubiCasa, 60 sheets | 0.030 | 0.044 (range 0.000 - 0.311) |
| `data/bridge`, 60 sheets | 0.263 | 0.971 (range 0.891 - 1.000) |
| `demo_plans`, 6 sheets | 0.176 | 0.652 (range 0.000 - 0.995) |
| reference sheet, 12 pages | - | 0.020 (range 0.001 - 0.083) |

Pooled by corpus this looks like a clean 22x separation. Per sheet it is
not a separation at all: **the reference sheet -- the only corpus whose
planting has been verified by eye -- sits inside CubiCasa's range**, and
two of the six `demo_plans`, including `3-RICHEST-7labels-open-paving-wet.png`,
carry no saturated pixels whatever. A gate on sheet colour would suppress
the very sheet the detector was written for while claiming to protect
colour plans. **Do not retry this.**

Three more were tried on the same sheets and none separates either:

| candidate | true planting | false planting | verdict |
| --- | --- | --- | --- |
| share of green pixels in elongated components | 0.0 - 0.1% | 0.0% (6266), 3.4% (11578) | overlaps |
| green lying within 7px of dark linework | 0.15 - 1.00 | 0.09 - 0.99 | overlaps |
| non-green ink inside the reported bed | 0.2 - 1.4% | 0.0 - 0.8% (11260) | overlaps |

#### What does separate is the size of the bed

A bed is a feature on a sheet; it is never the sheet. As a share of sheet
area:

| verified by eye as planting | | verified by eye as not planting | |
| --- | --- | --- | --- |
| reference `page-3` | 8.68% | CubiCasa 6266 | 66.11% |
| reference `page-3_cropped` | 22.34% | CubiCasa 12787 | 71.58% |
| reference tightest crop | 38.22% | CubiCasa 9285 | 100.00% |
| reference `page-1` | 1.73 - 8.13% | | |
| `demo_plans/2-TWO-STOREY` | 0.92% | | |
| `data/bridge/113782` | 0.92% | | |

Largest true bed 38.22%, smallest false one 66.11%, nothing in between
across four corpora. `MAX_PLANTING_SHEET_SHARE` is set mid-gap at 0.50.
Swept with `scripts/open_air_accuracy.py`:

| bound | IoU | recall | precision |
| --- | --- | --- | --- |
| none | 48.2% | 93.0% | 50.0% |
| 0.90 | 62.8% | 91.4% | 66.7% |
| 0.70 | 66.4% | 91.4% | 70.8% |
| **0.50** | **72.5%** | **91.4%** | **77.9%** |
| 0.40 | 72.5% | 91.4% | 77.9% |
| 0.30 | 72.5% | 91.4% | 77.9% |
| 0.00 | 74.6% | 91.4% | 80.3% |

The flat run from 0.30 to 0.50 is the evidence that the constant is not
fitted: across that span it removes the same three regions and scores
identically. It buys **24.3 of the 26.4 IoU points** that deleting the
colour route would, and keeps the route. Recall is 91.4% at every value
the bound can take, including at 0.00, so those three giant regions
carried no true open air at all.

Counted before and after, the planting regions found are unchanged
everywhere the detector was right: reference sheet 4 -> 4 (2 on the ground
storey, 0 on the first, 2 on the second, re-measured by running the bound
at 1.01 and at 0.50), `data/bridge` 1 -> 1, `demo_plans` 1 -> 1. On
CubiCasa 43 -> 40. `output_scorecard.py` holds at 10 of 30.

An earlier revision of this paragraph said "reference sheet 12 -> 12". That
was wrong: 12 is the page count of the reference PDF, not its planting
count, and the two were conflated. The detector finds 4 planted regions
across that PDF's three storeys. The claim the sentence makes -- that the
bound does not touch this sheet -- is unaffected and was re-verified; only
the number was.

### Where the remaining wrong roofs actually are

With the colour bound in place, 22.1% of the pixels the build leaves open
were wrong. Diagnosed rather than guessed at, by painting each of the three
sources `extrude.open_to_sky` merges onto the sheet separately and
intersecting each with the annotation, over all 60 sheets:

| source of the false-positive pixel | px | share |
| --- | --- | --- |
| rooms `is_open_to_sky` accepted | 1,753,346 | 86.4% |
| what remains of the colour route | 276,693 | 13.6% |
| `GROUND_COVERS` labelled regions | **0** | 0.0% |

Labelled regions contribute nothing at all, as the 3.6% label rate implies.
Every one of the 1,753,346 room pixels came from a room the segmenter typed
`outdoor`; not one came from a printed label.

Splitting those room pixels by whether the segmenter itself predicted
OUTDOOR there:

| | px | share |
| --- | --- | --- |
| the segmenter also said outdoor -- the model is wrong | 1,578,359 | 90.0% |
| the segmenter did not -- the traced polygon covers more ground than the evidence | 174,987 | 10.0% |

So nine tenths of it is the OUTDOOR class being wrong, which no
inference-time rule reaches and which retraining is out of scope for.

**What the annotation calls the pixels we get wrong**, pooled over all 60
sheets:

| true class | px | share |
| --- | --- | --- |
| background | 1,188,376 | 58.5% |
| room | 472,068 | 23.3% |
| bedroom | 158,162 | 7.8% |
| wall | 141,517 | 7.0% |
| bath, circulation, storage, kitchen | 54,692 | 2.7% |
| door, window | 15,226 | 0.7% |

Nearly three fifths of it is not a roof taken off a room at all -- it is
open ground painted on annotated background, page the drawing does not
occupy.

### The instrument was measuring itself

That background figure has one cause, and it is the harness rather than the
build. `open_air_accuracy.py` runs every sheet at `split=1` so the
prediction and the annotation share one frame. Twelve of the sixty sheets
are ones `ingest.split_sheet` would cut into two or three plans, and on
those the segmenter sees the gutter between the plans and calls it outdoor.

| | false-positive px | share |
| --- | --- | --- |
| 12 sheets the splitter would cut | 1,244,689 | **61.3%** |
| 48 single-plan sheets | 785,352 | 38.7% |

61.3% of the error on 20% of the sheets, in a configuration nothing ships.
The five worst sheets by false positives -- 11855, 10394, 8329, 9285,
12912 -- are all multi-plan sheets.

The harness now declines to score them, which is the rule
`output_scorecard.py` already applies to walls. Measured this session:

| corpus | IoU | recall | precision |
| --- | --- | --- | --- |
| all 60 sheets, as previously reported | 72.5% | 91.4% | 77.9% |
| 12 multi-plan sheets alone | 65.1% | 86.2% | 72.7% |
| **48 single-plan sheets -- what the build does** | **80.5%** | **96.3%** | **83.0%** |

No build code changed; only what the instrument agrees to judge.

**Ink density does not separate a spurious open region from a real
terrace.** Tried as a cheap discriminator for the regions painted on blank
page, on the theory that a drawn terrace has hatching, a boundary or a
label in it and page margin has nothing. Measured over the 100 open regions
the build produces on this corpus: real terraces (at least half their area
annotated OUTDOOR, n=74) have a median dark-pixel share of 0.069, and
regions at least 90% on annotated background (n=4) a median of 0.055 --
the wrong way round, and overlapping at every percentile. The 18 wholly
spurious regions span 0.020 to 0.989. **Do not retry this.**

What is left after the multi-plan sheets are set aside is 17.0% of built
open pixels, of which the largest identified populations are covered
exterior structures the segmenter reads as outdoor and CubiCasa does not --
`BILTAK` (carport) on 3718 at 161,006 px, `AVOKUISTI` (open porch) on 8329,
a glazed balcony on 6632 at 57,812 px. CubiCasa's own ontology maps
`CarPort` and `Garage` to STORAGE, not Outdoor (`planto3d/cubicasa.py:97`).
That is a disagreement about whether a roofed outdoor structure is open to
the sky, and settling it needs the model retrained, not a rule added.

Scored alone again *with* the bound in place, the colour route now
contributes 0.0% IoU on CubiCasa -- everything it still finds there is
filtered out by the sliver test in `real_open_regions` before it reaches
a roof. Its entire net contribution to this corpus was the harm.

`extrude.real_open_regions` already sized open regions, but only from
below -- its docstring says *"a sliver of misread paving cannot punch a
hole through a floor"*. Nothing bounded them from above, which is the
direction that costs a whole roof rather than a corner of one.

### Gardens above ground level already work, and the spec was stale

The spec's geometry item for this workstream -- landscaping "including
gardens above ground level, not only at grade" -- describes a defect that
no longer exists. Checked before writing any fix, and the answer is that
it was fixed on 2026-08-20 by `a698554`, twelve days before the spec was
written.

Ground cover is placed inside the per-storey loop of
`extrude.floors_to_parts`. `storey_ft = base_ft + SLAB_THICKNESS_FT`
(`planto3d/extrude.py:1707`) is that storey's own walking surface, and the
only override is the ground-storey exception at
`planto3d/extrude.py:1718` -- a cover on storey 0 that is not mostly
inside the footprint drops to site level, because a lawn beside the house
lies on the ground rather than on the plinth. Nothing else keys on the
storey index, so every upper storey builds its planting on itself.

Measured three ways this session rather than read off the source:

**Synthetically**, a three-storey block with one 360x300 px region placed
on each storey in turn, built through `floors_to_parts` at 20 px/ft, from
each of the three sources `open_to_sky` merges:

| source | storey 0 | storey 1 | storey 2 |
| --- | --- | --- | --- |
| `FloorPlan.planting` (colour) | 0.686 m | 3.581 m | 6.477 m |
| `labelled_regions["lawn"]` | 0.686 m | 3.581 m | 6.477 m |
| room labelled TERRACE GARDEN | 0.686 m | 3.581 m | 6.477 m |

against storey floor levels of 0.686, 3.581 and 6.477 m. One lawn mesh
each time, at its own storey, never dropped and never at grade.

**On the reference house** (`data/soni_residence`, 3 storeys, scale
25.90 px/ft), `extract` finds 2 planting regions on floor 0, 0 on floor 1
and 2 on floor 2. The build produces 4 lawn meshes: two at 0.686 m and
two at **6.477 m**, the third storey's floor. The roof garden is built on
the roof.

**And it is not then slabbed over.** Built at the right height and roofed
is the same as not built: on the synthetic two-storey block, top-storey
planting takes the roof from 219.381 to 172.279 m2 of mesh area, and on a
three-storey block whose top storey steps back, mid-storey planting takes
the slabs from 433.393 to 386.291 m2. `open_to_sky` collects
`floor.planting` at `planto3d/extrude.py:436`; the slab above cuts to it
at `planto3d/extrude.py:1587` and the roof at `planto3d/extrude.py:1766`.

**Looked at, not only asserted.** `demo_plans/2-TWO-STOREY-stairs.gif` has
1 planting region, and it is on the upper storey, not the ground one.
Rendered through `scripts/demo.py`, the green patch appears inside the
open top storey with the roof cut away over it, at 3.581 m against a
ground floor at 0.686 m.

No production code changed. What was missing was coverage:
`tests/test_storey_placement.py` swept the labelled-room route across
three storeys but not the colour route, which is the one the reference
house's roof garden actually arrives by. Four tests were added there. Both
were checked against a deliberate regression -- restricting
`floor.planting` to `index == 0` fails two of them, and dropping
`floor.planting` from `open_to_sky` fails the third -- so they bite rather
than merely pass.

Suite 894 passed. `output_scorecard.py` 10 of 30, `open_air_accuracy.py`
80.5% IoU / 96.3% recall / 83.0% precision over 48 scored sheets: all
unchanged, as a test-only change must leave them.

## Room squaring is not failing, and the log was saying it was

A run over the reference sheet printed *"squaring moved a room's area too
far"* forty-three times, which reads as a building whose rooms are mostly
built from raw traces. Chased, and it was not that.

Squaring is offered every contour the segmenter produces, long before
anything filters them by real-world size. Of 190 outlines offered on that
sheet, 82 were refused -- but their **median area is 1,900 px², which at
26.28 px/ft is 2.75 square feet**. They are specks. The pipeline discards
everything under 12 sq ft a few steps later, so they never become rooms
and never reach the model.

Counting only outlines large enough to survive that filter:

| | |
| --- | --- |
| Outlines offered to squaring | 190 |
| Large enough to become a room | 66 |
| **Of those, refused** | **7 (11%)** |
| Rooms in the finished model | 62 |

And the seven are refused correctly. Their vertex counts are 88, 94, 59,
43, 38, 32 and 23 -- convoluted blobs rather than rooms, where forcing a
rectilinear loop really would move the area more than a fifth. The guard
is doing its job.

Across 25 CubiCasa plans the picture is the same and milder: 91% square
cleanly, and the refusals are 5% with fewer than four vertices and 4%
collapsing to fewer than four lines. **The drift ceiling never fires on
CubiCasa at all** -- 0 of 477 outlines, with the 99th percentile of drift
at 0.173 against a ceiling of 0.2.

Two hypotheses were tested on the way and both were wrong: it is not
`refine_windows` mangling the outlines (refusals are identical with it
disabled, though it does halve the window pixels, which is worth its own
look), and it is not resolution (at 400 dpi the reference sheet has zero
drift refusals among the contours my survey reached).

What was actually wrong was the message. It logged a speck at the same
volume as a room, so forty-three fragments read as forty-three broken
rooms. It now reports the region's size in squared wall-thicknesses and
drops to debug below sixteen of them: the same run prints **7**, which is
the true number, and each one is a room worth knowing about.

## The outline-wall augmentation worked, and cost more than it bought

`unfill_walls` was added because outline-drawn walls were much the worst
of eight conventions. A 24-epoch run at 512 followed, all epochs
completed, best at epoch 20 with a validation Dice of 0.7726 against the
installed checkpoint's 0.7757. Epochs 21 to 24 gave 0.767, 0.771, 0.772,
0.772 -- plateaued, not cut short.

**On the convention it was aimed at, it worked:**

| Convention | Installed | Retrained | |
| --- | --- | --- | --- |
| **Outline walls** | 0.534 | **0.702** | **+0.168** |
| Reversed print | 0.014 | 0.429 | +0.415 |
| Hatched walls | 0.688 | 0.722 | +0.034 |
| Photocopied | 0.746 | 0.728 | −0.018 |
| Solid poché | 0.811 | 0.789 | −0.022 |
| **As drawn** | **0.747** | 0.718 | **−0.029** |
| Finer pen | 0.752 | 0.676 | −0.076 |

The spread across the eight collapsed from 0.28 to **0.045**. The model
became *more even*, not better, which is what an augmentation of this kind
should do.

**On the corpus we can actually measure, it is worse:**

| | Installed | Retrained | |
| --- | --- | --- | --- |
| Median class IoU, pooled | **0.713** | 0.673 | −0.040 |
| circulation | **0.733** | 0.579 | **−0.154** |
| kitchen | **0.778** | 0.699 | −0.079 |
| bedroom | 0.713 | **0.766** | +0.053 |
| wall | **0.697** | 0.673 | −0.024 |
| window | 0.089 | 0.093 | +0.004 |
| Wall coverage | 96.6% | **97.4%** | +0.8 |
| Wall agreement | **92.2%** | 90.7% | −1.5 |
| **Window recall** | **62.1%** | 57.4% | **−4.7** |
| Window precision | **43.5%** | 41.4% | −2.1 |
| Scale median error | 17.3% | **16.3%** | better |
| Scale within a fifth | 33/48 | 33/48 | — |

**The installed checkpoint is kept.** The gains are on conventions
simulated by a transform written for the purpose; the losses are against
CubiCasa's own annotations. Trading 4.7 points of window recall and 0.154
of circulation IoU for robustness to a convention we have no real
examples of is the wrong way round while windows are still the weakest
thing in the project.

This is not a verdict on the augmentation, which did exactly what it was
built to do. `unfill` ran at 0.3, putting about a third of training draws
through hollow walls; a lower rate would likely buy much of the
robustness for less of the cost, and that is one run away from being
known. The checkpoint is kept rather than discarded.

Worth recording separately: **validation Dice fell while the run
succeeded.** 0.7757 to 0.7726, because the validation split is drawn from
the same convention as the training set and so measures the thing that
got slightly worse rather than the thing that got much better. Anyone
judging a run of this kind on Dice alone would have called it a failure.

## What a corpus of real Indian and American plans found

BRIDGE: ~2,400 plans collected from listing sites, sampled at 60 here.
Unlike CVC-FP it carries **no structural ground truth** -- its XML holds
Pascal VOC boxes for symbols and region captions like "master bedroom has
a double bed, sofa" -- so nothing here can be scored against it. What it
is good for is finding out whether the pipeline survives contact with the
drawings people actually publish, and it found three faults in an hour.

**Every one of them crashed.** `.gif` was not in `IMAGE_SUFFIXES`, so the
pipeline handed each sheet to the PDF rasterizer and got "unable to get
page count" back -- while `read_image` was reading all 60 perfectly well.
Every plan in BRIDGE is a GIF. With the suffix added: **60 of 60
reconstruct, 57 with usable geometry**, median 26 walls and 11 rooms.

**Half the world's dimensions were unreadable.** These sheets write
`12'-6" x 13'-8"`, and the hyphen between feet and inches was not in the
pattern, so the pair parsed as nothing. Nor did `12'-8" x 14'` parse,
because inches were mandatory and a room written `14'` is fourteen feet
exactly. Both forms are now read, which gained one more pair even on the
reference sheet, 14 to 15.

**OCR has a resolution floor and these sheets sit under it.** At 600 px
across, a plan's dimension text is present and unreadable -- Tesseract
returned `12-6" x 13-8`, losing the foot marks entirely. `read_text_boxes`
now enlarges a sheet whose longest side is under 1200 px, capped at twice:

| Enlargement | Lines read | Dimension pairs | Sheets with any |
| --- | --- | --- | --- |
| none | 103 | 1 | 1/30 |
| **2x** | 154 | **8** | **6/30** |
| 3x | 179 | 8 | 6/30 |

Three times reads no more than two, because interpolation cannot invent
strokes that were never sampled. End to end on the 60 sheets, **7 now take
their scale from printed dimensions where none did before**, and the
number reporting a confident size goes from 19 to 24.

CubiCasa is unchanged by all of this at 17.3% median error and 33 of 48
within a fifth, which is the point: none of it is tuned to anything, and
the sheets that were already readable read the same.

Worth noting what this is. Reading a drawing's own printed sizes is the
one route to better scale that does not fit a constant to a dataset, and
it needed no convention detection at all -- just reading the text that was
already on the page.

## A second drafting tradition, at last

CVC-FP: 122 scanned plans in four subsets that differ deliberately in
origin, style, quality and resolution, against CubiCasa's 5,000 sheets
from one Finnish source. Read by `planto3d/cvc_fp.py`. Forty times
smaller and much harder, which is the point.

**What it cannot settle, said first.** There is no metric ground truth
anywhere in its 122 annotations, so **the largest failure on the
end-to-end scorecard is the one this corpus cannot judge**. It also
labels every space simply "Room" with no type, and carries no floor
grouping. It tests walls, rooms and openings, and nothing else.

Over 30 sheets, against the same measurement on CubiCasa:

| | CVC-FP | CubiCasa |
| --- | --- | --- |
| **Wall coverage** | **96.7%** | 96.6% |
| Wall agreement | 77.5% | 92.2% |
| Sheets yielding usable geometry | **30/30** | — |
| wall IoU | 0.636 | 0.697 |
| room IoU | 0.530 | 0.717 |
| **window IoU** | **0.239** | 0.089 |
| door IoU | 0.136 | 0.560 |

> **Discrepancy note (wall coverage, wall agreement):**
> `scripts/cvc_fp_accuracy.py` now exists and measures wall coverage at
> **96.3%** (not 96.7%) and wall agreement at **69.0%** (not 77.5%). The
> difference is in the painting/dilation method of the original throwaway
> code, which was not recorded; the script's figures are the reproducible
> ones. The historical numbers above are kept as measured at the time; the
> interpretation below still holds directionally.

**Walls generalise, and that is the headline.** Coverage on a drawing
tradition the model has never seen is 96.7% against 96.6% on the one it
was trained on. Every sheet yields usable geometry. Agreement falls to
77.5%, so more is invented, but essentially none of the real wall is
missed. The thing the whole reconstruction rests on survives a change of
convention.

**Windows get nearly three times better**, 0.089 to 0.239. That is the
strongest evidence yet for what has been suspected all along: windows do
not fail because the model cannot see them, they fail because CubiCasa
draws them at 0.11% of a page. CVC-FP draws them at 0.9% and at 1.75 wall
thicknesses deep against CubiCasa's 1.33, and the model finds them. The
window problem is substantially a property of the training corpus rather
than of the network.

**The door collapse is not a door collapse.** It looked like the model
failing badly, and it is the two corpora meaning different things:

| | Median span | Median depth | Aspect |
| --- | --- | --- | --- |
| CubiCasa door | 3.62 gauges | **0.79** | 4.6 — a thin strip |
| CVC-FP door | 3.50 gauges | **3.19** | 1.1 — nearly square |

CubiCasa annotates the **door leaf** sitting in the wall. CVC-FP
annotates the **swing arc**, the quarter-circle the door sweeps into the
room. A model trained on one and scored against the other cannot do well,
and the 0.136 says nothing about whether it finds doors. Any figure
comparing door or opening accuracy across these two corpora is measuring
the annotation guideline, not the model.

The `outdoor` row is a mapping artefact for the same reason and is left
out above: CVC-FP's only named space is "Parking", which this loader maps
to `OUTDOOR`, and 81 small polygons across 75 sheets is not the same
thing as CubiCasa's balconies and terraces.

## What a change of drafting convention actually costs

Every figure in this audit rests on CubiCasa5K, which is one drafting
tradition. Sourcing a second corpus is the real answer to "does this
generalise" and it is expensive, so this is the cheap one first: take the
sheets we have and redraw them the way other conventions draw them,
holding the annotation fixed so the ground truth stays valid. Only the
inking changes -- same walls, same rooms -- so any drop is caused by the
convention alone. `scripts/convention_stress.py`, 15 sheets:

| Convention | Wall IoU | Wall recall | Room IoU | Reconstructs |
| --- | --- | --- | --- | --- |
| Solid poché walls | **0.811** | 0.966 | 0.975 | 13/15 |
| Finer pen | 0.752 | 0.851 | 0.954 | 13/15 |
| **As drawn** | **0.747** | 0.899 | 0.961 | 14/15 |
| Photocopied | 0.746 | 0.897 | 0.961 | 14/15 |
| Toned paper | 0.740 | 0.881 | 0.960 | 14/15 |
| Heavier pen | 0.732 | 0.917 | 0.961 | 14/15 |
| Hatched walls | 0.688 | 0.864 | 0.949 | 13/15 |
| **Outline walls** | **0.534** | 0.689 | 0.933 | 15/15 |
| **Reversed print** | **0.014** | 0.017 | 0.228 | **4/15** |

Three things follow, and the first two are more reassuring than expected.

**The model is not fragile.** Pen weight, paper tone, photocopying and
JPEG all land within 0.015 of the drawing as issued. Whatever else is
wrong, it is not that the segmenter has memorised one rendering.

**Hatched walls were predicted to be the problem and are not.** The
concern was that a model trained on uniform wall fills would read a
hatched partition as background. It reads them as wall: recall falls from
0.899 to 0.864, a single point, and IoU by 0.059. Nor is the premise
quite right -- CubiCasa's own walls are far from uniform, running from
0.05 to 0.95 dark-ink share with a median of 0.72, so the model has
already seen walls drawn as outlines and walls drawn solid. Targeted
hatching augmentation would be solving a problem worth 0.059, and it
would cost a retrain. **Not recommended.**

**Two real failures, one of them free to fix.** Outline walls -- two
lines with white between them, and no fill at all -- cost 0.214 of wall
IoU and take recall to 0.689. A reversed print destroys the model
outright: 0.014 IoU, and 4 sheets of 15 still reconstructable.

The outline case needs the model, so `training/augment.py` gained
`unfill_walls`, which redraws a filled wall hollow using the mask to find
it -- the wall stays labelled a wall, only its inking changes. It is
armed at 0.3 and **takes effect at the next training run**, whenever that
happens; nothing in the shipped checkpoint has seen it yet, so the 0.534
above still stands.

### Reversed prints are now turned the right way up

A blueprint, a negative scan or a dark-mode export carries the same
drawing with its tones inverted, and it was the single most damaging
thing that could happen to a sheet without changing a line of it. It
needs no model work at all, because the two populations do not come close
to overlapping: across 66 sheets from two drawing sets the lowest
ordinary median grey is **195** and the highest reversed one is **60**.
`read_image` now reverses anything below 128, in the middle of that gap.

| | Wall IoU | Reconstructs |
| --- | --- | --- |
| Ordinary sheet | 0.747 | 14/15 |
| Reversed, before | 0.014 | 4/15 |
| **Reversed, after** | **0.747** | **14/15** |

Identical to an ordinary sheet, which is what it should be: it is the
same drawing.

### What this does and does not tell us

It is a lower bound on the damage, not an estimate of it. A real
convention differs in more ways than can be simulated -- symbol
vocabulary, annotation habits, what is drawn at all, what is left to the
reader -- and a transform written by the same person who reads the result
is not an independent test. Gap #4 stands. What this narrows is the
*scope* of a second corpus: it should be chosen for outline-drawn walls
and unfamiliar symbol sets rather than for scan quality or wall hatching,
which the model already handles.

## The five sheets that split wrong, and why

`scripts/split_accuracy.py` over 60 CubiCasa sheets: **55/60 exact (92%)**,
precision 80%, recall 86%. Nobody had looked at the five. They are not one
fault, they are three, and only one of them is a tuning problem.

| Sheet | Wanted | Got | Failure mode |
| --- | --- | --- | --- |
| 12787 | 1 | 2 | boundary rule cut a single apartment -- **fixed** |
| 3720 | 1 | 2 | boundary rule cut a single apartment -- **fixed** |
| 8150 | 1 | 2 | a legend below the plan, behind a real gutter |
| 11378 | 2 | 1 | adjoining units, party wall, no gutter exists |
| 8583 | 2 | 1 | adjoining units, party wall, no gutter exists |

Two of the five are now fixed and the score is **57/60 at 92% precision,
86% recall**; see below. The three modes are kept described because the
two that remain are the harder ones.

### Which rule fired, and whether it was right

Splits were attributed to the rule that produced them, mirroring
`split_sheet`'s own order:

| Rule | Fired | Right | Wrong |
| --- | --- | --- | --- |
| Gutter, columns | 10 | **10** | 0 |
| Gutter, rows | 2 | 1 | 1 |
| **Boundary rule** | **3** | **1** | **2** |

Column gutters are perfect. The boundary-rule fallback in
`_boundary_cuts` is right one time in three, and causes two of the three
over-splits. Its docstring already records that it "split eleven single
plans across sixty sheets and got one of them right" before
`_pieces_look_like_plans` was added to guard it; the guard took it from
eleven firings to three, but not from wrong to right.

**Fixed, and without the trade.** Removing the rule entirely looked like
the option, but it costs sheet 9285 -- a genuine three-plan sheet only
that rule catches. Looking at how many lines it proposes separates the
cases perfectly:

| Sheet | Wanted | Boundary cuts | Right? |
| --- | --- | --- | --- |
| 12787 | 1 | **1** | no |
| 3720 | 1 | **1** | no |
| 9285 | 3 | **2** | yes |

A lone boundary line is not evidence of anything: a plot border, a
dimension band and a strong internal wall all look exactly like the edge
between two plans. Two lines leaving three comparable pieces is a much
stronger claim. Requiring a pair -- `MIN_BOUNDARY_CUTS` -- and leaving
the ordinary two-plan sheet to the gutter rules, which are already
perfect at it:

| | Exact | Precision | Recall |
| --- | --- | --- | --- |
| As shipped | 55/60 | 80% | 86% |
| Boundary rule disabled | 56/60 | 92% | **79%** |
| **Requiring a pair of cuts** | **57/60** | **92%** | **86%** |

Better than either alternative, and recall is untouched. The evidence is
three sheets, which is thin, but the argument does not rest on them: it
only ever makes the weakest rule more cautious, and the rule it defers
to is right ten times out of ten.

### Mode 1 -- the boundary rule cuts internal structure (12787, 3720)

Both are photographed or scanned brochure pages rather than CAD output:
12787 has a grey paper field, highlighter and biro on it, and
`paper_tones` finds no paper tone at all; 3720 is a scan with a grey room
fill. The boundary rule then cuts along a strong internal wall -- x=457
through the middle of a one-bedroom flat on 12787, x=283 through a studio
on 3720.

A saturated ink mask looked like the cause and **is not**. Ink fraction
across the 55 sheets that split correctly has a median of 0.154 and a
98th percentile of 0.587, so a heavily inked sheet is ordinary here; two
sheets with ink above 0.5 (10711, 11002) split correctly, and two of the
three sheets where `paper_tones` finds nothing (11002, 12403) are also
right. Neither signal separates the failures. The rule itself is the
fault, not the mask feeding it.

### Mode 2 -- a legend behind a real gutter (8150)

A hand-drawn sketch with a key beneath it: three symbols against the
words *pauhella-leivinuuni*, *porttöuuni*, *kamina*. The white band at
y=889 is a genuine gutter and the splitter is right to find it. What is
below the band is a legend, and it carries enough ink to satisfy
`_pieces_look_like_plans`, so the sheet comes back as a plan plus a
legend built as a second storey.

This is the mode most likely to matter beyond CubiCasa. A title block, a
key, a north point and a revision table are on almost every professional
sheet, and every one of them sits behind a real gutter.

### Mode 3 -- adjoining units with a party wall (11378, 8583)

Not a tuning failure. Both sheets are terraced blocks: 11378 carries two
units each marked *4H+K+S, 99,0 m²*, 8583 two marked *C5 4H+K+S 101,0
m²*. The units share a party wall, so the drawings touch and **there is
no gutter at any threshold**. On 11378 the ink profile has no quiet band
anywhere on either axis except the outer margins; on 8583 the only quiet
bands are three in the top third, which collapse to one cut that
`_pieces_look_like_plans` then rightly rejects.

Splitting these would need to detect repeated structure -- two copies of
one unit -- rather than whitespace, which is a different technique from
anything in `ingest.py`.

It is also worth asking whether they should be split at all. The ground
truth here is CubiCasa's count of `Floor` groups, and on these two sheets
that counts **dwellings, not storeys**: they are two homes side by side
on one level, and reconstructing them as one building with a party wall
down the middle is arguably the more faithful answer. Two of the five
failures may be a disagreement about the question rather than the answer.

## Diagonal walls are not worth recovering

The extractor opens the wall mask horizontally and again vertically and
keeps what survives either, so a genuinely diagonal wall survives neither.
That has been a documented limitation from the start. What it costs,
measured on the annotations over 40 plans:

| | |
| --- | --- |
| Wall pixels surviving neither opening | **2.7%** |
| Median plan | 2.3% |
| Plans losing more than 10% | **0 of 40** |

Recovering 2.7% of wall means redesigning the extraction model, against a
window class sitting at IoU 0.096 while every other class is 0.52 to
0.77. It is the wrong trade, and it is recorded here so it is not
reconsidered without new evidence.

## Two things measured and deliberately not changed

**Preferring doors harder does not help.** Doors carry a +1.6% bias
against the wall estimate's -19.7%, so using them wherever possible looks
obvious. Swept over 24 plans, the current threshold of three is already
the best available on every measure:

| doors required | plans using doors | median error | within 20% | worst |
| --- | --- | --- | --- | --- |
| 5 | 5 | 18.8% | 14/24 | 56.8% |
| 4 | 8 | 18.3% | 14/24 | 56.8% |
| **3** | **11** | **18.0%** | **15/24** | **51.0%** |
| 2 | 15 | 18.1% | 14/24 | 51.0% |
| 1 | 21 | 18.1% | 14/24 | 61.1% |

Fewer doors means a noisier median, and the noise cancels the bias
advantage exactly. Recorded so it is not tried again.

**The wall-thickness constant stays at 9 inches.** See above -- the two
populations disagree by 26 points and fitting either breaks the other.

**Parapet-height enclosure cannot be evidence for open air. It is
circular.** The spec proposed that "a room enclosed by parapet-height walls
on a top storey is open whatever it is called". Checked before
implementing: `PARAPET_HEIGHT_FT` (`planto3d/extrude.py:61`) is a
constant, and `_open_air_walls` (`:943`) decides which walls are parapets
by asking `is_open_to_sky(room)` at line 957 -- the very question the
candidate proposes to answer. Parapet height is an **output** of the
open-air decision, never an input to it.

Nor could it be otherwise. A floor plan is a horizontal section; it
carries no height information about anything, and the segmenter has one
WALL class with no notion of how tall a wall is. There is nothing upstream
that distinguishes a parapet from a full-height wall, so no amount of
implementation would make this candidate work. **Do not retry this** unless
elevations are being read, which the spec places out of scope.

**Neighbour propagation does not ship: it spends precision and buys
nothing.** The spec's third candidate -- "a room whose neighbours are all
outdoor probably is too" -- was implemented as
`features.propagate_open(rooms, scale)`: a single pass, computed from the
input state, that opens a room carrying neither a printed label nor a
predicted type when every room within `NEIGHBOUR_GAP_FT` of it already
reads open and at least one does. It was wired into `extrude.open_to_sky`
and scored with `scripts/open_air_accuracy.py` over the 60-plan CubiCasa
corpus, 671 rooms, pooled in pixels:

| neighbour gap | IoU | recall | precision |
| --- | --- | --- | --- |
| off (baseline) | 72.5% | 91.4% | **77.9%** |
| 0.5 ft | 72.6% | 91.9% | 77.6% |
| 1.0 ft | 72.4% | 91.4% | 77.7% |
| 2.0 ft | 72.5% | 91.4% | 77.9% |

The gap was swept rather than chosen because it is a fitted constant. At
2.0 ft the rule never fires at all -- a wider gap gives each room more
neighbours, and "all of them open" gets harder to satisfy, so the
candidate disappears rather than growing. At 1.0 ft it moves recall not at
all and costs 0.2 points of precision: every pixel it added was wrong. At
0.5 ft it buys 0.5 points of recall for 0.3 points of precision, moving
IoU by a tenth of a point.

Precision is the weak leg here -- recall was already 91.4% and roughly a
fifth of built-open pixels are wrong -- and the plan's rule was explicit
that a candidate lowering precision does not ship whatever it does to
recall, because a roof wrongly removed reads worse in a render than one
wrongly left on. Precision falls at every value where the rule does
anything. The code and its five tests were reverted. **Do not retry this.**

**Fuzzy matching of room labels against the feature vocabulary does not
ship.** OCR mangles labels, so matching them by edit distance instead of
by substring looks obvious. Measured over 67 plans (the 60 in
`data/bridge` plus the 7 demo sheets), 647 text boxes, scoring each box's
normalised text against all 460 keywords with
`difflib.SequenceMatcher.ratio()` and counting only hits the current
substring rule does not already make:

| cutoff | new hits | of which right |
| --- | --- | --- |
| 0.90 | 0 | — |
| 0.85 | 5 | 3 (`ath`→BATH, `dow.`→DOWN, `Ram,`→RAMP) |
| 0.82 | 9 | 3 |
| 0.80 | 15 | 5 |
| 0.75 | 37 | 5 |

**There is no cutoff that works, and the plan that motivated this proves
it.** The wrapped label on
`demo_plans/1-BEST-measured-scale-50walls-19rooms.gif` reads
`per to Be ow`, which scores **0.800** against OPEN TO BELOW. Two boxes
reading `Master Bedroom` score **0.828** against MASTER BATHROOM. Any
cutoff low enough to build the void is low enough to tile a bedroom
floor, and the same band brings `DEN`→DN (a staircase in a den),
`GATH. ROOM`→BATHROOM, and two-letter fragments `oT` and `ob`→OTB, each
of which punches a hole through a floor slab. A minimum length on the
matched string does not separate them either: the false positive is
14 characters and the true one is 12.

Joining stacked lines was kept because it needs no cutoff at all -- it is
geometry, and what it recovers matches the vocabulary exactly. Fuzziness
was the part that could not be made safe. **Do not retry this.**

## Scale error, broken down by source, and four routes tried that did not close it

`scripts/scale_accuracy.py` gained a per-source table on this branch. Over
the same 30 ground-truthed CubiCasa sheets used throughout this section, it
shows the two inferred sources fail in opposite ways:

```
source        plans  median err  within 20%     bias
doors            15      10.4%      14/15    +0.9%
walls            15      20.2%       6/15   -20.2%
```

Pooled: **17.7% median, 20/30 within a fifth — unchanged by this branch.**
The population splits evenly, and the pooled figure had been hiding that
**doors are essentially unbiased and accurate, while walls are
systematically undersized by a fifth.** The `bias` column is what makes it
legible: walls' bias (−20.2%) very nearly equals their median error
(20.2%), meaning almost all of it points one way — a mis-centred constant,
not scatter. Doors' bias (+0.9%) against a 10.4% median is scatter without
direction, which is a different problem and not one a constant can fix.

Four things were tried against this picture. Three made it worse or did
nothing; one works and was deliberately not taken.

**Correcting the door constant alone makes things worse.** Substituting
CubiCasa's own 2'3" for the assumed 2'6" moves the pooled figure from
17.7% to **20.1%**, and sheets within a fifth from 20/30 to **15/30**. The
bias table explains why: doors were already right. The plausible
mechanism is that the detector measures an opening span while the
annotation records the door leaf — different things, coincidentally close
at the shipped constant. **Do not retry this.**

**Widening the door-width acceptance band does not help.** The 15 sheets
that fall back to walls are not short of openings — they carry 6 to 17
each against a minimum of 3. Sweeping `MIN_DOOR_GAUGES`/`MAX_DOOR_GAUGES`
from (1.5, 8.0) through (1.0, 10.0), (0.8, 12.0) and (0.5, 16.0) moves only
15→17 sheets onto the door route and makes the pooled figure *worse*,
17.7% → 18.1%. **Do not retry this.**

**Wall gauge is a structurally poor scale estimator, and no constant
repairs it.** The wall-derived error tracks the measured gauge almost
exactly — gauge 16 px gives about −33%, gauge 20 px about −19%, gauge 28
px about +20% — while the true scale on those same sheets stays near
31–33 px/ft. So the drawings share a scale and their *real wall
thicknesses differ*. Implied real wall thickness across the 30 sheets:

| | ft |
| --- | --- |
| min | 0.478 |
| p25 | 0.598 |
| median | 0.648 |
| p75 | 0.803 |
| max | 1.176 |

The interquartile spread is **±16% of the median**, which is the floor
under any single constant — it cannot be tuned away, only lived with.

**Element sizes can now vary by drafting tradition, and the detector
never fires here.** `planto3d/calibrate.py` gained `CONVENTIONS`,
`detect_convention` and `element_sizes`, selecting per-tradition door and
wall values from Finnish/Swedish room names. It fired on **0 of 30**
CubiCasa sheets, because these rasters carry almost no readable text —
consistent with the OCR yield already recorded above. Pooled accuracy is
therefore byte-for-byte unchanged. The mechanism is unit-tested and
correct; it is simply unexercised by any corpus currently available.

**The one measured route to the spec's gate was deliberately not taken.**
Substituting 0.633 ft for the assumed 0.75 ft wall thickness *globally*
moves the pooled figure from 17.7% to **9.9%** and sheets within a fifth
from 20/30 to **23/30**. It is available and unspent. It was rejected
because it fits the constant to the only corpus with metric ground truth,
and would silently degrade the Indian and Spanish plans that cannot be
checked — the same standing rule recorded above under "the one that is a
trap". The number is recorded so the decision stays visible rather than
lost.

**Independent evidence that 0.633 ft is tradition-specific, from a second
corpus.** The rejection above was made on principle -- CubiCasa is the only
corpus with metric ground truth, so fitting to it cannot be checked
elsewhere. It can now be checked. Running the reference house
(`data/soni_residence`, an Indian masonry building) through `extract` this
session reports `scale_source: dimensions`, `scale_assumed: False`: it takes
its scale from printed dimensions, so the wall constant does not affect it
either way. What it yields on the way is the useful part -- a measured wall
gauge of 24.0 px against a true scale of 25.90 px/ft, so an implied real
wall thickness of **0.93 ft (11 inches)**.

Against CubiCasa's median implied thickness of 0.648 ft, that is a **43%
difference between the two traditions**, measured rather than assumed.
Adopting 0.633 ft globally would have passed the gate on CubiCasa while
making the wall-derived estimate badly wrong for any masonry plan that
lacks printed dimensions. The standing rule held; this is the evidence for
it. **Do not retry this.**

0.633 ft and the 0.648 ft in `CONVENTIONS["nordic"]` are not the same
quantity, though they are close: 0.633 is the value that *minimises pooled
scale error* over the 30 sheets (found by search, the number above), while
0.648 is the plain *median* of the implied-wall-thickness table earlier in
this section. `CONVENTIONS["nordic"]` uses the median deliberately — a
raw, defensible statistic of the corpus, not a constant fitted against the
error metric it would then be judged by.

## The ceiling

Room squaring came out at **exactly 100% on a clean annotation mask** and
92.4% on the predicted ones. That gap is no longer in the geometry code —
it is in the mask.

The same is true of windows, of the rooms that still come back rough, and
of most of the scale error. **The pipeline's accuracy is now bounded by
the segmenter**, and only two things lift that bound:

1. **Higher training resolution.** Unblocked, one run, helps everything.
2. **More varied training data.** CubiCasa is 5,000 Finnish apartments,
   and a model trained only on those reads other traditions less well.
   No amount of geometry code fixes that.

Everything else on the list is polish under a ceiling neither raises.
