# Where the project stands

A handoff note, so work can resume from a fresh session without the
conversation that produced it. Read alongside the [README](../README.md),
which covers what the project does and how to run it.

Last updated after the session that built the envelope-closing stage.

## Done and working

The whole pipeline runs end to end: PDF, PNG or JPEG in, a materialled
`.glb` plus six rendered views out.

| Stage | State |
| --- | --- |
| Ingestion, cropping | Complete. Crops from borders common to every page |
| Segmentation | U-Net+ResNet34 trained; classical baseline as fallback |
| Wall, room, opening extraction | Complete, including envelope closing |
| Scale calibration | Dimensions, then doors, then walls, then a ratio |
| Room labelling | From OCR, with the ink isolated from hatching |
| 3D extrusion | Walls, slabs, roof, plinth, stairs, railings, frames |
| Vocabulary | 335 feature keywords, 139 floor-finish keywords |
| Materials | Seventeen surfaces, including per-room floor finishes |
| Views | Top, front, back, left, right, aerial |
| Web app | Gradio, multi-file upload, live 3D viewer |
| Photoreal guides | Depth, edge and shaded renders ready |

384 tests. Around 4,500 lines.

## The photoreal pass has been run once

It worked. Depth conditioning held the massing: the twin roof terraces, the
stepped form and the plinth all came through recognisably from the model.
Conditioning at **0.5** gave the richest image; higher values held the
geometry more tightly but looked increasingly like a shaded model.

What the first run lacked, measured against the reference the project is
aiming at, was almost entirely prompt and camera rather than geometry:

| Wanted | First run gave |
| --- | --- |
| Warm limestone cladding | cool white concrete |
| Amber interior and landscape lighting | cool blue-white |
| Hedges, planters, uplit boundary | bare lawn |
| Two cars | none |
| Camera around 26 degrees, facade-led | 42 degrees, roof-led |

Both have since been changed. The prompt now names lighting as fittings
rather than as a mood -- wall washers, uplighting, amber interiors -- and
describes stone coursing and slim dark frames. The guide camera dropped from
30 to 26 degrees so the facade dominates rather than the roof.

Site features stay conditional on what the plan actually shows. A render
claiming a garden the drawing does not have has stopped describing the
building, which a test enforces.

**Next run:** regenerate the guides and try again, keeping conditioning near
0.5.

```bash
python scripts/run_pipeline.py "data/soni_residence/DOC-20260817-WA0027.PDF" output_unet --checkpoint models/unet_cubicasa.pt
python -c "from planto3d.photoreal import build_guides; build_guides('output_unet/house.glb','output_unet/guides')"
```

Then upload `output_unet/guides/guide-depth.png` to the notebook. Section 5
sweeps the conditioning strength so the trade-off between holding the
geometry and enriching the image can be seen rather than guessed.

## The biggest methodological weakness

**The geometry layer is tuned against one building.** Wall fill 99, room fill
200, the working resolution, the crop consensus, the envelope tolerances and
the colour signals were all measured from the Soni Residence sheets. That is
overfitting to a single sample, and it is the first thing a careful reader
should challenge.

The two layers stand differently, and the distinction is worth making
explicitly rather than letting a reader assume the worse of both:

- The **segmentation model** was trained on 5,000 CubiCasa plans and scored
  on a held-out split it never saw. That is properly validated.
- The **geometry and heuristics** around it were fitted to one drawing set.

Some constants are derived rather than fitted -- a 2'6" door is a standard,
not a measurement of this house -- and the scale fallback was validated
against the reference rather than tuned to it: doors were predicted to work,
then checked, giving 27.2 px/ft against a true 28.15. But the grey levels,
the resolution and the crop consensus are genuinely fitted.

**This has now been measured.** `scripts/generalisation_test.py` runs the
stages over CubiCasa samples drafted by other people in other conventions.
Predictions were recorded before the first run so they could not be
rationalised afterwards, and all three held:

| | U-Net | Classical baseline | Colour signals |
| --- | --- | --- | --- |
| Usable geometry | 12 of 12 samples | -- | -- |
| Mean wall pixels | 7.2% | 0.0% | -- |
| Mean walls found | 31 | 1 | -- |
| Detected anything | -- | no walls at all on 10 of 12 | windows 5/12, planting 1/12 |

Across sheets from 954x979 to 3536x1879. So the trained model generalises,
and the heuristics around it do not -- which is now measured rather than
asserted, and is the honest way to present the project.

The reading to take from this: the segmentation model is the contribution,
the classical baseline is a scaffold that only ever worked on clean CAD
sheets, and the colour signals are one drafting office's convention rather
than a general technique.

## One image is assumed to be one storey

Found by running a random CubiCasa sample end to end rather than testing the
stages in isolation, which is why it had not surfaced before.

Many sheets lay several floor plans **side by side on a single image** --
basement, ground and first floor in a row. The pipeline treats each image as
one storey, so it reconstructs all three plans as a single flat floor: on
sample 9285 that produced 135 walls in a long thin strip that is three
buildings wide and one storey tall.

Nothing detects this. The pipeline is quite happy, reports plausible numbers,
and produces a model that is confidently wrong -- which is worse than
failing. The reference drawings hid it because they put one floor per sheet,
as an architectural drawing set does.

`ingest.split_sheet` now does this, cutting at gutters -- bands of near-empty
page far wider than the gaps inside a drawing, measured as a fraction of the
sheet so the rule holds at any resolution. It works on clean sheets and
leaves single plans alone, which ten tests pin down.

**It does not fire on CubiCasa's own exports.** Those images carry a
transparency checkerboard baked into the colour channels, so no column is
ever close to empty -- the minimum column ink on sample 9285 is 8%, and a
gutter simply cannot look like one. Reading transparency properly was added
for the same reason and helps files that carry a real alpha channel, but not
these.

So the split is real and useful for ordinary sheets, and CubiCasa's
multi-plan exports still need splitting by hand. Detecting a checkerboard
background and erasing it before the gutter search is the next step.

## Known limitations

These are understood, not mysteries. Each is a deliberate boundary or a
measured shortfall.

- **Photographs segment poorly.** A compressed phone photo of a plan
  produces mush. The pipeline targets clean digital drawings.
- **Greyscale plans lose two signals.** Windows are read from cyan strips
  and planting from green ink; a greyscale sheet falls back to the model,
  which scores 0.12 IoU on windows.
- **Diagonal walls are not recovered at all.** The orientation filters that
  make walls exactly axis-aligned erase anything diagonal. Correct for
  rectilinear plans, wrong for anything else. Pinned by a test.
- **Some windows are still lost**, where their host wall was missed even
  after envelope closing.
- **Multi-storey alignment assumes a shared drawing frame**, true within one
  drawing set and not across separately drafted plans.

## Things learned the hard way

Recorded because each cost real time to find and would be easy to undo.

- **Room mask must not be morphologically closed.** Room fill flows through
  every doorway; only the hairline door leaf separates one room from the
  next. Closing bridges exactly those lines and collapsed 15 rooms into one
  blob holding 96% of the fill.
- **Contours simplify by absolute pixels, not a share of perimeter.** A
  relative tolerance gives the longest contour — the building footprint —
  the coarsest treatment, cutting diagonal shortcuts across corners.
- **Walls decompose by orientation, not by contour.** Walls meet at corners,
  so the wall class forms one connected ring per enclosure; tracing its
  contour returns the building outline, not individual walls.
- **OCR needs the ink isolated.** Labels printed over hatching are
  unreadable, and those are disproportionately the outdoor ones, because
  outdoor areas are what drafters hatch. Keeping only near-black ink took a
  sheet from 24 words to 71.
- **Glass must not be shaded diffusely.** It mirrors the sky, so facing away
  from the key light it dropped to ambient and every window on the shaded
  side rendered as a black hole.
- **Ground cover is found by three sources that overlap.** Segmented rooms,
  dimension labels and colour frequently find the same area; laying all
  three down drew the terrace garden three times over.
- **Under-capture was an OCR problem, not a geometry one.** Sweeping the
  planting closing kernel across a 6x range moved capture by four points.
- **A slab is the ceiling of the storey below it.** Built from its own
  footprint alone, it leaves rooms underneath open to the sky wherever the
  plan steps in.
- **Only two thirds of each perimeter had a wall on it.** Segmentation loses
  exterior runs wherever a drawing is busy, and every window that would have
  bound to a missing wall was dropped with it. Gaps are filled only where a
  room lies just inside, since a terrace edge is legitimately open.
- **A pale palette is useless in a web viewer.** Off-whites separated by a
  few percent are architecturally tasteful and all wash to the same
  pink-white under a viewer's bright lighting. Surfaces have to be spaced
  far enough apart in tone to survive it.
- **Low-resolution input looks like a colour bug.** Room names drive the
  finishes, planting, paving, railings and stairs, so a screenshot upload
  produces a plain grey model with nothing to explain why. The app now says
  so; the silent failure was worse than the limitation.
- **A verandah is not a balcony.** It is a covered space at ground level, and
  railing it puts a balustrade across the front door. The same care is needed
  for patio against deck, and courtyard against garden.

## The vocabulary

`planto3d/features.py` is where most of the project's domain knowledge lives.
Room labels drive geometry, not merely colour: water sinks into the ground, a
void removes the floor above it, an open edge is railed, stairs become a
flight, and interior floors take a finish from what the room is for.

It covers what a plan can actually carry rather than what one reference set
happened to use -- every balcony and terrace variant, loggia, lanai,
breezeway, portico; pools from lap to plunge to reflecting; gardens from zen
to kitchen to sunken; lightwells, airwells and ventilation courts; the full
staircase family down to half landings.

South Asian and Middle Eastern terms are included -- otla, osari, baramda,
chabutra, jharokha, chhat, baithak, majlis, diwan, deorhi, mumty, chajja --
because these drawings use them and no published floor-plan glossary covers
them. That gap is worth knowing about before trusting any external reference.

Two rules hold the whole thing together, and breaking either causes failures
that look like something else entirely:

- **Matching is longest-first across the entire vocabulary**, not within each
  rule. Ordering by rule cannot survive this many terms: "TERRACE GARDEN"
  contains "TERRACE", "POOL DECK" contains "DECK", "MASTER BATHROOM" contains
  "MASTER".
- **Labels are normalised before matching**, since drafters and OCR both
  mangle them. "W.C.", "DRESS/TOILET", "SIT-OUT" and the truncated "BAL" all
  have to match, and keywords are tested with and without spaces.

Short marks are matched as whole words only. "UP" is often the entire
annotation on a flight of stairs, but as a substring it would take GROUP,
CUP and DINING.

## Where to look

```
planto3d/features.py    what a room label means for the model
planto3d/extract.py     mask to geometry, including envelope closing
planto3d/calibrate.py   OCR, dimension parsing, the scale fallback chain
planto3d/extrude.py     everything that becomes 3D
planto3d/materials.py   the sixteen surfaces
docs/superpowers/       the original design spec and implementation plan
```

The commit history is the fullest record: each message explains why a change
was made and what it fixed, not merely what it touched.
