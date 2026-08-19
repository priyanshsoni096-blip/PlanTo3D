# Where the project stands

A handoff note, so work can resume from a fresh session without the
conversation that produced it. Read alongside the [README](../README.md),
which covers what the project does and how to run it.

Last updated after the session that taught the segmenter to predict room
types.

**The one thing to do next: retrain.** `notebooks/train_on_colab.ipynb` now
trains eleven classes instead of five, and until it has been run the room
types are code with no weights behind them. Everything else works as it did
in the meantime.

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

**Which notebook to use.** There are two, and the simpler one is the reliable
one:

- `photoreal_on_colab.ipynb` -- upload a depth guide, generate, download.
  This produced the good renders and is what to reach for.
- `app_on_colab.ipynb` -- the whole app on a GPU with the photoreal pass
  wired in and a public link. Better in principle, but it returned an HTML
  error page through the share tunnel on first use, which is usually the
  runtime being reclaimed or running out of memory rather than a code fault.
  Worth returning to; not worth blocking on.

**Next run:** regenerate the guides and try again, keeping conditioning near
0.5.

## The interface, for later

Two things are known and neither is urgent:

- The interactive viewer washes out masonry and cannot be told not to, so
  the result leads with a rendered view instead. That is settled.
- The desktop app cannot run the photoreal pass, because diffusion needs a
  GPU. `app_on_colab.ipynb` is the answer if it can be made reliable.

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

## Room function no longer needs text

This was the largest single obstacle to working on plans other than the one
the project started with, and it is worth stating plainly because it was
invisible for a long time.

Everything that makes the finished model look like a house rather than a
grey massing study -- floor finishes, planting, paving, railings, wet areas,
stairs -- hung off knowing what each room was *for*. That knowledge came
from OCR alone: read the word BATHROOM off the drawing, lay a tiled floor.

**Most drawings have no room names on them.** Over sixty CubiCasa plans the
pipeline read a room name on three. The rest print a disclaimer and a
watermark and nothing else; a typical sheet's only text is "SUUNTAA-ANTAVA,
EI MITTAKAAVASSA". Their rooms are identifiable, but from what is drawn
inside them -- a hob, a toilet, a sauna bench -- not from anything written.

So the great majority of plans were being reconstructed correctly and then
finished as bare floors. No amount of vocabulary work could have fixed it.
The 335-keyword vocabulary was matching against text that was not there, and
every hour spent widening it was aimed at the wrong target.

**The fix is to ask the model instead.** CubiCasa annotates a room type on
every space, and the loader was discarding all of it -- forty types
collapsed onto one ROOM class, on the reasoning that CubiCasa's Finnish
residential categories had no equivalent for a Temple or a Verandah. True,
but it threw away the bedroom, kitchen, bath and balcony along with them.

The classes now kept, grouped by what they change downstream rather than by
architectural category:

| Class | Absorbs | Because |
| --- | --- | --- |
| `bedroom` | Bedroom | Boarded floor |
| `kitchen` | Kitchen, Kitchenette, Scullery | Tiled, wet |
| `bath` | Bath, Shower, Sauna, SwimmingPool | Floor built to get wet |
| `storage` | Closet, Storage, Utility, Garage, TechnicalRoom | Hard floor, not lived in |
| `circulation` | Entry, Lobby, Hall, DraughtLobby | Moved through, not stayed in |
| `outdoor` | Balcony, Terrace, Porch, CoveredArea | **Earns a railing** |
| `room` | LivingRoom, Dining, Office, Undefined, anything unrecognised | No particular requirement |

A sauna is not a bathroom, but both want a floor built to get wet, and that
is the only distinction the geometry makes use of. An unrecognised type
becomes a generic room rather than background, so a vocabulary CubiCasa adds
later degrades to a plain floor instead of a hole in one.

Measured over sixty plans, these are not rare: kitchen on 55, outdoor on 48,
circulation on 47, bedroom on 45, bath on 46. Against names read on three.

**Where both exist, the printed name wins.** The segmenter is trained on
Finnish apartments and calls a verandah an outdoor space, which would rail
it like a balcony -- exactly the bug reported earlier. The drawing saying
VERANDAH is the better authority, so `features.feature_for` checks the label
first and falls back to the prediction.

**This needs a retrain to take effect.** Indices 0-4 are unchanged, so the
existing five-class checkpoint still loads and behaves exactly as it did: a
batch of eight plans reconstructs identically, every room simply arrives
uncategorised. `scripts/batch_evaluate.py` reports both routes side by side
and says outright when a checkpoint predates the room types.

One consequence worth knowing: room classes are traced separately rather
than together, because an open kitchen has no wall between it and the
dining area it opens onto. Tracing them as one class returned a single room
where there are plainly two.

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

**It does not fire on CubiCasa's own exports, and the reason is instructive.**

The first guess was a transparency checkerboard flattened into the colour
channels. That was wrong: sample 9285 is 40% pure white. Two fixes were built
on that wrong premise -- flattening real alpha channels onto white, and
detecting whatever light tone dominates a sheet rather than assuming paper is
white. Both are defensible robustness improvements and both are kept, but
neither addresses this.

The actual obstacle is that **each plan on the sheet is enclosed in its own
boundary box, and dimension lines run the full height**. Minimum column ink is
8% because something crosses every single column. There is no empty gutter to
find, and no threshold will conjure one.

What would work is looking for the boundary boxes themselves -- long
uninterrupted vertical rules, which is what actually separates the plans --
rather than looking for emptiness between them. That is the next attempt, and
it is a different algorithm rather than a tuning change.

Until then the split handles ordinary sheets, and CubiCasa's multi-plan
exports need splitting by hand.

## The three things asked for next

In priority order, with what is known about each so the work can start
rather than re-derive it.

**1. Good results on unseen plans.** The segmentation model already
generalises -- 12 of 12 CubiCasa samples produced usable geometry, and a
batch of 24 reconstructed 21. What does not transfer is everything tuned to
one drawing set.

The largest part of this has been dealt with: room function no longer needs
text, which is its own section above. It was the dominant failure by a wide
margin -- names read on 3 plans in 21 -- and it is the reason unfamiliar
sheets came out as grey massing studies. **It needs a retrain to take
effect.**

What remains after that:

- The **classical baseline** finds nothing on unfamiliar sheets: no walls at
  all on 10 of 12. It is a scaffold for clean CAD drawings and should be
  described as one rather than improved.
- The **colour-driven windows and planting** rarely fire -- windows on 5 of
  12, planting on 1. These are one drafting office's convention. The honest
  improvement is to give `refine_windows` a way to tell a sheet that marks
  windows in colour from one that does not, rather than assuming; the model
  already finds the openings without help.
- **Planting on greyscale plans** has no route at all, since it is colour
  only. The `outdoor` class is the obvious source once a retrained model
  supplies it.
- **Around five windows per set** are still lost where the host wall was
  never detected.

**2. Splitting a sheet that holds several storeys.** `ingest.split_sheet`
does this by finding empty gutters and works on ordinary sheets. It cannot
work on CubiCasa's, where boundary boxes and dimension lines cross every
column. Detecting the long vertical rules that bound each plan is the
approach that would, and it is a different algorithm rather than a threshold
change.

**3. The interface still shows a colourless model.** Diagnosed once as
resolution -- room names drive the finishes, and a screenshot upload names
nothing, which the app now warns about. Worth confirming that is the whole
story before changing anything else: run the pipeline on the original PDF
from the command line, open the `.glb` in Windows 3D Viewer, and compare with
what the app shows for the same file. If the command-line model has colour
and the app's does not, the fault is in the viewer or in what the app hands
it, not in the materials.

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
