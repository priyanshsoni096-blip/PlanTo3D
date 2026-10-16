# Bug report, 2026-10-16

Every entry here was found by running the code against real drawings, not
by reading it and imagining what might go wrong. Each one names the input
that produced it and the figure that changed when it was fixed, so any of
them can be re-run and disagreed with.

**How the hunt was done.** The pipeline was run over every corpus on this
machine -- 6 demo plans, 25 BRIDGE sheets (Indian drawings), 20 CVC-FP
sheets (Spanish/Catalan), the reference house -- recording crashes, empty
results, implausible buildings and roofs full of holes; and separately over
the awkward files a user will actually hand it: a greyscale scan, an image
with an alpha channel, an 80 px thumbnail, a 6,000 px scan, a blank page, a
black page, a reversed print, a one-pixel image, a non-ASCII filename, and
a text file renamed `.png`.

Nothing crashed on any of the 52 real sheets. What follows is what did go
wrong.

## Fixed

### 1. A covered terrace lost its roof

`demo_plans/3` prints **TERRASSE COUVERTE**. The vocabulary matched
`TERRASSE`, classified the room as open to the sky, and the finished model
had a hole in the roof over a terrace the drawing says is covered -- visible
in the render as a sunken, open bay.

The word that settles it was being ignored. A covered word, matched whole,
in any of the four languages the vocabulary already spans -- `COVERED`,
`ROOFED`, `SHELTERED`, `COUVERT(E)`, `CUBIERTO/A`, `TECHADO`, `KATETTU`,
`KATOLLINEN` -- now turns an open category into a **canopy**: open-sided,
with a roof. Nothing else moves, because covering changes nothing about a
porch or a verandah, which are paved ground either way. Open-to-sky IoU over
the 60 held-out plans is unchanged at 76.3%.

### 2. A blank page came back as a building

A blank white page, an all-black scan and a one-pixel image each produced a
model with **0 walls, 0 rooms, and a scale of exactly 32.0 px/ft** -- and
reported success. An empty building that claims to be a real one is worse
than a refusal, because nothing downstream can tell the two apart.

`extract` now refuses a page with no walls **and** no rooms, naming the file
and saying what to try instead. Rooms alone are still enough to build: a
sheet whose partitions are too faint to segment has a plan worth having.

### 3. A scale nobody measured was credited to the walls

The same blank pages reported their scale source as **"walls"**.
`wall_gauge` falls back to a reference 24 px when it cannot measure, and
24 px over a 9-inch wall is exactly the 32.0 px/ft the assumed drafting
ratio gives -- so the assumption came back wearing the clothes of a
measurement. On this project that is the worst kind of defect: every
downstream judgement about confidence reads that field.

A wall-derived scale is now only offered where a wall was actually read;
otherwise the source is `ratio` and `scale_assumed` is true.

### 4. A region built from a label could be larger than the sheet

BRIDGE sheet `116794.gif` is **498 x 372 px**. A region built from one of
its printed labels came out **336 x 568 px** -- taller than the page -- and
cut a hole in the roof **276% of the building's own footprint**.

Regions from labels are sized by multiplying printed dimensions by the
inferred scale and centring them on the text. Both inputs can be wrong, and
nothing bounded the result to the drawing. Regions are now clipped to the
page, and one larger than half the sheet is dropped with a log line: at that
size it is not telling us where anything is. The same sheet now reports 90%.

### 5. A garage had no roof

BRIDGE `116794.gif` and `117564.gif` carry one room each, labelled **3-Car
Garage**, and came out with **90% of the roof missing**. `GARAGE`
classifies as paving, every ground cover counted as open to the sky, and a
garage is certainly paved.

The first attempt was to roof paving that lies inside the building and keep
paving outside it open. Measured, it changed nothing on these sheets,
because the hole came from the *room* rather than from a region built from
a label -- worth recording, since it looked like the obvious fix.

What the drawing actually distinguishes is the word. A garage, a carport, a
car porch, covered, basement or stilt parking and a covered walkway are
roofed; a driveway, a courtyard, a patio and open parking are not. Roofed
names are now excluded from open-to-sky in any of the four languages the
vocabulary spans. Both sheets now report **0%**, while `11001.gif` keeps
the 12% of genuinely open area it had. Floor finishes come from a separate
map and are unchanged -- a garage is still concrete. Over the 60 held-out
plans, open-to-sky IoU and the scorecard are both unchanged (76.3%, 42 of
60), which is expected: CubiCasa sheets rarely print a readable garage
label.

### 6. "Not an image" was reported as a missing file

A text file renamed `plan.png` raised `FileNotFoundError: could not read
page image`, which sends the reader looking for a file that is sitting right
there. It now says the file is not an image the reader understands, and
names the formats that are.

### 6. The demo rendered the plainest scheme the project has

`scripts/demo.py` never passed the style, colour or landscaping to the
renderer, though `notebooks/run_on_colab.ipynb` did. The one command meant
to show the project off produced grey walls on bare ground. It takes
`--style`, `--colour` and `--landscaping` now, defaulting to the
best-looking combination rather than the plainest.

### 7. Two scripts crashed instead of answering --help

`crop_pages.py` raised `IndexError` and `random_plan.py` raised
`ValueError` trying to read `--help` as a seed -- after it had already
deleted its output directory at import time, so asking a question destroyed
the previous run. Both parse arguments properly now, and `random_plan`
parses before it clears anything. A test runs `--help` against all 23
scripts.

## The vocabulary, audited as a whole

460 keywords in 15 categories, matched by substring. Every keyword was run
against the matcher that actually sees it, and every category list was read
against what that category does to the building.

**Structurally sound:** no duplicate keywords, and no keyword shadowed by a
shorter one from another category.

**Six keywords fired inside ordinary words**, taking the room with them:

| keyword | caught inside | what the model did |
| --- | --- | --- |
| BAL | BALLROOM, BALANCE SPACE | removed the roof |
| SPA | SPARE ROOM, SPARE BEDROOM, SPACE | tiled it as a wet area |
| DOME | DOMESTIC HELP, DOMESTIC STORE | built a dome over it |
| DUCT | PRODUCTION ROOM | cut a hole through the floor |
| OTB | FOOTBALL COURT | cut a hole through the floor |
| STEP | STEPNEY STORE | built a staircase in a tyre store |

All six are whole-word marks now, and still read what they are for: `BAL 2`,
`SPA ROOM`, `DUCT`, `STEPS`. `POOL TABLE ROOM` was built as water and left
open to the sky; `POOL` has to stay a substring for `SWIMMING POOL` and
`POOL DECK`, so the games room is named instead.

**Six keywords could never match anything.** Labels are normalised before
matching -- punctuation becomes a space -- and the keywords were not, so
`BUTLER'S PANTRY`, `DOUBLE-HEIGHT`, `TRIPLE-HEIGHT`, `OWNER'S SUITE`,
`CHEF'S KITCHEN` and `BUTLER'S KITCHEN` were dead entries. The keywords are
normalised as the lookup tables are built.

**Two groups were filed under the wrong meaning.** `GREEN WALL`, `VERTICAL
GARDEN` and `LIVING WALL` were ground cover, so planting on a wall was laid
across the floor and the roof removed above it. `LOGGIA`, `ARCADE`,
`COLONNADE`, `CLOISTER`, `SCREENED PORCH`, `GAZEBO`, `PAVILION` and `CABANA`
were open to the sky, though each is a roofed structure by definition; they
are canopies now. `LANAI`, `BREEZEWAY`, `PERGOLA`, `TRELLIS` and `ARBOR`
stay open, being used for unroofed spaces as often as roofed ones.

**Tried and reverted:** `OPEN TO SKY` reads as a void rather than as open.
The name argues otherwise, but on Indian drawings OTS marks a light well --
a shaft through the floors, which is what a void builds. Moved to open it
would keep every floor and remove the roof, turning the shaft into a terrace
on each storey.

**Judgement calls left alone:** a verandah, a porch and a sit-out are paved
and open here, which is a choice rather than an error -- on the reference
house the rooms opened to the sky are exactly the right ones (parking,
landscape, both balconies, the planted deck). An indoor jacuzzi or hot tub
still reads as water.

Spanish `BALCON` used to match through the `BAL` substring, which a test
already recorded as an accident rather than support. It now reads as
nothing, like the eight other unsupported Spanish and Portuguese terms.

## Fixed earlier the same day, from the accuracy work

Recorded in full in `docs/AUDIT.md`:

- **The smallest-room filter was sized by an estimate**, so a plan whose
  scale read 27% high lost real rooms (39 → 40 plans right on every check).
- **Gutters narrower than 0.6% of the sheet were invisible**, so sheets
  carrying two plans reconstructed as one storey (40 → 41).
- **Walls were read far thicker than the drawing draws them**: on plan 6094,
  whose walls gauge 20 px, runs of 68 and 54 px carried 43% of all painted
  wall (41 → 42).

## Known limitations, verified and not bugs

- **Diagonal plans reconstruct poorly.** BRIDGE `116412.gif` is drawn at 45
  degrees; the extractor recovers axis-aligned walls only, so it yields 3
  rooms. This is a deliberate trade recorded in `docs/AUDIT.md` ("Diagonal
  walls are not worth recovering"): diagonal wall costs 2.7% of wall pixels
  across the corpus, and no plan over 10%.
- **An 80 px thumbnail still builds something.** It produces 20 walls at a
  scale of 4.0 px/ft. The enlargement path lifts thin walls back above the
  geometry's floors, which is what lets small scans work at all; the result
  is poor but not wrong, and refusing it would refuse legitimate small
  scans.
- **Demo plans share numbers with corpus sheets.** `demo_plans/1` and BRIDGE
  `114012.gif` report identical geometry because the demo plans were copied
  from those corpora.
- **Buildings are not twice as tall as they should be.** The reference house
  measured 19.0 m and the two-storey demo 11.0 m on what looked like the
  vertical axis, roughly double a 9-foot storey. glTF is Y-up: those were
  depths. Read correctly the house is 34 ft over three storeys, matching the
  34.6 ft to coping recorded earlier, and the demo 20.7 ft over two.
- **`--landscaping premium` leaving bare ground.** It switches on ground,
  planting and a boundary, and planting is drawn where the drawing shows
  planting. `demo_plans/3` is a line drawing with none, so bare ground with
  a boundary wall is the honest result.
- **Storeys stack where they should.** On the house PDF the three floors
  centre within 2 px of each other (1136, 1135, 1137).
- **Every feature category builds something.** All 15 were driven through
  `floors_to_parts` one at a time against an otherwise identical storey:
  each produces geometry the plain case does not -- dome 640 faces, stairs
  180, tower 20, tank 24, and so on down to wet and lawn at 12. None of
  them is a word in the vocabulary with nothing behind it.
- **A ramp is a slope, not steps.** It books its geometry under the stairs
  material, which looked wrong; `_ramp` builds a slab falling at 1:12, the
  usual accessible gradient.

## Rough edges, not fixed

- **`--scale-room` runs the pipeline twice** -- once to measure the room
  being quoted, once to rebuild with the stated scale -- and prints its
  summary twice. It is the correct answer arrived at wastefully, on a path
  used once per drawing.
