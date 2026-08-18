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
| Materials | Sixteen surfaces, including per-room floor finishes |
| Views | Top, front, back, left, right, aerial |
| Web app | Gradio, multi-file upload, live 3D viewer |
| Photoreal guides | Depth, edge and shaded renders ready |

384 tests. Around 4,500 lines.

## The one thing not yet done

**Run the photoreal pass.** [`notebooks/photoreal_on_colab.ipynb`](../notebooks/photoreal_on_colab.ipynb)
is ready and needs a GPU, so it runs on Colab rather than locally.

Regenerate the guides first, since the model has changed since they were
last built:

```bash
python scripts/run_pipeline.py "data/soni_residence/DOC-20260817-WA0027.PDF" output_unet --checkpoint models/unet_cubicasa.pt
python -c "from planto3d.photoreal import build_guides; build_guides('output_unet/house.glb','output_unet/guides')"
```

Then upload `output_unet/guides/guide-depth.png` to the notebook. Section 5
sweeps the conditioning strength so the trade-off between holding the
geometry and enriching the image can be seen rather than guessed.

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
