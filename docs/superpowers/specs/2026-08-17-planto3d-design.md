# PlanTo3D — Design Spec

## Problem

Converting a 2D architectural floor plan into a 3D house model is normally
manual: a designer traces walls, sets heights, and places doors/windows by
hand in SketchUp or Revit. PlanTo3D automates this — given a 2D floor plan
(image or PDF), it outputs a geometrically accurate, viewable, and
photoreal-enhanced 3D model, with no manual tracing.

Reference test case: the "Soni Residence" plan set (RDA architects) — three
floors (ground ~3,050 sq ft, first ~3,940 sq ft, terrace ~1,390 sq ft), each
with printed room labels and dimensions, clean CAD-drafted line work.

## Scope

- Input: clean, digital-quality architectural floor plans (CAD-drafted,
  like the reference set), not hand-drawn scans. That's a harder, separate
  problem and explicitly out of scope for v1.
- Multi-floor: v1 handles all three floors of the reference set, stacked
  into one building.
- Output: a geometrically accurate 3D mesh, enhanced toward photorealism via
  an automated materials/lighting/AI-stylization pass (not a guaranteed
  pixel-match to hand-art-directed renders).

## Architecture

Linear pipeline, each stage consumes the previous stage's output:

```
PDF (3 pages)
  → rasterize
  → [1: Segmentation]        U-Net + ResNet encoder → per-pixel class mask
  → [2: Geometry Extraction] OpenCV contours + minAreaRect → wall/room/opening vectors
  → [3: Scale Calibration]   OCR dimension text → pixels-to-feet factor
  → [4: Floor Alignment]     shared stair/lift core → common coordinate origin
  → [5: 3D Extrusion]        trimesh/Open3D → walls, openings, slabs, roof (.glb)
  → [6: Viewer]              Three.js → rotatable in-browser model
  → [7: Materials/Lighting]  Blender bpy → auto materials, lights, landscaping, cars
  → [8: Photoreal Pass]      ControlNet-guided diffusion → stylized final render
```

## Components

1. **Ingestion** — rasterize each PDF page to PNG at fixed DPI; crop out the
   fixed RDA title-block/legend region (shared across all 3 sheets).

2. **Segmentation** — U-Net with a ResNet34 encoder (`segmentation-models-
   pytorch`), encoder pretrained on ImageNet, trained on CubiCasa5K.
   Evaluated on CubiCasa5K's own test split via Dice/IoU. Outputs a
   per-pixel class mask: wall / door / window / room / background. The model
   classifies boundaries only — it does not assign room *type* (see Geometry
   Extraction).

   *Why not the official CubiCasa weights:* the published checkpoint
   (`model_best_val_loss_var.pkl`) is for `hg_furukawa_original`, a stacked
   hourglass network — the weights cannot load into a U-Net. The reference
   implementation also targets Python 3.6.5 / PyTorch 1.0.0 / OpenCV 3.1.0,
   so adopting it would mean porting a 2019 codebase before training
   anything. Training our own keeps a modern stack and keeps the measurable
   training/evaluation contribution in the project.

   *Training environment:* Colab (free T4). No local GPU is available —
   the development machine has Intel integrated graphics and no CUDA — but
   only this stage needs one. Inference on a single floor plan runs in
   seconds on CPU, so every other stage stays local.

   *Dataset source:* the Kaggle mirror (`qmarva/cubicasa5k`, ~6 GB), which
   splits into `colorful` (276), `high_quality` (992), and
   `high_quality_architectural` (3,732) samples. Training weights toward
   `high_quality_architectural`, whose CAD-style line work most resembles
   the reference sheets. Annotations are SVG polygons over 80+ categories
   and must be collapsed to the five classes above. The dataset downloads
   inside Colab rather than travelling from the local machine.

3. **Geometry extraction** — OpenCV contour detection + `cv2.minAreaRect`
   converts the mask into: wall line segments (start, end, thickness),
   closed room polygons, and door/window openings tied to their host wall.
   Room labels come from OCR text (see next stage), not from the
   segmentation model — CubiCasa5K's Finnish taxonomy has no categories for
   "Temple," "Verandah," or "Dress/Toilet," which appear in the reference
   plans.

4. **Scale calibration** — Tesseract OCR reads printed room dimension text
   (e.g. `15'0"X18'0"`) and room name labels. Dimension text derives a
   pixels→feet scale factor per page; name labels become the room's stored
   label. All geometry is converted to real-world feet, then to meters for
   the mesh stage.

5. **Floor alignment** — detects the shared lift/staircase core present on
   all three floor plans and uses it as a common anchor to register
   ground/first/terrace floors to one coordinate origin before stacking.
   Falls back to a manual anchor point (explicitly specified per floor) if
   automatic detection fails or is low-confidence — this is not allowed to
   be a silent single point of failure for the whole pipeline.

6. **3D extrusion** — trimesh/Open3D: extrude walls to a standard height
   per floor, cut door/window openings at their detected positions, add
   floor slabs and ceiling planes, and generate a flat roof/parapet from
   the terrace floor's outer wall polygon. Outputs `.glb`, meters, Y-up.

7. **Viewer** — Three.js web page: upload a floor plan, see the 3D model
   appear and rotate in-browser. No install required.

8. **Materials & lighting (Blender, core not stretch)** — a `bpy` script
   auto-assigns materials by OCR room label (kitchen → tile, bedroom →
   wood/carpet, detected windows → glass, walls → plaster), places lights
   at every detected window plus one per room centroid, places greenery in
   OCR-labeled landscape/garden areas, places car placeholders in the
   detected parking area, and positions a camera at an aerial 3/4 angle.
   Rendered with Blender Cycles.

9. **Photoreal enhancement pass** — the Blender render is fed as a
   structural guide (depth/edge map) into a ControlNet-guided diffusion
   pass with a descriptive prompt (materials, warm lighting, landscaping),
   producing a stylized final image closer to professional archviz output.
   This is an approximation, not a guaranteed match to any specific
   reference render.

## Data flow & formats

- Internal representation, per floor, JSON:
  ```json
  {
    "walls": [{"start": [x, y], "end": [x, y], "thickness": t}],
    "rooms": [{"polygon": [[x, y], ...], "label": "BEDROOM"}],
    "openings": [{"wall_id": id, "position": p, "width": w, "type": "door|window"}]
  }
  ```
  Coordinates in feet until the mesh stage, then converted to meters.
- Final mesh: `.glb`, meters, Y-up (Three.js convention).

## Error handling

- OCR dimension read fails, or a room polygon doesn't close → flag and skip
  that room with a logged warning; the pipeline continues rather than
  crashing.
- Floor-alignment anchor detection is low-confidence → fall back to a
  manually specified anchor point rather than producing silently misaligned
  floors.
- No ground-truth mask exists for the Soni Residence itself, so its
  validation is the 2D visual-overlay check (redraw extracted geometry over
  the original), not Dice/IoU — that metric applies only to the CubiCasa5K
  validation split.

## Testing

- Unit tests for geometry extraction against a synthetic hand-made floor
  plan (one rectangle room, one door) before running on real segmentation
  output — isolates CV bugs from model bugs.
- Segmentation model evaluated on CubiCasa5K's own test split (Dice/IoU).
- End-to-end validation on the real Soni Residence PDF, in order: 2D overlay
  visual check → rough 3D block preview (no materials) → full pipeline
  through the photoreal pass.

## Build strategy

Build a thin end-to-end skeleton first — rasterize → segment → extract →
crude extrude, no polish — and run it against the real Soni Residence pages
before investing in the viewer or Blender/photoreal stages. This surfaces
the riskiest unknowns (model domain-gap on this drawing style, alignment
heuristic reliability, OCR accuracy) early, while they're still cheap to
fix.

## Tech stack

Python, PyTorch + `segmentation-models-pytorch` (segmentation, trained on
Colab), OpenCV + Tesseract (geometry/OCR), trimesh/Open3D (mesh), Three.js
(viewer), Blender `bpy` (materials/render), Stable Diffusion + ControlNet
(photoreal pass). Everything except training runs locally on CPU.

## Project structure

```
PlanTo3D/
  data/            # CubiCasa5K + Soni Residence reference pages
  segmentation/    # U-Net training/eval
  geometry/        # extraction, calibration, alignment
  mesh/            # extrusion, roof, export
  viewer/          # Three.js app
  render/          # Blender + photoreal pass scripts
  tests/
```

## Known risks (accepted, not blocking)

- **Domain gap**: CubiCasa5K's Finnish-style plans differ from this RDA CAD
  sheet style; the trained model may need calibration specifically on plans
  in this style rather than relying on zero-shot transfer. Partly mitigated
  by weighting training toward the `high_quality_architectural` subset,
  whose line work is closest to the reference sheets.
- **Annotation remapping**: CubiCasa5K's SVG annotations cover 80+
  categories and must be collapsed to five classes. Getting that mapping
  wrong (for example, which categories count as "wall") degrades the model
  in ways that look like a training problem rather than a labelling one.
- **Alignment heuristic**: automatic lift/staircase anchor detection is
  itself a small CV problem that could fail; manual fallback exists but
  adds friction if triggered often.
- **Timeline**: this is realistically weeks of work, not days, for one
  person — the photoreal pass (stages 8–9) should not consume time from the
  core pipeline (stages 1–6) before that's proven working.
