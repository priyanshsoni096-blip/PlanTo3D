# PlanTo3D — Design System

## Product context

PlanTo3D turns a 2D architectural floor plan into a 3D model of the building.
You upload a drawing — PDF, PNG, JPEG, one file per storey or one multi-page
PDF — and it reads the walls, rooms, doors and windows straight off the sheet,
then builds and renders the house.

The product is an **instrument, not a toy**. Its single most important
behaviour is that it tells you what it knows and what it guessed. Where a plan
prints its room dimensions, the model is *measured* in real feet. Where it does
not, the scale is *inferred* from standard door widths or wall thickness, and
the interface says so in plain words. Every design decision below serves that:
the UI must never let a confident-looking render disguise an assumed number.

**Who it is for.** Architects, students, builders and homeowners who have a
plan on paper and want to see the building. They are not 3D artists. They will
not tune materials. They want the model, six views, and an honest answer about
its size.

**Jobs to be done**

1. "I have this floor plan — show me the building." (the whole job, in one click)
2. "Is this model actually the right size?" (the measured-vs-inferred question)
3. "Did it read my drawing correctly?" (the detection overlay)
4. "Make it look like the house I'm imagining." (the five appearance choices)
5. "Give me the file." (download the `.glb`)

## Key pages & architecture

Single-screen application. No navigation, no dashboard, no account wall — the
whole product is one workspace that moves through three states.

- **Empty state** — a large, inviting drop target for the plan, with the five
  appearance choices and the storey-height slider visible but secondary. The
  guidance about preferring the original PDF over a screenshot belongs here,
  where it can still change what the user uploads.
- **Working state** — the pipeline runs in stages (read the sheet → segment →
  extract geometry → extrude → render six views). Show the stage, because the
  run is long enough that a spinner alone reads as a hang.
- **Result state** — the model and its evidence. Rendered hero view; the
  interactive orbit viewer beside it; the scale verdict as a prominent badge;
  the per-floor table of walls, rooms and room names read; the six views; and
  the detection overlay showing what was found in the drawing.

Secondary surfaces, if the flow is extended later: a full-bleed viewer, and a
comparison of two appearance settings side by side.

## Key features to design for

- **Multi-file upload** with explicit storey ordering (ground floor first).
- **Storey height slider**, 7–14 ft in 0.5 ft steps. Plans do not state ceiling
  height, so this is a real input, not a preference.
- **Five appearance choices**, each a small set of named options, never a
  colour picker (an earlier version had per-surface colours and it was a
  spreadsheet, not a choice):
  - Style — modern, luxury, traditional, minimalist
  - Colour — light, dark, warm
  - Time of day — day, sunset, night
  - Landscaping — none, basic, premium
  - Creativity — strict, balanced, creative
- **Scale provenance badge** — the emotional centre of the interface. Four
  states, ordered by confidence: measured from printed dimensions; inferred
  from door widths; inferred from wall thickness; assumed from a drafting
  ratio. Anything below "measured" must visibly say the proportions are right
  but the absolute size is inferred.
- **Room-names warning** — when few or no room labels are read, finishes,
  planting, paving, railings and stairs are all missing, and the cause is
  nearly always input resolution. This warning must be actionable, not scolding.
- **Detection overlay** — the drawing with the found walls and rooms painted
  over it, per floor. This is how a user audits the machine.
- **Six rendered views** — top, front, back, left, right, aerial.
- **Interactive `.glb` viewer**, plus download.

## Branding & styling

### Voice

Precise, plain, unhedged. Short declarative sentences. Never "we think" or
"approximately-ish" — state the number and state where it came from. No
exclamation marks, no emoji, no marketing adjectives. This is the voice already
used throughout the product's own copy and it should not change.

### Colour

Dark-first. The product's output is renders and a 3D scene; a dark shell keeps
the eye on the image and matches the viewer's own near-black canvas.

Neutrals — cool slate, not pure grey:

- `--ink-950: #0B0D10` — app background (matches the 3D viewer's clear colour)
- `--ink-900: #111419` — panel background
- `--ink-800: #191D24` — raised surface / card
- `--ink-700: #262C35` — border, divider
- `--ink-500: #6B7683` — muted text, labels
- `--ink-200: #C3CAD3` — secondary text
- `--ink-50:  #F2F5F8` — primary text

Accent — blueprint cyan-blue. One accent only; it marks the primary action and
the active choice, nothing else:

- `--accent-500: #3B9EFF` — primary action, active state
- `--accent-600: #2179D6` — pressed
- `--accent-100: #D6E9FF` — accent text on dark fill

Semantic — these carry the honesty of the scale verdict and must be readable at
a glance, so they never double as decoration:

- `--measured: #3FD08A` — measured from printed dimensions (high confidence)
- `--inferred: #E3A13C` — inferred from doors or walls (proportions right, size inferred)
- `--assumed:  #E0703C` — assumed from a drafting ratio (nothing measurable found)
- `--danger:   #E4574E` — the run failed

Data colours for the detection overlay are fixed by the pipeline — walls in red,
rooms in green — and are painted onto the image, not into the chrome. Do not
introduce chrome elements in those two hues, or the legend stops meaning
anything.

Light mode uses the same ramp inverted: `#F7F9FB` app background, `#FFFFFF`
panels, `#0B0D10` text, identical accent and semantic values.

### Typography

- **UI:** Inter. Weights 400, 500, 600. Nothing heavier — bold headlines read
  as marketing here.
- **Numbers and measurements:** JetBrains Mono. Every quantity the product
  reports — pixels per foot, wall counts, room counts, feet — is set in mono.
  This is not decoration: it is how a reader tells a measured value from prose.
- Scale: 12 / 13 / 14 / 16 / 20 / 28 / 40 px. Body 14. Labels 12, uppercase,
  `0.06em` tracking, `--ink-500`.
- Line height 1.5 for prose, 1.2 for headings and numerals.

### Spacing & layout

- 4px base unit. Use 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64.
- Two-column workspace: controls left at a fixed `360px`, results right taking
  the remainder. Below `1024px` they stack, controls first.
- Max content width `1600px`; the render and viewer are allowed to grow with
  the window because they are the product.
- Panels: `--ink-900` fill, `1px solid --ink-700`, radius `10px`. Cards inside
  panels: `--ink-800`, radius `8px`.
- Gallery grids: 3 columns desktop, 2 tablet, 1 mobile, `12px` gap.

### Shape, border & shadow

- Radii: `6px` controls, `8px` cards, `10px` panels, `999px` pills and badges.
- Borders do the work that shadows do elsewhere — on a dark shell a 1px
  `--ink-700` border separates surfaces more cleanly than a glow.
- One shadow only, for genuinely floating things (dropdown, toast):
  `0 8px 24px rgba(0,0,0,0.45)`.

### Components

- **Buttons** — primary is `--accent-500` fill, `--ink-950` text, 600 weight,
  `40px` high. Secondary is transparent with `--ink-700` border. No gradients.
- **Radio groups** for the five appearance choices — rendered as segmented
  pills, not radio dots. Options are few and mutually exclusive; segments show
  the whole choice set at once, which is the point of offering named options.
- **Slider** — `--accent-500` track fill, mono numeral in the thumb label.
- **Drop zone** — dashed `2px --ink-700` border, `--accent-500` on drag-over,
  file chips listed beneath in upload order with their storey number.
- **Scale badge** — pill, semantic colour at 12% opacity as fill, full strength
  for the border and label, mono for the number. One line of plain prose
  underneath explaining the provenance.
- **Summary table** — floor, walls, rooms, names read. Mono for all counts,
  Inter for names. Zebra striping is unnecessary at four columns; a `--ink-700`
  rule between rows is enough.
- **Warnings** — left border `3px` in the semantic colour, `--ink-800` fill,
  never a full-bleed coloured banner. The room-names warning is common and must
  not shout.

## Motion

Restrained and functional. Nothing bounces.

- Transitions `160ms cubic-bezier(0.2, 0, 0.2, 1)` for hover, focus and
  selection.
- The working state advances through named stages with a determinate bar where
  progress is knowable and a slow indeterminate sweep where it is not.
- Results fade and rise `8px` over `240ms`, staggered `40ms` between the render,
  the viewer, the summary and the galleries — enough to read as arrival, not
  enough to delay.
- Respect `prefers-reduced-motion`: drop all transforms, keep opacity.

## Specific requirements

- **Never let the interface look more certain than the data.** If the scale was
  inferred, the badge says so at the same visual weight as the number itself.
- **The overlay is a first-class output, not a debug view.** It is how the user
  checks the machine's reading, and it gets the same presentation quality as
  the renders.
- Accessible contrast throughout: body text ≥ 4.5:1, large text and semantic
  badges ≥ 3:1, on both shells.
- Every control keyboard reachable, with a visible `2px --accent-500` focus
  ring offset `2px`.
- The empty state must teach the one thing that changes outcomes: upload the
  original PDF, not a screenshot.
