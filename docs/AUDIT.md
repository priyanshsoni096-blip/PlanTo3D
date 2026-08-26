# Where every part of the project stands

An honest stage-by-stage audit. Every figure here is measured, and the
source of each measurement is named so it can be re-run and disagreed
with.

**What it is measured on.** 60 CubiCasa plans (Finnish apartments), one
Indian villa used only as a control, and one web-sized sheet carrying two
plans. That is **two and a half drafting conventions**, and it is the
single biggest qualification on everything below.

```bash
python scripts/batch_evaluate.py <cubicasa> --checkpoint models/unet_cubicasa.pt --limit 60
python scripts/scale_accuracy.py <cubicasa> --checkpoint models/unet_cubicasa.pt
python scripts/split_accuracy.py <cubicasa>
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
| Renderer | Complete | Linear light, filmic curve, antialiasing, ambient occlusion |
| Photoreal pass | Works | Runs on a T4, conditioned on the model's depth |
| Notebooks | Complete | `train_on_colab`, `run_on_colab` |
| Tests | 682 passing | — |

## Gaps, worst first

| # | Gap | Measured | Why it matters | General? |
| --- | --- | --- | --- | --- |
| 1 | **Windows barely detected** | IoU **0.096**, finds **37%** | Every façade is sparser than the drawing. Next-worst class is 5× better | **Yes** |
| 2 | **Scale, 17.7% median error** | Walls −19.7% bias on 20 plans | Sets the whole building's size | Partly — see below |
| 3 | ~~Sheet splitting misses~~ **closed** | Recall **79%**, 55/60 exact | A missed split reconstructs several plans as one flat building, confidently | **Yes** |
| 4 | **Only 2½ conventions tested** | — | The generality claim rests on Finnish apartments | **Yes** |
| 5 | Storage rooms weak | IoU 0.525 | Storage reads as ordinary rooms | Yes |
| 6 | Bath rooms weak | IoU 0.586 | Wet floors missed | Yes |
| 7 | ~~Prompt truncated~~ **closed** | 68 tokens, site features preserved | — | — |
| 8 | OCR reads few names | 12/60 plans | Largely mitigated: types come from the model now | Inherent |
| 9 | ~~Diagonal walls~~ **not worth building** | Costs **2.7%** of wall pixels, no plan over 10% | See below | — |
| 10 | Classical baseline | No walls on 10/12 unseen | Scaffold for clean CAD only; the trained model is the contribution | Documented |
| 11 | Colour heuristics | Windows 5/12, planting 1/12 | One drafting office's convention | Documented |

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

Never checked until now, and everything downstream rests on it. Measured
by painting the built walls back at their own thickness and comparing
with the annotation, over 30 plans:

| | Median | Plans below 70% |
| --- | --- | --- |
| **Coverage** — annotated wall that gets built | **99.0%** | **0 of 30** |
| **Agreement** — built wall that really is wall | 88.4% | 4 of 30 |

Essentially every wall in the drawing is found. What remains is
over-building on a handful of plans, and two hypotheses for it were
tested and rejected:

- **Not envelope closing.** Sample 10711 has *zero* invented walls and
  still sits at 62% agreement, so the disagreement is in the walls read
  off the drawing.
- **Not thickness.** Extracted walls come out *thinner* than annotated
  ones, at a median ratio of 0.79, so a wall painted at its own thickness
  lands inside the real one rather than spilling outside it.

That leaves position, on four plans of thirty. Small, and not chased.

Worth recording alongside it: the predicted wall gauge runs about 10–15%
thicker than the annotated one, while the extracted segments run 21%
thinner. The orientation opening erodes what the segmenter over-predicts.

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
