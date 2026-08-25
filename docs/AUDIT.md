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
| 9 | Diagonal walls not recovered | — | Documented limitation, erased by the orientation filters | Yes |
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

## Why windows are the next step

| Class | Native | At 512 input | At 768 input |
| --- | --- | --- | --- |
| Wall | 22 px | 8.4 px | 12.5 px |
| Door | 16 px | 6.5 px | 9.8 px |
| **Window** | **4 px** | **1.5 px** | **2.2 px** |

A window is **one and a half pixels** by the time the network sees it. It
is not that the model is bad at windows; it is that windows are barely
present in its input. The class weight is already the highest in the loss
at 3.72, and weighting cannot recover information that was resampled away.

Training at 768 raises every thin class by half again. It costs roughly
double the GPU time per epoch and helps every plan rather than one
population.

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
