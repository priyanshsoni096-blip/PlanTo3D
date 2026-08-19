"""Pick a plan at random and run the whole pipeline on it.

Reaching for the familiar drawing is how a pipeline ends up fitted to one
building. This picks one nobody chose, runs every stage, and scores the
scale against what CubiCasa recorded.

    python scripts/random_plan.py [seed]

The seed is printed so a run can be repeated. The first use of this found a
real bug: it drew a single-storey apartment that the sheet splitter was
cutting in two along an internal wall, calibrating the halves at 34% under
true.
"""

import logging
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

import cv2

from planto3d.cubicasa import ground_truth_scale
from planto3d.pipeline import draw_overlay, run
from planto3d.preview import render_views
from planto3d.segment import load_segmenter

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

ROOT = Path(os.environ.get("PLANTO3D_CUBICASA", "data/cubicasa5k"))
OUT = Path(os.environ.get("PLANTO3D_OUT", "output_random"))
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(10**6)
random.seed(seed)

plans = sorted(ROOT.glob("*/*/F1_scaled.png"))
chosen = random.choice(plans)
print(f"seed {seed} -> {chosen.parent.parent.name}/{chosen.parent.name}")

image = cv2.imread(str(chosen))
print(f"sheet {image.shape[1]}x{image.shape[0]}\n")

segmenter = load_segmenter("models/unet_cubicasa.pt")
result = run(chosen, OUT, segmenter=segmenter, crop=False)

expected = ground_truth_scale(chosen.parent / "model.svg")
print(f"storeys found     {len(result.floors)}")
print(f"walls             {result.wall_count}")
print(f"rooms             {result.room_count}")
print(f"openings          {result.opening_count}")
print(f"scale             {result.scale:.1f} px/ft (from {result.scale_source})")
if expected:
    print(f"  true scale      {expected:.1f} px/ft  ({(result.scale - expected) / expected:+.0%})")

for floor in result.floors:
    names = ", ".join(floor.named_rooms) or "none read"
    types = sorted({room.category for room in floor.plan.rooms if room.category})
    print(
        f"  floor {floor.index + 1}: {len(floor.plan.walls)} walls, "
        f"{len(floor.plan.rooms)} rooms | names: {names} | types: {types or 'none'}"
    )

if result.model_path:
    views = render_views(result.model_path, OUT, resolution=(900, 660))
    print(f"\nrendered {len(views)} views")
    for floor in result.floors:
        path = OUT / f"detected-{floor.index}.png"
        cv2.imwrite(str(path), draw_overlay(floor))
    shutil.copy(chosen, OUT / "input.png")
    print(f"model: {result.model_path}")
else:
    print("\nno model produced")
