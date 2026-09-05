"""Run the whole project once, on one plan, and show what it produced.

    python scripts/demo.py

That is the whole command. It picks a demo plan, reads it, builds the 3D
model, renders it, and prints where everything landed.

Every other script in this directory does one stage or measures one thing.
This one exists to be run once by somebody who wants to see the project
work rather than to measure it.

    python scripts/demo.py --plan demo_plans/2-TWO-STOREY-stairs.gif
    python scripts/demo.py --lighting dusk
    python scripts/demo.py --fast          # skip Blender: ~69s, not ~270s
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from planto3d import blender_render
from planto3d.pipeline import draw_overlay, run
from planto3d.preview import render_views as render_fast
from planto3d.segment import load_segmenter

# The plan the pipeline handles best: a printed scale it can actually read,
# 50 walls and 19 rooms, so the model has enough in it to be worth looking
# at. Chosen because a demo that opens on the weakest case teaches the
# wrong thing about the project; the scorecard scripts cover the hard ones.
DEFAULT_PLAN = Path("demo_plans/1-BEST-measured-scale-50walls-19rooms.gif")
DEFAULT_CHECKPOINT = Path("models/unet_cubicasa.pt")

# Wide enough to see the building, small enough to keep the demo bearable.
# Measured on this machine, this plan, at 900px: the six Blender views took
# 198.8s against 68.5s for everything else, so the pass is roughly three
# quarters of the run. scripts/render_blender.py defaults to 1200 for a real
# render; six views multiply that by six, which is why the demo steps down.
# --fast skips the pass entirely.
DEMO_WIDTH = 900


def stage(number: int, total: int, title: str) -> float:
    print(f"\n[{number}/{total}] {title}")
    print("-" * 60)
    return time.monotonic()


def done(started: float) -> None:
    print(f"      ...{time.monotonic() - started:.1f}s")


def main(
    plan: Path,
    output_dir: Path,
    checkpoint: Path | None,
    lighting: str,
    fast: bool,
) -> None:
    # The pipeline logs at INFO, which is most of what there is to watch
    # during the slow stages. Left on deliberately: a demo that prints
    # nothing for ninety seconds looks broken.
    #
    # Onto stdout rather than logging's default stderr, and unbuffered,
    # so the log lines land between the stage headers instead of ahead of
    # them. Two streams into one terminal interleave by flush order, not
    # by the order they were written -- the first run of this script
    # printed the whole pipeline log above the banner.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("      %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Blender's glTF importer attaches its own stdout handler *and* lets
    # records propagate to the root, so every line it emits printed twice
    # -- once stamped by its formatter, once by ours. Fourteen mesh nodes
    # times six views is 168 duplicated lines burying the pipeline's own
    # output. Silencing it here rather than in blender_render keeps the
    # library importable without a logging opinion.
    logging.getLogger("io_scene_gltf2").setLevel(logging.WARNING)

    if not plan.exists():
        raise SystemExit(
            f"No plan at {plan}.\n"
            "demo_plans/ is not tracked in git -- pass --plan with a floor "
            "plan of your own (PDF, PNG, JPG or GIF)."
        )

    use_blender = not fast and blender_render.available()
    total = 4 if use_blender else 3

    print("=" * 60)
    print(f"PlanTo3D -- {plan.name}")
    print("=" * 60)

    started = stage(1, total, "Reading the drawing")
    if checkpoint is not None and not checkpoint.exists():
        print(f"      no checkpoint at {checkpoint}, using the classical baseline")
        checkpoint = None
    result = run(plan, output_dir, segmenter=load_segmenter(checkpoint))
    done(started)

    started = stage(2, total, "What it found")
    for floor in result.floors:
        print(
            f"      floor {floor.index}: {len(floor.plan.walls):3d} walls  "
            f"{len(floor.plan.rooms):3d} rooms  "
            f"{len(floor.named_rooms):2d} named"
        )
        if floor.named_rooms:
            print(f"        {', '.join(sorted(set(floor.named_rooms)))}")
        cv2.imwrite(str(output_dir / f"overlay-{floor.index}.png"), draw_overlay(floor))

    # The scale is the number worth reading twice: it is what makes the
    # model measurable in feet rather than merely proportioned, and it is
    # the project's largest single source of error. Where it came from
    # matters as much as its value -- a printed dimension is evidence,
    # a door-width assumption is an inference.
    if result.scale:
        print(f"\n      scale: {result.scale:.2f} px/ft  (from {result.scale_source})")
    else:
        print("\n      scale: could not be determined -- the model is unitless")
    print(f"      model: {result.model_path}")
    done(started)

    if result.model_path is None:
        raise SystemExit("\nNo model was built, so there is nothing to render.")

    started = stage(3, total, "Rendering (fast rasterizer)")
    views = render_fast(result.model_path, output_dir)
    print(f"      {len(views)} views: {', '.join(sorted(views))}")
    done(started)

    if use_blender:
        started = stage(4, total, f"Rendering (Blender Cycles, {lighting})")
        print("      six views, path-traced -- this is the slow one\n")
        from planto3d.style import LIGHTING_PRESETS

        rendered = blender_render.render_views(
            result.model_path,
            output_dir / "blender",
            resolution=(DEMO_WIDTH, DEMO_WIDTH * 3 // 4),
            lighting=LIGHTING_PRESETS[lighting],
        )
        for name, path in sorted(rendered.items()):
            print(f"      {name:8} {path}")
        done(started)

    print("\n" + "=" * 60)
    print(f"Everything is in {output_dir.resolve()}")
    print("=" * 60)
    print(f"  overlay-*.png    what the segmenter saw, drawn over the plan")
    print(f"  {result.model_path.name:16} the 3D model -- open in any glTF viewer")
    print(f"  *.png            six views from the fast rasterizer")
    if use_blender:
        print(f"  blender/         the same six views, path-traced")
    else:
        reason = "--fast" if fast else 'not installed; pip install -e ".[render]"'
        print(f"\n  (Blender render skipped: {reason})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="trained segmenter; falls back to the classical baseline if absent",
    )
    # Same names style.LIGHTING_PRESETS uses, as scripts/render_blender.py
    # does -- one vocabulary for the hour of day across the project.
    from planto3d.style import LIGHTING_PRESETS

    parser.add_argument(
        "--lighting", choices=sorted(LIGHTING_PRESETS), default="midday"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the Blender pass",
    )
    arguments = parser.parse_args()
    main(
        arguments.plan,
        arguments.output,
        arguments.checkpoint,
        arguments.lighting,
        arguments.fast,
    )
