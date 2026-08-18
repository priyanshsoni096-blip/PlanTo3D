"""Build a 3D model from a floor plan.

Usage:
    python scripts/run_pipeline.py <source> <output_dir> [--checkpoint model.pt]

``source`` may be a PDF, a single image, or a directory of images -- one per
storey, in filename order.

Without a checkpoint the classical baseline is used, which is tuned to clean
CAD sheets. Pass a trained checkpoint to segment plans in other styles.
"""

import argparse
import logging
from pathlib import Path

import cv2

from planto3d.pipeline import draw_overlay, run
from planto3d.preview import render_views
from planto3d.segment import load_segmenter


def main(
    source: str,
    output_dir: str,
    checkpoint: Path | None = None,
    crop: bool = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(output_dir)

    result = run(Path(source), out, segmenter=load_segmenter(checkpoint), crop=crop)

    print("\n" + "=" * 58)
    for floor in result.floors:
        print(
            f"floor {floor.index}: {len(floor.plan.walls):3d} walls  "
            f"{len(floor.plan.rooms):3d} rooms  {len(floor.named_rooms):2d} named"
        )
        print(f"   {floor.named_rooms}")
        cv2.imwrite(str(out / f"overlay-{floor.index}.png"), draw_overlay(floor))

    scale = f"{result.scale:.2f} px/ft" if result.scale else "unknown"
    print("-" * 58)
    print(f"totals: {result.wall_count} walls, {result.room_count} rooms, scale {scale}")
    print(f"model:  {result.model_path}")

    if result.model_path is not None:
        views = render_views(result.model_path, out)
        print(f"views:  {', '.join(sorted(views))}")

    print(f"output written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="PDF, image, or directory of images")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="trained segmenter; omit to use the classical baseline",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="skip title-block cropping, for images that are already just the plan",
    )
    arguments = parser.parse_args()
    main(
        arguments.source,
        arguments.output_dir,
        arguments.checkpoint,
        crop=not arguments.no_crop,
    )
