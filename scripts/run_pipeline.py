"""Build a 3D model from a floor plan PDF.

Usage:
    python scripts/run_pipeline.py <pdf_path> <output_dir>
"""

import logging
import sys
from pathlib import Path

import cv2

from planto3d.pipeline import draw_overlay, run


def main(pdf_path: str, output_dir: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(output_dir)

    result = run(Path(pdf_path), out)

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
    print(f"overlays written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
