"""Render a built model with Blender, deterministically.

The pipeline's own rasterizer (planto3d/preview.py) is fast and plain.
This is the slow, pretty one: real materials, soft shadows and global
illumination, with no generative step -- so unlike the diffusion pass,
every surface in the image is one the drawing supports.

Needs the render extra:  pip install -e ".[render]"   (659 MB)

    python scripts/render_blender.py house.glb output
"""

import argparse
import logging
from pathlib import Path

from planto3d import blender_render


def main(model: str, output_dir: str, resolution: int, samples: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not blender_render.available():
        raise SystemExit(
            'Blender is not installed. Run: pip install -e ".[render]"  '
            "(659 MB), or use scripts/run_pipeline.py for the fast renderer."
        )

    height = int(resolution * 3 / 4)
    rendered = blender_render.render_views(
        Path(model), Path(output_dir),
        resolution=(resolution, height), samples=samples,
    )
    for name, path in sorted(rendered.items()):
        print(f"{name:8} {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="a .glb written by the pipeline")
    parser.add_argument("output_dir")
    parser.add_argument("--resolution", type=int, default=1200, help="width in pixels")
    parser.add_argument(
        "--samples", type=int, default=blender_render.DEFAULT_SAMPLES,
        help="Cycles samples; 32 is the measured floor, higher costs little",
    )
    arguments = parser.parse_args()
    main(arguments.model, arguments.output_dir, arguments.resolution, arguments.samples)
