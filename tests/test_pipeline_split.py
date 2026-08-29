"""planto3d/pipeline.py's run() must behave identically after the split
into extract() + build() -- this is a pure refactor, not a behavior
change. Most tests here use a synthetic FloorPlan built directly in code,
the same way tests/test_open_to_sky.py does, so they need no image or OCR
and run fast; one test (below) drives run() and extract()+build() from an
actual synthetic image, the same pattern tests/test_image_input.py uses,
to confirm the equivalence for real rather than by construction.
"""

import cv2
import numpy as np
from pathlib import Path

from planto3d.classical import ROOM_FILL, WALL_FILL
from planto3d.geometry_types import FloorPlan, Room, Wall
from planto3d.pipeline import FloorResult, PipelineResult, build, extract, run


def _make_result() -> PipelineResult:
    outline = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    walls = [
        Wall(start=outline[i], end=outline[(i + 1) % 4], thickness=6.0)
        for i in range(4)
    ]
    room = Room(polygon=outline, label="BEDROOM")
    plan = FloorPlan(walls=walls, rooms=[room], openings=[], footprint=outline)
    floor = FloorResult(index=0, image_path=Path("nonexistent.png"), plan=plan)
    return PipelineResult(
        floors=[floor],
        scale=20.0,
        model_path=None,
        scale_source="dimensions",
        page_size=(400, 300),
    )


def test_build_produces_a_model_from_an_already_extracted_result(tmp_path):
    result = _make_result()
    built = build(result, tmp_path)
    assert built.model_path is not None
    assert built.model_path.is_file()
    assert built is result  # mutated in place, matching the file's existing style


def test_build_preserves_extracted_geometry():
    result = _make_result()
    original_room_count = result.room_count
    original_wall_count = result.wall_count
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as workdir:
        built = build(result, Path(workdir))
    assert built.room_count == original_room_count
    assert built.wall_count == original_wall_count


def _plan_image(size=320):
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[40:280, 40:280] = WALL_FILL
    image[52:268, 52:268] = ROOM_FILL
    return image


def test_run_is_equivalent_to_extract_then_build(tmp_path):
    # run()'s own docstring says "See extract + build"; this drives both
    # paths from the same synthetic plan and compares every field a UI or
    # a later pipeline stage would read, so that claim is verified rather
    # than merely true by construction (run() is in fact just `build(
    # extract(...))`, but nothing stops that drifting apart later).
    image = _plan_image()
    source_a = tmp_path / "a" / "plan.png"
    source_a.parent.mkdir()
    cv2.imwrite(str(source_a), image)
    source_b = tmp_path / "b" / "plan.png"
    source_b.parent.mkdir()
    cv2.imwrite(str(source_b), image)

    via_run = run(source_a, tmp_path / "out_run", crop=False)
    extracted = extract(source_b, tmp_path / "out_split", crop=False)
    via_split = build(extracted, tmp_path / "out_split")

    assert via_run.wall_count == via_split.wall_count
    assert via_run.room_count == via_split.room_count
    assert via_run.opening_count == via_split.opening_count
    assert via_run.scale == via_split.scale
    assert via_run.scale_source == via_split.scale_source
    assert via_run.model_path is not None and via_run.model_path.is_file()
    assert via_split.model_path is not None and via_split.model_path.is_file()
