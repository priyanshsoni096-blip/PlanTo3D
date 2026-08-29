"""planto3d/pipeline.py's run() must behave identically after the split
into extract() + build() -- this is a pure refactor, not a behavior
change. Uses a synthetic FloorPlan built directly in code, the same way
tests/test_open_to_sky.py does, so this test needs no image or OCR and
runs fast.
"""

from pathlib import Path

from planto3d.geometry_types import FloorPlan, Room, Wall
from planto3d.pipeline import FloorResult, PipelineResult, build, extract


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
