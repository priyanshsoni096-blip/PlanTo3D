"""Tests for rebuilding the held-out benchmark from the CubiCasa archive.

The 60 test plans every result is quoted against once lived only in a temp
folder, and were lost with it. The list is now committed; this is what turns
the list back into files. What is pinned here: the right files land in the
right place, nothing already there is overwritten, a plan the archive lacks
stops the run, and the draw that produced the list is deterministic.
"""

import zipfile

import pytest


def _archive(tmp_path, entries):
    path = tmp_path / "cubicasa5k.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for entry in entries:
            archive.writestr(f"cubicasa5k/{entry}/model.svg", f"<svg>{entry}</svg>")
            archive.writestr(f"cubicasa5k/{entry}/F1_scaled.png", f"png {entry}")
    return path


def test_the_list_reads_as_category_and_plan(tmp_path):
    from scripts.rebuild_benchmark import read_list

    listing = tmp_path / "list.txt"
    listing.write_text("/colorful/9448/\n\n/high_quality/10004/\n")
    assert read_list(listing) == ["colorful/9448", "high_quality/10004"]


def test_extraction_writes_both_files_for_every_plan(tmp_path):
    from scripts.rebuild_benchmark import extract

    archive = _archive(tmp_path, ["colorful/9448", "high_quality/10004"])
    out = tmp_path / "data"

    written, kept = extract(archive, ["colorful/9448", "high_quality/10004"], out)

    assert (written, kept) == (4, 0)
    assert (out / "colorful/9448/model.svg").read_text() == "<svg>colorful/9448</svg>"
    assert (out / "high_quality/10004/F1_scaled.png").read_bytes() == b"png high_quality/10004"


def test_a_file_already_there_is_left_alone(tmp_path):
    # A re-run must not undo anything a person put beside the plans, such as
    # the window label files the correction tooling writes.
    from scripts.rebuild_benchmark import extract

    archive = _archive(tmp_path, ["colorful/9448"])
    out = tmp_path / "data"
    (out / "colorful/9448").mkdir(parents=True)
    (out / "colorful/9448/model.svg").write_text("kept")

    written, kept = extract(archive, ["colorful/9448"], out)

    assert (written, kept) == (1, 1)
    assert (out / "colorful/9448/model.svg").read_text() == "kept"


def test_a_plan_missing_from_the_archive_stops_the_run(tmp_path):
    # A benchmark rebuilt with a plan quietly missing would score 59 plans
    # and be reported as 60.
    from scripts.rebuild_benchmark import extract

    archive = _archive(tmp_path, ["colorful/9448"])
    with pytest.raises(ValueError, match="colorful/99999"):
        extract(archive, ["colorful/9448", "colorful/99999"], tmp_path / "data")


def test_an_entry_cannot_escape_the_output_folder(tmp_path):
    from scripts.rebuild_benchmark import extract

    archive = _archive(tmp_path, ["colorful/9448"])
    with pytest.raises(ValueError, match="outside"):
        extract(archive, ["../escape"], tmp_path / "data")
    assert not (tmp_path / "escape").exists()


def test_the_draw_is_deterministic_and_stays_inside_the_split():
    from scripts.rebuild_benchmark import draw

    split = [f"{category}/{number}" for category in ("colorful", "high_quality")
             for number in range(100, 140)]

    first = draw(split, per_category=5, seed=20260913)

    assert first == draw(split, per_category=5, seed=20260913)
    assert first != draw(split, per_category=5, seed=1)
    assert len(first) == 10 and set(first) <= set(split)
    for category in ("colorful", "high_quality"):
        chosen = [entry for entry in first if entry.startswith(category + "/")]
        assert len(chosen) == 5
        assert chosen == sorted(chosen, key=lambda entry: int(entry.split("/")[1]))


def test_the_command_defaults_to_the_committed_benchmark():
    from scripts.rebuild_benchmark import build_parser

    arguments = build_parser().parse_args(["cubicasa5k.zip"])
    assert arguments.list.as_posix() == "data/cubicasa_test60.txt"
    assert arguments.out.as_posix() == "data/cubicasa5k"
