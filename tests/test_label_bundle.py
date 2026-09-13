"""Tests for carrying reviewed window labels from a laptop to a Colab run.

Corrections are made in LabelMe beside a local copy of the dataset, but
training on Colab downloads a fresh copy that has none of them. A bundle is
how they travel. What is pinned here: only reviewed files travel, they land
back at the same plan, and an archive cannot write outside the dataset.
"""

import json
import zipfile

import numpy as np
import pytest

from planto3d.classes import WALL


def _plan(root, relative, reviewed):
    from planto3d.window_labels import mask_to_labelme

    folder = root / relative
    folder.mkdir(parents=True)
    image = folder / "F1_scaled.png"
    image.write_bytes(b"not really a png")
    labels = mask_to_labelme(np.full((4, 4), WALL, dtype=np.int64), image_name=image.name)
    labels["flags"]["reviewed"] = reviewed
    (folder / "F1_scaled.json").write_text(json.dumps(labels))
    return image


def test_only_reviewed_labels_go_into_the_bundle(tmp_path):
    from planto3d.window_labels import bundle_reviewed

    root = tmp_path / "data"
    done = _plan(root, "high_quality_architectural/6044", reviewed=True)
    _plan(root, "high_quality_architectural/2564", reviewed=False)

    bundle = tmp_path / "labels.zip"
    count = bundle_reviewed([done, root / "high_quality_architectural/2564/F1_scaled.png"], root, bundle)

    assert count == 1
    with zipfile.ZipFile(bundle) as archive:
        assert archive.namelist() == ["high_quality_architectural/6044/F1_scaled.json"]


def test_unpacking_puts_each_label_beside_the_same_plan(tmp_path):
    from planto3d.window_labels import bundle_reviewed, load_reviewed, unpack_bundle

    laptop = tmp_path / "laptop"
    image = _plan(laptop, "colorful/17", reviewed=True)
    bundle = tmp_path / "labels.zip"
    bundle_reviewed([image], laptop, bundle)

    colab = tmp_path / "colab"
    (colab / "colorful" / "17").mkdir(parents=True)
    assert unpack_bundle(bundle, colab) == 1
    assert load_reviewed(colab / "colorful" / "17" / "F1_scaled.png") is not None


def test_an_archive_cannot_write_outside_the_dataset(tmp_path):
    # A zip entry named ../../something would otherwise land anywhere on the
    # machine the notebook runs on.
    from planto3d.window_labels import unpack_bundle

    bundle = tmp_path / "evil.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape/F1_scaled.json", "{}")

    with pytest.raises(ValueError):
        unpack_bundle(bundle, tmp_path / "data")
    assert not (tmp_path / "escape").exists()


def test_the_export_command_really_bundles(tmp_path):
    # Two faults got past the function-level tests: the flag was never
    # registered, and once it was, the function it calls was never imported.
    # Checking --help caught neither. This runs the real command on a real
    # split file and opens the zip it writes.
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = tmp_path / "data"
    _plan(root, "high_quality_architectural/6044", reviewed=True)
    _plan(root, "high_quality_architectural/2564", reviewed=False)
    (root / "high_quality_architectural/6044/model.svg").write_text("<svg/>")
    (root / "high_quality_architectural/2564/model.svg").write_text("<svg/>")
    split = root / "train.txt"
    split.write_text("/high_quality_architectural/6044/\n/high_quality_architectural/2564/\n")
    bundle = tmp_path / "labels.zip"

    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "export_windows.py"),
         str(root), str(split), "--bundle", str(bundle)],
        capture_output=True, text=True, cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    assert completed.returncode == 0, completed.stderr
    with zipfile.ZipFile(bundle) as archive:
        assert archive.namelist() == ["high_quality_architectural/6044/F1_scaled.json"]


def test_unpacking_ignores_anything_that_is_not_a_label_file(tmp_path):
    from planto3d.window_labels import unpack_bundle

    bundle = tmp_path / "mixed.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("colorful/17/F1_scaled.json", json.dumps({"flags": {"reviewed": True}, "shapes": []}))
        archive.writestr("colorful/17/F1_scaled.png", b"an image should never be replaced")

    root = tmp_path / "data"
    assert unpack_bundle(bundle, root) == 1
    assert not (root / "colorful" / "17" / "F1_scaled.png").exists()
