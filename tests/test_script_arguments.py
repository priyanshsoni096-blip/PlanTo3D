"""Every script must answer --help instead of crashing.

Two took their arguments straight from sys.argv: `crop_pages.py` raised
IndexError on --help, and `random_plan.py` raised ValueError trying to read
"--help" as a seed -- after it had already deleted its output directory at
import time, so asking a question destroyed work.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(path.name for path in (REPO / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    assert len(SCRIPTS) > 20


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_answers_help(script):
    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--help"],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": ""},
    )
    assert completed.returncode == 0, completed.stderr[-400:]
    assert "usage" in (completed.stdout + completed.stderr).lower()


def test_asking_for_help_does_not_delete_the_output_directory(tmp_path):
    output = tmp_path / "output_random"
    output.mkdir()
    (output / "keep.txt").write_text("work from an earlier run")

    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "random_plan.py"), "--help"],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "", "PLANTO3D_OUT": str(output)},
    )

    assert completed.returncode == 0
    assert (output / "keep.txt").is_file()
