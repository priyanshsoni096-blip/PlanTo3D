"""Compare two checkpoints on every held-out measurement, in one command.

    python scripts/compare_checkpoints.py data/cubicasa5k --baseline models/unet_cubicasa.pt --candidate models/unet_cubicasa_fixedwindows.pt

Deciding whether a retrained model replaces the installed one used to mean
running six measurement scripts twice each and reading twelve summaries side
by side. This runs them all, reads each summary, and prints one table with
every figure marked better or worse for the candidate.

It reads the scripts' printed summaries rather than importing them, so every
number is exactly what that script reports on its own. A script that fails or
prints no summary stops the comparison with its name: a missing figure must
never read as "unchanged".

It does not decide. It lists what got better and what got worse, and which
of those matter is a judgement recorded in docs/AUDIT.md, not a rule coded
here.

    --limit N     plans per script (default 60, the held-out set)
    --jobs N      scripts run at once (default 6; each holds a model in memory)
    --save DIR    also write every script's raw output there, for the record
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _value(pattern: str, text: str, script: str, what: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(
            f"{script}: could not find {what} in its output -- the script "
            "failed or its summary changed, so this figure cannot be compared"
        )
    return float(match.group(1))


def parse_class_accuracy(text: str) -> dict[str, float]:
    script = "class_accuracy"
    row = r"^{name}\s+\d+\s+[\d.]+%\s+([\d.]+)\s+([\d.]+)%"
    figures = {}
    for name in ("wall", "door", "window"):
        figures[f"{name} IoU"] = _value(row.format(name=name), text, script, f"the {name} row")
    recall = re.search(row.format(name="window"), text, re.MULTILINE)
    figures["window pixel recall %"] = float(recall.group(2))
    return figures


def parse_scale_accuracy(text: str) -> dict[str, float]:
    script = "scale_accuracy"
    return {
        "scale median error %": _value(r"^median error\s+([\d.]+)%", text, script, "the median error"),
        "plans scaled within a fifth": _value(r"^within 20%\s+(\d+)/\d+", text, script, "plans within 20%"),
    }


def parse_scorecard(text: str) -> dict[str, float]:
    script = "output_scorecard"
    figures = {
        "right on every check": _value(
            r"correct on every count:\s*(\d+)/\d+", text, script, "the scorecard headline"
        )
    }
    for check, count in re.findall(r"^\s+(\w+)\s+(\d+) of \d+\s*$", text, re.MULTILINE):
        figures[f"fails on {check}"] = float(count)
    return figures


def parse_wall_accuracy(text: str) -> dict[str, float]:
    script = "wall_accuracy"
    return {
        "wall coverage %": _value(r"^coverage\s+\(.*?\)\s+([\d.]+)%", text, script, "median coverage"),
        "wall agreement %": _value(r"^agreement\s+\(.*?\)\s+([\d.]+)%", text, script, "median agreement"),
    }


def parse_window_detection(text: str) -> dict[str, float]:
    script = "window_detection_accuracy"
    return {
        "window detection recall %": _value(r"^recall:\s+([\d.]+)%", text, script, "recall"),
        "window detection precision %": _value(r"^precision:\s+([\d.]+)%", text, script, "precision"),
    }


def parse_open_air(text: str) -> dict[str, float]:
    return {"open-air IoU %": _value(r"^IoU\s+([\d.]+)%", text, "open_air_accuracy", "the IoU headline")}


# Script, parser. Every script takes root, --checkpoint and --limit.
MEASURES = [
    ("class_accuracy.py", parse_class_accuracy),
    ("scale_accuracy.py", parse_scale_accuracy),
    ("output_scorecard.py", parse_scorecard),
    ("wall_accuracy.py", parse_wall_accuracy),
    ("window_detection_accuracy.py", parse_window_detection),
    ("open_air_accuracy.py", parse_open_air),
]


def lower_is_better(name: str) -> bool:
    return name.startswith("scale median error") or name.startswith("fails on")


@dataclass
class Row:
    name: str
    baseline: float
    candidate: float
    better: bool | None  # None when the two are equal


def compare(baseline: dict[str, float], candidate: dict[str, float]) -> list[Row]:
    rows = []
    for name, base in baseline.items():
        if name not in candidate:
            raise ValueError(f"{name}: present for the baseline but not the candidate")
        cand = candidate[name]
        if cand == base:
            better = None
        else:
            better = (cand < base) if lower_is_better(name) else (cand > base)
        rows.append(Row(name, base, cand, better))
    return rows


def _run(script: str, root: str, checkpoint: str, limit: int) -> str:
    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), root, "--checkpoint", checkpoint, "--limit", str(limit)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout + completed.stderr


def measure_both(root: str, baseline: str, candidate: str, limit: int, jobs: int, save: Path | None):
    work = [(script, parser, label, checkpoint)
            for label, checkpoint in (("baseline", baseline), ("candidate", candidate))
            for script, parser in MEASURES]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outputs = list(pool.map(lambda item: _run(item[0], root, item[3], limit), work))

    figures = {"baseline": {}, "candidate": {}}
    for (script, parser, label, _), output in zip(work, outputs):
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
            (save / f"{label}_{script.removesuffix('.py')}.log").write_text(output, encoding="utf-8")
        figures[label].update(parser(output))
    return figures["baseline"], figures["candidate"]


def _format(name: str, value: float) -> str:
    # By what the figure is, not its size: 97.0% must not print as "97"
    # beside 97.6%, and a count of plans is never 44.0.
    if name.endswith("IoU"):
        return f"{value:.3f}"
    if name.endswith("%"):
        return f"{value:.1f}"
    return f"{value:.0f}"


def print_table(rows: list[Row], baseline: str, candidate: str) -> None:
    width = max(len(row.name) for row in rows)
    print(f"baseline:  {baseline}\ncandidate: {candidate}\n")
    print(f"{'measure':{width}}  {'baseline':>9}  {'candidate':>9}")
    print("-" * (width + 26))
    for row in rows:
        mark = {True: "better", False: "WORSE", None: "same"}[row.better]
        print(f"{row.name:{width}}  {_format(row.name, row.baseline):>9}  "
              f"{_format(row.name, row.candidate):>9}  {mark}")
    worse = [row.name for row in rows if row.better is False]
    better = [row.name for row in rows if row.better is True]
    print("-" * (width + 26))
    print(f"{len(better)} better, {len(worse)} worse, {len(rows) - len(better) - len(worse)} the same")
    if worse:
        print("worse: " + ", ".join(worse))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="CubiCasa folder, e.g. data/cubicasa5k")
    parser.add_argument("--baseline", required=True, help="the installed checkpoint")
    parser.add_argument("--candidate", required=True, help="the checkpoint that might replace it")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--save", type=Path, default=None)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    base, cand = measure_both(
        arguments.root, arguments.baseline, arguments.candidate,
        arguments.limit, arguments.jobs, arguments.save,
    )
    print_table(compare(base, cand), arguments.baseline, arguments.candidate)


if __name__ == "__main__":
    main()
