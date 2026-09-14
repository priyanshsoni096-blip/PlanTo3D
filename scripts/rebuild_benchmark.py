"""Rebuild the held-out CubiCasa benchmark from the dataset archive.

    python scripts/rebuild_benchmark.py path/to/cubicasa5k.zip
    python scripts/rebuild_benchmark.py path/to/cubicasa5k.zip --verify-list

Every CubiCasa figure in docs/AUDIT.md and the README is measured on 60 plans
from the dataset's held-out test split. They once lived only in a session
temp folder and were lost with it, so the list is committed as
data/cubicasa_test60.txt and this turns it back into files: each plan's
model.svg and F1_scaled.png, extracted into data/cubicasa5k/ (git-ignored,
since the dataset is not ours to redistribute).

The list was drawn with ``draw``: 20 plans per category from test.txt, with
random.Random(20260913). ``--verify-list`` re-runs that draw against the
archive and checks it still produces the committed list exactly.

Nothing already on disk is overwritten, so a re-run cannot undo files a
person has put beside the plans. A plan the archive lacks, or a list entry
that would write outside the output folder, stops the run before anything is
written: a benchmark silently one plan short would be reported as 60.
"""

import argparse
import random
import sys
import zipfile
from pathlib import Path

ARCHIVE_ROOT = "cubicasa5k"
FILES = ("model.svg", "F1_scaled.png")

# How data/cubicasa_test60.txt was drawn.
BENCHMARK_SEED = 20260913
BENCHMARK_PER_CATEGORY = 20


def read_list(path: Path) -> list[str]:
    """Entries like ``colorful/9448`` from a CubiCasa split-style list."""
    return read_list_text(Path(path).read_text())


def draw(split: list[str], per_category: int, seed: int) -> list[str]:
    """The benchmark draw: a seeded sample per category, sorted by plan number.

    Categories are sampled in sorted order from one generator, so the same
    split and seed always give the same list.
    """
    by_category: dict[str, list[str]] = {}
    for entry in split:
        by_category.setdefault(entry.split("/")[0], []).append(entry)

    rng = random.Random(seed)
    chosen: list[str] = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda entry: int(entry.split("/")[1]))
        sample = rng.sample(pool, per_category)
        chosen += sorted(sample, key=lambda entry: int(entry.split("/")[1]))
    return chosen


def extract(archive_path: Path, entries: list[str], out_root: Path) -> tuple[int, int]:
    """Write each entry's files under ``out_root``. Returns (written, kept)."""
    out_root = Path(out_root)
    root = out_root.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

        # Every entry is checked before any file is written.
        for entry in entries:
            target = (root / entry).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"refusing {entry!r}: it would write outside {root}")
            missing = [name for name in FILES if f"{ARCHIVE_ROOT}/{entry}/{name}" not in names]
            if missing:
                raise ValueError(f"{entry} is not in the archive (missing {', '.join(missing)})")

        written = kept = 0
        for entry in entries:
            for name in FILES:
                target = out_root / entry / name
                if target.exists():
                    kept += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(f"{ARCHIVE_ROOT}/{entry}/{name}"))
                written += 1
    return written, kept


def verify_list(archive_path: Path, list_path: Path) -> bool:
    with zipfile.ZipFile(archive_path) as archive:
        split = read_list_text(archive.read(f"{ARCHIVE_ROOT}/test.txt").decode())
    return draw(split, BENCHMARK_PER_CATEGORY, BENCHMARK_SEED) == read_list(list_path)


def read_list_text(text: str) -> list[str]:
    return [line.strip().strip("/") for line in text.splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("archive", type=Path, help="cubicasa5k.zip")
    parser.add_argument("--list", type=Path, default=Path("data/cubicasa_test60.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument(
        "--verify-list", action="store_true",
        help="re-run the seeded draw and check it still gives the committed list",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.verify_list:
        matches = verify_list(arguments.archive, arguments.list)
        print(f"{arguments.list} {'matches' if matches else 'does NOT match'} the seeded draw")
        sys.exit(0 if matches else 1)

    entries = read_list(arguments.list)
    written, kept = extract(arguments.archive, entries, arguments.out)
    print(f"{len(entries)} plans: {written} file(s) written, {kept} already present -> {arguments.out}")


if __name__ == "__main__":
    main()
