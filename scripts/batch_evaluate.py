"""Run the whole pipeline over many plans and report where it breaks.

Tuning against one building is how the geometry layer ended up fitted to a
single drawing set. This runs the complete pipeline -- segmentation through
to an exported model -- over a directory of plans and reports what succeeded,
what fell back, and what failed, so decisions can be made against a
distribution rather than one house.

Usage:
    python scripts/batch_evaluate.py <plans_dir> [--checkpoint model.pt] [--limit N]

``plans_dir`` is searched recursively for sample folders containing an
``F1_scaled.png``, which is CubiCasa5K's layout, or for loose image files.
"""

import argparse
import logging
import tempfile
import traceback
from collections import Counter
from pathlib import Path

from planto3d.pipeline import IMAGE_SUFFIXES, run
from planto3d.segment import load_segmenter

# A model is only worth exporting if it has enough geometry to be a building.
MIN_WALLS = 8
MIN_ROOMS = 3


def find_plans(root: Path) -> list[Path]:
    """Sample folders in CubiCasa layout, or loose images."""
    samples = sorted(root.rglob("F1_scaled.png"))
    if samples:
        return samples
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def evaluate(plan: Path, segmenter) -> dict:
    """Run one plan, capturing failures rather than letting them stop the batch."""
    row = {"plan": plan.parent.name or plan.stem, "ok": False, "error": None}
    try:
        with tempfile.TemporaryDirectory(prefix="planto3d_batch_") as workdir:
            result = run(plan, Path(workdir), segmenter=segmenter, crop=False)
            row.update(
                {
                    "walls": result.wall_count,
                    "rooms": result.room_count,
                    "openings": result.opening_count,
                    "named": sum(len(f.named_rooms) for f in result.floors),
                    "scale": result.scale,
                    "scale_source": result.scale_source,
                    "exported": result.model_path is not None,
                    "ok": result.wall_count >= MIN_WALLS and result.room_count >= MIN_ROOMS,
                }
            )
    except Exception as error:  # a bad plan must not end the batch
        row["error"] = f"{type(error).__name__}: {error}"
        logging.debug(traceback.format_exc())
    return row


def main(plans_dir: Path, checkpoint: Path | None, limit: int | None) -> None:
    logging.basicConfig(level=logging.ERROR)
    segmenter = load_segmenter(checkpoint)

    plans = find_plans(Path(plans_dir))[:limit]
    if not plans:
        raise SystemExit(f"no plans found under {plans_dir}")

    print(f"evaluating {len(plans)} plan(s)\n")
    print(f"{'plan':12}{'walls':>7}{'rooms':>7}{'open':>6}{'named':>7}  {'scale':>7}  source")
    print("-" * 62)

    rows = []
    for plan in plans:
        row = evaluate(plan, segmenter)
        rows.append(row)
        if row["error"]:
            print(f"{row['plan']:12}  FAILED  {row['error'][:38]}")
        else:
            scale = f"{row['scale']:.1f}" if row["scale"] else "-"
            print(
                f"{row['plan']:12}{row['walls']:7d}{row['rooms']:7d}{row['openings']:6d}"
                f"{row['named']:7d}  {scale:>7}  {row['scale_source']}"
            )

    good = [r for r in rows if r["ok"]]
    failed = [r for r in rows if r["error"]]
    print("-" * 62)
    print(f"reconstructed:  {len(good)}/{len(rows)}")
    print(f"crashed:        {len(failed)}/{len(rows)}")

    if good:
        print(f"\nmedian walls {sorted(r['walls'] for r in good)[len(good) // 2]}, "
              f"rooms {sorted(r['rooms'] for r in good)[len(good) // 2]}, "
              f"openings {sorted(r['openings'] for r in good)[len(good) // 2]}")
        named = sum(1 for r in good if r["named"] > 0)
        print(f"room names read on {named}/{len(good)}")
        print("\nwhere the scale came from:")
        for source, count in Counter(r["scale_source"] for r in good).most_common():
            print(f"   {source:12} {count:3d}")

    if failed:
        print("\nfailures:")
        for kind, count in Counter(
            r["error"].split(":")[0] for r in failed
        ).most_common():
            print(f"   {kind:24} {count:3d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()
    main(arguments.plans_dir, arguments.checkpoint, arguments.limit)
