"""Export the model's window predictions for correction in LabelMe.

    python scripts/export_windows.py <cubicasa root> <cubicasa root>/train.txt --checkpoint models/unet_cubicasa.pt --limit 30
    python scripts/export_windows.py <cubicasa root> <cubicasa root>/train.txt --status

Writes one LabelMe file beside each image (``F1_scaled.json`` beside
``F1_scaled.png``), so opening the folder in LabelMe shows the predicted
windows already drawn. Fix the rectangles -- move, resize, delete, add, all
labelled ``window`` -- then tick **reviewed** and save. Training only uses
files carrying that tick; see ``planto3d/window_labels.py``.

A file that already exists is never overwritten, so re-running cannot wipe
out corrections. ``--status`` reports how many are done.

Train and validation plans only. The held-out test split is what every
result in this project is measured on, and correcting it and then training
on it would make those results mean nothing, so it is refused.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import cv2

from planto3d.cubicasa import sample_paths
from planto3d.window_labels import bundle_reviewed, labels_path, mask_to_labelme


def refuse_held_out(split_file) -> None:
    """Stop before touching the benchmark's plans."""
    stem = Path(split_file).stem.lower()
    if stem == "test" or stem.startswith("cubicasa_test"):
        raise SystemExit(
            f"refusing to export from {split_file}: those are the held-out test "
            "plans every result is measured on. Use train.txt."
        )


def status(pairs) -> None:
    exported = reviewed = windows = 0
    for image, _ in pairs:
        path = labels_path(image)
        if not path.is_file():
            continue
        exported += 1
        labels = json.loads(path.read_text(encoding="utf-8"))
        if labels.get("flags", {}).get("reviewed") is True:
            reviewed += 1
            windows += sum(1 for s in labels.get("shapes", []) if s.get("label") == "window")
    print(f"{len(pairs)} plan(s) in scope, {exported} exported, {reviewed} reviewed "
          f"({windows} window(s) in reviewed files)")


def main(
    root: Path,
    split_file: Path,
    checkpoint: Path | None,
    limit: int,
    show_status: bool,
    bundle: Path | None = None,
) -> None:
    refuse_held_out(split_file)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    everything = sample_paths(root, split_file)
    if bundle is not None:
        # Every reviewed file in the split, not only the first --limit: a
        # correction made on plan 31 must not be left behind.
        count = bundle_reviewed([image for image, _ in everything], root, bundle)
        print(f"{count} reviewed label file(s) -> {bundle}")
        print("Put it on Drive at MyDrive/planto3d/window_labels.zip for train_on_colab.")
        return

    pairs = everything[:limit]
    if show_status:
        status(pairs)
        return

    from planto3d.segment import load_segmenter

    segmenter = load_segmenter(checkpoint)
    written = kept = 0
    for image_path, _ in pairs:
        target = labels_path(image_path)
        if target.exists():
            kept += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        labels = mask_to_labelme(segmenter(image), image_name=image_path.name)
        target.write_text(json.dumps(labels, indent=1), encoding="utf-8")
        written += 1
        print(f"{image_path.parent.name:8} {len(labels['shapes']):3d} window(s) -> {target.name}")

    print(f"\n{written} written, {kept} already present and left alone")
    print("Open the plan folders in LabelMe, correct the rectangles, tick 'reviewed', save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="folder the split file's paths are relative to")
    parser.add_argument("split_file", type=Path, help="train.txt or val.txt -- never test.txt")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--status", action="store_true", help="report progress and stop")
    parser.add_argument(
        "--bundle", type=Path, default=None,
        help="zip every reviewed label file in the split, for train_on_colab",
    )
    arguments = parser.parse_args()
    main(
        arguments.root,
        arguments.split_file,
        arguments.checkpoint,
        arguments.limit,
        arguments.status,
        bundle=arguments.bundle,
    )
