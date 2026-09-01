"""Correct what the plan reader got wrong, then build the model.

Room labels decide what actually gets built -- railings, paving, planting,
open-to-sky treatment, floor finishes. They come from OCR, and on real
drawings OCR frequently reads nothing at all: a three-storey plan measured
here read 0 of 55 room names. The model still builds, but far plainer than
the drawing deserves, because nothing told it which spaces are terraces.

A correction is a room label that did not come from OCR. Everything
downstream already prefers a printed label over the segmenter's predicted
type -- ``features.feature_for``, ``site.classify_cover``,
``site.has_open_edge`` all read ``room.label`` first -- so saying "room 5 is
a balcony" needs no new machinery, only somewhere to say it. That place is
the seam between ``pipeline.extract`` and ``pipeline.build``.

Two steps, the same two the pipeline itself has:

    # 1. read the drawing and see what it found
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt --list

    # 2. fix what is wrong and build it
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --correct 1:5=open --correct 1:6=paving

``--correct`` takes ``FLOOR:ROOM=CATEGORY``, using the floor and room
numbers ``--list`` prints. Floors are numbered from 1 because that is how
the listing reads them out; rooms from 0 because that is their index.

The categories are the feature vocabulary, not the segmenter's room types:
plain room, water, lawn, void, paving, open, tank, chimney, tower, canopy,
ramp, dome, glazed, pitched, stairs, wet. They are different vocabularies
and share no words -- a room the model calls "outdoor" is corrected with
"open", not "outdoor". ``--list`` prints both so the difference is visible,
and an unknown category is refused with the full set rather than guessed at.
"""

import argparse
import copy
import logging
from pathlib import Path

import cv2

from planto3d.corrections import CATEGORY_LABELS, apply_room_corrections
from planto3d.features import feature_for
from planto3d.pipeline import build, draw_overlay, extract
from planto3d.preview import render_views
from planto3d.segment import load_segmenter

# Width of the label column in the listing. Long enough for the room names
# these plans actually print -- "CHEF'S KITCHEN/WASH AREA" is the longest
# seen -- without wrapping the line on an 80-column terminal.
LABEL_WIDTH = 26


def parse_correction(text: str) -> tuple[tuple[int, int], str]:
    """Turn ``FLOOR:ROOM=CATEGORY`` into the key and value corrections wants.

    Raises ``ValueError`` with the whole vocabulary rather than a bare
    "invalid" -- the two vocabularies in play look alike enough that a
    wrong guess is the expected mistake, not an exceptional one.
    """
    try:
        where, category = text.split("=", 1)
        floor_text, room_text = where.split(":", 1)
        floor, room = int(floor_text), int(room_text)
    except ValueError:
        raise ValueError(
            f"could not read correction {text!r}; expected FLOOR:ROOM=CATEGORY, "
            "for example 1:5=open"
        ) from None

    category = category.strip().lower()
    if category not in CATEGORY_LABELS:
        raise ValueError(
            f"{category!r} is not a feature category. One of: "
            + ", ".join(sorted(CATEGORY_LABELS))
        )

    # Floors are printed from 1 and indexed from 0.
    return (floor - 1, room), category


def show(result) -> None:
    """Print every room, so a correction can be aimed at one."""
    print(f"\n{'floor':>5} {'room':>4}  {'detected label':{LABEL_WIDTH}} "
          f"{'predicted type':16} feature")
    print("-" * (5 + 5 + LABEL_WIDTH + 18 + 10))
    for floor in result.floors:
        for index, room in enumerate(floor.plan.rooms):
            print(
                f"{floor.index + 1:>5} {index:>4}  "
                f"{(room.label or '--'):{LABEL_WIDTH}.{LABEL_WIDTH}} "
                f"{(room.category or '--'):16} {feature_for(room) or '--'}"
            )

    named = sum(len(floor.named_rooms) for floor in result.floors)
    print("-" * (5 + 5 + LABEL_WIDTH + 18 + 10))
    print(f"{named} of {result.room_count} room(s) carry a label read off the drawing")
    if named == 0:
        print(
            "\nNo room names were read. Floor finishes, planting, paving, railings\n"
            "and stairs all come from the labels, so the model will be much plainer\n"
            "than it should be. Correct the rooms that matter with --correct, or\n"
            "supply a higher-resolution scan."
        )


def main(
    source: str,
    output_dir: str,
    checkpoint: Path | None,
    corrections: list[str],
    listing: bool,
    wall_height_ft: float,
    crop: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(output_dir)

    result = extract(Path(source), out, segmenter=load_segmenter(checkpoint), crop=crop)

    scale = f"{result.scale:.2f} px/ft" if result.scale else "unknown"
    print(
        f"\n{len(result.floors)} storey(s), {result.wall_count} walls, "
        f"{result.room_count} rooms, {result.opening_count} openings, scale {scale}"
        f"{' (inferred)' if result.scale_assumed else ''}"
    )

    for floor in result.floors:
        cv2.imwrite(str(out / f"overlay-{floor.index}.png"), draw_overlay(floor))
    print(f"overlay(s) written to {out}")

    show(result)

    if listing:
        print("\n--list given; stopping before the build.")
        return

    if corrections:
        parsed = dict(parse_correction(text) for text in corrections)
        # Corrected off a copy, so the extraction stays pristine. Applying
        # successive passes to one result makes an override impossible to
        # undo: dropping a room back to no correction only skips re-writing
        # the label, leaving it stuck at whatever it was last set to.
        result = apply_room_corrections(copy.deepcopy(result), parsed)
        print(f"\napplied {len(parsed)} correction(s):")
        for (floor, room), category in sorted(parsed.items()):
            print(f"   floor {floor + 1} room {room} -> {category}")
        show(result)

    result = build(result, out, wall_height_ft=wall_height_ft)
    if result.model_path is None:
        print("\nno model was built -- no walls were found in this drawing")
        return

    views = render_views(result.model_path, out)
    print(f"\nmodel:  {result.model_path}")
    print(f"views:  {', '.join(sorted(views))}")
    print(f"output written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="PDF, image, or directory of images")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="trained segmenter; omit to use the classical baseline",
    )
    parser.add_argument(
        "--correct",
        action="append",
        default=[],
        metavar="FLOOR:ROOM=CATEGORY",
        help="override one room's feature, e.g. 1:5=open; repeatable",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="listing",
        help="show what was detected and stop, without building",
    )
    parser.add_argument("--wall-height-ft", type=float, default=9.0)
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="skip title-block cropping, for images that are already just the plan",
    )
    arguments = parser.parse_args()
    main(
        arguments.source,
        arguments.output_dir,
        arguments.checkpoint,
        arguments.correct,
        arguments.listing,
        arguments.wall_height_ft,
        crop=not arguments.no_crop,
    )
