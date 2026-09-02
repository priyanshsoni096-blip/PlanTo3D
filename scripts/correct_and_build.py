"""Correct what the plan reader got wrong, then build the model.

Room labels decide what actually gets built -- railings, paving, planting,
open-to-sky treatment, floor finishes. They come from OCR, and on real
drawings OCR frequently reads nothing at all: a three-storey plan measured
here read 0 of 55 room names. The model still builds, but far plainer than
the drawing deserves, because nothing told it which spaces are terraces.

A correction is a room label that did not come from OCR. Everything
downstream already prefers a printed label over the segmenter's predicted
type -- ``features.feature_for`` and ``site.classify_cover`` both read
``room.label`` first -- so saying "room 5 is a balcony" needs no new
machinery, only somewhere to say it. That place is the seam between
``pipeline.extract`` and ``pipeline.build``.

Two steps, the same two the pipeline itself has:

    # 1. read the drawing and see what it found
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt --list

    # 2. fix what is wrong and build it
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --correct 1:5=open --correct 1:6=paving

    # 3. save the corrections, and reuse them on every later run
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --correct 1:5=open --save-corrections plan-corrections.txt
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --corrections plan-corrections.txt

``--correct`` takes ``FLOOR:ROOM=CATEGORY``, using the floor and room
numbers ``--list`` prints. Floors are numbered from 1 because that is how
the listing reads them out; rooms from 0 because that is their index.

The categories are the feature vocabulary, not the segmenter's room types:
plain room, water, lawn, void, paving, open, tank, chimney, tower, canopy,
ramp, dome, glazed, pitched, stairs, wet. They are different vocabularies
and share no words -- a room the model calls "outdoor" is corrected with
"open", not "outdoor". ``--list`` prints both so the difference is visible,
and an unknown category is refused with the full set rather than guessed at.

A sheet carrying two floors is sometimes read as one, or a single plan is
occasionally cut in two. ``--split N`` and ``--no-split`` settle it by hand:

    # the automatic splitter missed the second floor -- force two plans
    python scripts/correct_and_build.py plan.pdf out --split 2

    # the automatic splitter wrongly cut one plan in half -- keep it whole
    python scripts/correct_and_build.py plan.pdf out --no-split

``--split`` uses the dividing line the splitter already found and skips only
the checks that reject it; it cannot invent a division where none was
proposed, and raises rather than guess when that happens.

Every inferred scale rests on an assumption about a standard element -- a
2'6" door, a 9" wall -- that does not hold on every building. ``--scale-room``
sidesteps all of that: read one room's real size off the drawing or off the
building itself, and the whole model is sized from it instead.

    # 1. see which room is which
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt --list

    # 2. state room 5 on floor 1 is 12ft x 10ft, and build at that scale
    python scripts/correct_and_build.py plan.pdf out --checkpoint models/unet_cubicasa.pt \
        --scale-room 1:5=12x10

``--scale-room`` takes ``FLOOR:ROOM=WxH``, the same floor/room numbering
``--list`` prints and ``--correct`` uses, with W and H the room's real width
and height in feet. Because the scale gates room filtering and label
placement inside ``extract``, the sheet is read a second time once the
stated scale is known, rather than patching the first result's number.
"""

import argparse
import copy
import logging
from pathlib import Path

import cv2

from planto3d.calibrate import scale_from_known_room
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

    # Floors are printed from 1 and rooms from 0 -- see --list's own output.
    # Unchecked, ``floor - 1`` on a floor below 1 wraps to a negative index
    # and silently corrects some other floor (0 -> -1 hits the last one),
    # and a negative room does the same without even that subtraction. That
    # was survivable while corrections were only ever typed as flags; the
    # corrections file this module can also write makes off-by-one on floor
    # numbering the expected mistake, not a rare one, so it must be caught
    # here rather than quietly relabelling the wrong room.
    if floor < 1:
        raise ValueError(
            f"floor {floor} in correction {text!r} is not valid; floors are "
            "numbered from 1, as --list prints them"
        )
    if room < 0:
        raise ValueError(
            f"room {room} in correction {text!r} is not valid; rooms are "
            "indexed from 0, as --list prints them"
        )

    category = category.strip().lower()
    if category not in CATEGORY_LABELS:
        raise ValueError(
            f"{category!r} is not a feature category. One of: "
            + ", ".join(sorted(CATEGORY_LABELS))
        )

    return (floor - 1, room), category


def parse_scale_room(text: str) -> tuple[tuple[int, int], float, float]:
    """Turn ``FLOOR:ROOM=WxH`` into the room to measure and its real size."""
    try:
        where, size = text.split("=", 1)
        floor_text, room_text = where.split(":", 1)
        width_text, height_text = size.lower().split("x", 1)
        floor, room = int(floor_text), int(room_text)
        width, height = float(width_text), float(height_text)
    except ValueError:
        raise ValueError(
            f"could not read {text!r}; expected FLOOR:ROOM=WxH in feet, "
            "for example 1:5=12x10"
        ) from None

    if floor < 1:
        raise ValueError(f"floor {floor} is not valid; floors are numbered from 1")
    if room < 0:
        raise ValueError(f"room {room} is not valid; rooms are indexed from 0")
    return (floor - 1, room), width, height


# The file format is the flag's own syntax, one per line. A user who can
# type a correction can read the file without being taught anything new,
# and the file can be edited by hand or produced by another tool.
CORRECTIONS_HEADER = (
    "# PlanTo3D room corrections. One FLOOR:ROOM=CATEGORY per line.\n"
    "# Floor numbers are the ones --list prints; rooms are indexed from 0.\n"
)


def corrections_to_lines(corrections: dict[tuple[int, int], str]) -> list[str]:
    """Render corrections as the same text ``--correct`` accepts."""
    return [
        f"{floor + 1}:{room}={category}"
        for (floor, room), category in sorted(corrections.items())
    ]


def corrections_from_lines(lines: list[str]) -> dict[tuple[int, int], str]:
    """Read corrections back, ignoring comments and blank lines.

    Reuses ``parse_correction`` so the file and the flag can never drift
    apart, and so a bad line is refused with the whole vocabulary rather
    than silently skipped -- a correction that quietly does nothing is
    worse than one that stops you.
    """
    corrections = {}
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        where, category = parse_correction(text)
        corrections[where] = category
    return corrections


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


def _print_extraction(result, out: Path) -> None:
    """Summarize an extraction, write its overlays, and list its rooms.

    Pulled out so the summary can be printed twice in one run: once for the
    initial read, and again after a ``--scale-room`` override re-reads the
    sheet, since the room count, opening count and "inferred" warning can
    all change once the real scale is known.
    """
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


def main(
    source: str,
    output_dir: str,
    checkpoint: Path | None,
    corrections: list[str],
    listing: bool,
    wall_height_ft: float,
    crop: bool,
    corrections_path: Path | None,
    save_path: Path | None,
    split: int | None = None,
    scale_room: str | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(output_dir)

    result = extract(
        Path(source), out, segmenter=load_segmenter(checkpoint), crop=crop, split=split
    )
    _print_extraction(result, out)

    if scale_room is not None:
        (floor_index, room_index), width, height = parse_scale_room(scale_room)
        try:
            floor = result.floors[floor_index]
        except IndexError:
            raise ValueError(
                f"floor {floor_index + 1} does not exist; this plan has "
                f"{len(result.floors)} floor(s)"
            ) from None
        try:
            room = floor.plan.rooms[room_index]
        except IndexError:
            raise ValueError(
                f"room {room_index} does not exist on floor {floor_index + 1}; "
                f"it has {len(floor.plan.rooms)} room(s)"
            ) from None

        stated = scale_from_known_room(room, width, height)
        if stated is None:
            raise ValueError(
                f"room {floor_index + 1}:{room_index} has no measurable area; "
                "pick a different room"
            )

        print(
            f"\nroom {floor_index + 1}:{room_index} stated as {width}x{height} ft "
            f"-> {stated:.2f} px/ft (was {result.scale:.2f}); re-reading the sheet"
        )
        # A patched result.scale would leave room filtering (MIN_ROOM_SQFT)
        # and labelled-region placement computed against the old number, so
        # the sheet is read again rather than the number swapped in place.
        result = extract(
            Path(source),
            out,
            segmenter=load_segmenter(checkpoint),
            crop=crop,
            split=split,
            scale_override=stated,
        )
        _print_extraction(result, out)

    if listing:
        if corrections or corrections_path is not None or save_path is not None:
            # --list stops before the corrections block runs at all, so any
            # of these would otherwise be silently ignored: --correct is
            # never parsed, --corrections is never read, and --save-corrections
            # writes nothing -- with --list printing no hint that it did not.
            print(
                "\n--list given; --correct/--corrections/--save-corrections "
                "are not applied or written this run. Drop --list to apply "
                "and save them."
            )
        print("\n--list given; stopping before the build.")
        return

    parsed = dict(parse_correction(text) for text in corrections)
    if corrections_path is not None:
        # The file first, the flags second, so a one-off --correct can
        # override a saved annotation without editing the file.
        from_file = corrections_from_lines(
            corrections_path.read_text(encoding="utf-8").splitlines()
        )
        print(f"\nread {len(from_file)} correction(s) from {corrections_path}")
        parsed = {**from_file, **parsed}

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            CORRECTIONS_HEADER + "\n".join(corrections_to_lines(parsed)) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(parsed)} correction(s) to {save_path}")

    if parsed:
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
        "--corrections",
        type=Path,
        default=None,
        metavar="PATH",
        help="read corrections from a file written by --save-corrections",
    )
    parser.add_argument(
        "--save-corrections",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the corrections given on this run to a file, to reuse later",
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
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        metavar="N",
        help="force the sheet into N plans, overriding the splitter",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="force the sheet to be read as a single plan",
    )
    parser.add_argument(
        "--scale-room",
        metavar="FLOOR:ROOM=WxH",
        help="set the scale from one room's real size in feet, e.g. 1:5=12x10",
    )
    arguments = parser.parse_args()
    if arguments.no_split and arguments.split is not None:
        parser.error("--split and --no-split contradict each other")
    main(
        arguments.source,
        arguments.output_dir,
        arguments.checkpoint,
        arguments.correct,
        arguments.listing,
        arguments.wall_height_ft,
        crop=not arguments.no_crop,
        corrections_path=arguments.corrections,
        save_path=arguments.save_corrections,
        split=1 if arguments.no_split else arguments.split,
        scale_room=arguments.scale_room,
    )
