"""Score sheet splitting against the floor count CubiCasa records.

A sheet holding three storeys must become three storeys, and a sheet
holding one must be left alone. Both failures are costly and they look
nothing alike: splitting a single plan into three stacks fragments of one
apartment into a tower, and the fragments are small enough to throw the
scale estimate badly off. Missing a real split lays several plans out as
one flat floor.

CubiCasa records the floors it holds as ``Floor`` groups in the annotation,
which gives a ground truth for every sample.

    python scripts/split_accuracy.py path/to/cubicasa

Report precision and recall separately. A splitter that never fires scores
well on one and nothing on the other.
"""

import argparse
import logging
import re
import warnings
from collections import Counter
from pathlib import Path

from planto3d.ingest import read_image, split_sheet

warnings.filterwarnings("ignore")

_FLOOR_GROUP = re.compile(r'class="Floor"')


def true_floors(svg_path: Path) -> int:
    """How many floors the annotation says the sheet holds."""
    text = svg_path.read_text(encoding="utf-8", errors="ignore")
    return max(len(_FLOOR_GROUP.findall(text)), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--verbose", action="store_true", help="list every disagreement")
    arguments = parser.parse_args()

    logging.disable(logging.WARNING)

    exact = 0
    scored = 0
    confusion: Counter = Counter()
    disagreements = []

    for image_path in sorted(arguments.root.glob("*/*/F1_scaled.png"))[: arguments.limit]:
        annotation = image_path.parent / "model.svg"
        if not annotation.is_file():
            continue

        expected = true_floors(annotation)
        try:
            found = len(split_sheet(read_image(image_path)))
        except Exception as error:
            print(f"{image_path.parent.name:10} FAILED {type(error).__name__}: {error}")
            continue

        scored += 1
        exact += found == expected
        confusion[(expected, found)] += 1
        if found != expected:
            disagreements.append((image_path.parent.name, expected, found))

    if not scored:
        print("nothing scored")
        return

    # A sheet counts as multi-storey if it holds more than one plan. Judging
    # the split as a yes/no question separates "fires at the right time"
    # from "counts correctly once it has fired".
    fired_and_should = sum(c for (e, f), c in confusion.items() if e > 1 and f > 1)
    fired_total = sum(c for (e, f), c in confusion.items() if f > 1)
    should_total = sum(c for (e, f), c in confusion.items() if e > 1)

    print(f"scored {scored} sheet(s)\n")
    print(f"exact floor count   {exact}/{scored}  ({exact / scored:.0%})")
    print()
    print("treating it as: does this sheet hold more than one plan?")
    print(f"   precision        {fired_and_should}/{fired_total}"
          f"  ({fired_and_should / fired_total:.0%})" if fired_total else "   precision        never fired")
    print(f"   recall           {fired_and_should}/{should_total}"
          f"  ({fired_and_should / should_total:.0%})" if should_total else "   recall           n/a")
    print()
    print(f"{'expected':>10}{'found':>8}{'count':>8}")
    for (expected, found), count in sorted(confusion.items()):
        mark = "" if expected == found else "   <-- wrong"
        print(f"{expected:>10}{found:>8}{count:>8}{mark}")

    if arguments.verbose and disagreements:
        print("\ndisagreements:")
        for name, expected, found in disagreements:
            print(f"   {name:10} expected {expected}, found {found}")


if __name__ == "__main__":
    main()
