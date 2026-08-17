"""Rasterize the floor plan PDF and crop every page to the shared drawing area.

Usage:
    python scripts/crop_pages.py <pdf_path> <output_dir>
"""

import logging
import sys
from pathlib import Path

import cv2

from planto3d.ingest import crop_pages, rasterize_pdf


def main(pdf_path: str, output_dir: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pages = rasterize_pdf(Path(pdf_path), Path(output_dir))
    print(f"rasterized {len(pages)} page(s)")

    cropped_paths = crop_pages(pages)
    for page, cropped_path in zip(pages, cropped_paths):
        original = cv2.imread(str(page))
        cropped = cv2.imread(str(cropped_path))
        print(
            f"  {page.name}: {original.shape[1]}x{original.shape[0]}"
            f" -> {cropped.shape[1]}x{cropped.shape[0]}  ({cropped_path.name})"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
