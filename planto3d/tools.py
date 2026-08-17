"""Locate the external binaries the pipeline shells out to.

Poppler and Tesseract are installed system-wide, not as Python packages, and
on Windows neither installer reliably puts itself on PATH -- the Tesseract
installer skips it entirely, and a winget package's directory is invisible to
any process that started before the install. Depending on ambient PATH means
the code works in one terminal and fails in another, so each tool is looked
up explicitly and the location is cached.
"""

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", ""))
_PROGRAM_FILES = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
_PROGRAM_FILES_X86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))

_WINGET_PACKAGES = _LOCAL_APP_DATA / "Microsoft" / "WinGet" / "Packages"


def _search_winget(package_prefix: str, executable: str) -> Path | None:
    """Find an executable inside a winget package directory."""
    if not _WINGET_PACKAGES.is_dir():
        return None
    for package in _WINGET_PACKAGES.glob(f"{package_prefix}*"):
        for found in package.rglob(executable):
            return found
    return None


@lru_cache(maxsize=1)
def poppler_bin_dir() -> str | None:
    """Directory holding poppler's binaries, or None to rely on PATH.

    pdf2image takes a directory rather than an executable path.
    """
    on_path = shutil.which("pdftoppm")
    if on_path:
        return None  # pdf2image will find it unaided

    found = _search_winget("oschwartz10612.Poppler", "pdftoppm.exe")
    if found:
        logger.info("using poppler from %s", found.parent)
        return str(found.parent)

    for candidate in (_PROGRAM_FILES, _PROGRAM_FILES_X86):
        for found in candidate.glob("poppler*/**/pdftoppm.exe"):
            logger.info("using poppler from %s", found.parent)
            return str(found.parent)

    logger.warning("poppler not found; PDF rasterization will fail")
    return None


@lru_cache(maxsize=1)
def tesseract_exe() -> str | None:
    """Path to the Tesseract executable, or None to rely on PATH."""
    on_path = shutil.which("tesseract")
    if on_path:
        return None

    for base in (_PROGRAM_FILES, _PROGRAM_FILES_X86, _LOCAL_APP_DATA / "Programs"):
        candidate = base / "Tesseract-OCR" / "tesseract.exe"
        if candidate.is_file():
            logger.info("using tesseract from %s", candidate)
            return str(candidate)

    found = _search_winget("UB-Mannheim.TesseractOCR", "tesseract.exe")
    if found:
        logger.info("using tesseract from %s", found)
        return str(found)

    logger.warning("tesseract not found; OCR will fail")
    return None


def configure_tesseract() -> None:
    """Point pytesseract at the executable when it is not on PATH."""
    import pytesseract

    location = tesseract_exe()
    if location:
        pytesseract.pytesseract.tesseract_cmd = location
