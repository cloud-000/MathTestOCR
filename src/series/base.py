"""The `Series` contract and shared discovery helpers.

A `Series` is the per-competition extension point. The defaults here implement
the common "one PDF per test" case (used by USAMTS); subclasses override only the
pieces that differ. Nothing here loads a model or does OCR -- that stays in the
pipeline; a series only describes *what* to parse and *how its numbering works*.
"""

from dataclasses import dataclass
from pathlib import Path

from .. import pdf_io

# Page-image extensions recognized when a test is a folder of pages rather than a PDF.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class Test:
    """One parseable test within a series.

    `id` names the output folder (``out/<series>/<id>/``); `source` is the PDF
    file or the folder of page images on disk.
    """

    id: str
    source: Path


def _natural_pages(folder: Path):
    """Page images in a folder, sorted so page_2 precedes page_10."""
    import re

    def key(p: Path):
        nums = [int(n) for n in re.findall(r"\d+", p.stem)]
        return (nums, p.stem)

    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS), key=key
    )


class Series:
    """Base class. Default behavior: each ``*.pdf`` in the data dir is a test."""

    name = "base"
    has_solutions = False
    has_answers = False

    # --- Discovery -------------------------------------------------------
    def discover_tests(self, data_dir):
        """Return the tests found under `data_dir` (default: one per top-level PDF)."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = [Test(id=p.stem, source=p) for p in sorted(root.glob("*.pdf"))]
        return tests

    def test_pages(self, test: Test, workdir):
        """Return page-image paths for `test` in reading order.

        A PDF source is rendered to PNGs in `workdir`; a folder source yields its
        page images directly.
        """
        src = Path(test.source)
        if src.is_dir():
            return _natural_pages(src)
        if src.suffix.lower() == ".pdf":
            return pdf_io.pdf_to_images(src, workdir)
        raise ValueError(f"unsupported test source: {src}")

    # --- Numbering quirks ------------------------------------------------
    def match_marker(self):
        """Return a series-specific marker matcher, or None to use the default.

        A matcher is a callable ``text -> (problem_number, match_end) | None``.
        """
        return None

    def postprocess(self, problems):
        """Hook to clean up the merged problem list. Default: unchanged."""
        return problems

    # --- Solutions -------------------------------------------------------
    def solution_source(self, test: Test):
        """Return the solution source (PDF/folder) for `test`, or None."""
        return None

    def scrape_answers(self, test: Test) -> dict:
        """Return {problem_number: answer} for `test`, or {} if unavailable.

        The answer-key counterpart to `solution_source`: for series that publish
        only an answer key (no worked solutions), the `solutions` command writes
        these as ``problem_<n>_answer.txt``.
        """
        return {}
