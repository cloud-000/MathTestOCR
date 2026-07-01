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
    # When True, the `solutions` command OCRs each solution page to markdown,
    # concatenates every page into one string, and hands the whole thing to
    # `parse_solutions` -- letting the series segment its solution document its
    # own way (multi-page spans, "Solution N by ..." blocks) instead of the
    # per-page marker pipeline. Requires the nanonets engine; also skips DETR
    # detection, so it is faster on text-only solution PDFs.
    custom_solution_parser = False

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

        A PDF source is rendered to PNGs in `workdir`, dropping any page
        `skip_page` rejects; a folder source yields its page images directly
        (no skip filtering -- there is no embedded text to check cheaply).
        """
        src = Path(test.source)
        if src.is_dir():
            return _natural_pages(src)
        if src.suffix.lower() == ".pdf":
            return pdf_io.pdf_to_images(src, workdir, skip_page=self.skip_page)
        raise ValueError(f"unsupported test source: {src}")

    def skip_page(self, text: str) -> bool:
        """Return True to exclude a rendered PDF page from parsing.

        `text` is that page's embedded PDF text (empty for a scanned page with
        no text layer, in which case the default of keeping the page is the
        only safe choice). Override to drop cover sheets, instructions pages,
        or trailing answer-format pages without spending OCR on them.
        """
        return False

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

    def parse_solutions(self, full_text: str) -> dict:
        """Segment the whole-test solution OCR into {problem_number: text}.

        Called only when `custom_solution_parser` is True; `full_text` is every
        solution page's markdown concatenated in reading order. The default
        splits on this series' problem markers (reusing the nanonets layout
        splitter) and joins each problem's text across page breaks. Override to
        implement a series-specific solution layout.
        """
        from ..nanonets import parse_layout

        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["kind"] == "text" and item["problem"] is not None:
                grouped.setdefault(item["problem"], []).append(item["text"])
        return {n: "\n".join(parts) for n, parts in grouped.items()}
