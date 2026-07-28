"""The `Series` contract and shared discovery helpers.

A `Series` is the per-competition extension point. The defaults here implement
the common "one PDF per test" case (used by USAMTS); subclasses override only the
pieces that differ. Nothing here loads a model or does OCR -- that stays in the
pipeline; a series only describes *what* to parse and *how its numbering works*.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .. import config, pdf_io

# Page-image extensions recognized when a test is a folder of pages rather than a PDF.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# --- Answer-key line parsing (shared by series answer parsers) ---
# One "N." / "N)" entry inside an answer-key line. Several entries often share a
# line (multi-column keys OCR'd row-wise: "1. 5   5. 3025"), so entries are
# found by this marker: a 1-3 digit number at the start or after whitespace,
# with whitespace after the dot. A decimal answer like "12.5" never has a space
# after its point, and 4-digit years ("2023.") are too long, so neither splits
# an entry.
_ANSWER_ENTRY_RE = re.compile(r"(?:^|(?<=\s))(\d{1,3})\s*[.)]\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_RUN_RE = re.compile(r"_{2,}")  # answer blanks: "1. ________ 42"
_SOLUTION_INDEX_RE = re.compile(
    r"^\s*(?:\*{1,2}|#+\s*|\\textbf\{)?\s*Solution\s+(\d+)\b",
    re.IGNORECASE,
)


def numbered_answers_in_line(line: str):
    """Extract answer-key entries from one OCR'd line.

    Returns ``(leading, pairs)``: `leading` is the text before the first entry
    (empty for a pure answer-key line -- callers use it to tell key lines from
    prose that merely mentions "3. "), and `pairs` is ``[(number, answer), ...]``
    with each answer running up to the next entry. HTML tags (table cells)
    become spaces and answer-blank runs are dropped before matching.
    """
    text = _BLANK_RUN_RE.sub(" ", _TAG_RE.sub(" ", line)).strip()
    matches = list(_ANSWER_ENTRY_RE.finditer(text))
    if not matches:
        return text, []
    leading = text[: matches[0].start()].strip()
    pairs = []
    for m, nxt in zip(matches, matches[1:] + [None]):
        end = nxt.start() if nxt is not None else len(text)
        pairs.append((int(m.group(1)), text[m.end() : end].strip()))
    return leading, pairs


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
    ignored_test_substrings: tuple[str, ...] = ()

    # --- Discovery -------------------------------------------------------
    def discover_tests(self, data_dir):
        """Return the tests found under `data_dir` (default: one per top-level PDF)."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = [Test(id=p.stem, source=p) for p in sorted(root.glob("*.pdf"))]
        return tests

    def ignore_test(self, test: Test) -> bool:
        """Return True when a discovered test should never be processed.

        The default supports declarative, case-insensitive substring exclusions
        against the stable test id. Override this hook when a series needs more
        specific logic than ``ignored_test_substrings`` can express.
        """
        test_id = test.id.casefold()
        return any(part.casefold() in test_id for part in self.ignored_test_substrings)

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

    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        """Per-page cleanup applied before statement problem tagging.

        This is the statement-side counterpart of ``clean_solution_markdown``:
        a series may remove running headers, answer-form furniture, or OCR
        wrapper markup that would otherwise be mistaken for problem content or
        influence the strictly-increasing marker guard. The raw OCR remains in
        the cache; only the text handed to ``parse_layout`` is cleaned.
        """
        return markdown

    def validate_statement_markdown(self, page_index: int, markdown: str) -> bool:
        """Return whether a statement page OCR is complete enough to cache/use.

        The default accepts every non-runaway transcription. A born-digital
        series may compare it with cheap page metadata recorded by
        ``test_pages`` and force the OCR retry ladder when a clean-looking
        response silently omitted problems.
        """
        return True

    # --- Numbering quirks ------------------------------------------------
    def match_marker(self):
        """Return a series-specific marker matcher, or None to use the default.

        A matcher is a callable ``text -> (problem_number, match_end) | None``.
        """
        return None

    def postprocess(self, problems):
        """Hook to clean up the merged problem list. Default: unchanged."""
        return problems

    def duplicate_scope(self, test_id: str, across: bool = False):
        """Return a comparison-scope key for `dedup`, or None to opt out.

        The `dedup` command only compares problem statements whose tests share a
        scope. The default None disables duplicate detection for the series --
        conservative and series-agnostic, like `layout_options`. A series that
        reuses problems across sibling tests overrides this to bucket them:
        PUMaC returns the year, so its A/B divisions (and subject rounds) of a
        given year are compared against each other but not across years.

        `across` is the command's ``--across-years``: an opted-in series should
        widen its bucket (PUMaC drops the year so every test is comparable),
        while None stays None either way.
        """
        return None

    def layout_options(self):
        """Return the nanonets layout/figure heuristics for this series.

        The base defaults (see `config.LayoutOptions`) are the conservative,
        series-agnostic behavior: no page-spanning Picture filter, no gap-based
        problem-start fallback, and tables kept verbatim. Override to opt into
        layout-specific heuristics (MATHCOUNTS does).
        """
        return config.LayoutOptions()

    # --- Solutions -------------------------------------------------------
    def solution_source(self, test: Test):
        """Return the solution source (PDF/folder) for `test`, or None."""
        return None

    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        """Per-page cleanup of solution-OCR markdown before problem tagging.

        Applied to each page both when the pipeline assigns figures to problems
        and to the text handed to `parse_solutions` -- but never to what
        `parse_answers` sees, so a series can strip content that would corrupt
        problem numbering (Mandelbrot's out-of-order answer-key box) while its
        answer parser still reads it from the raw markdown. Default: unchanged.
        """
        return markdown

    def solution_figure_floor(self, pdf_page, image):
        """Rendered y below which a solution page's figures are furniture, or None.

        The figure-side partner of `clean_solution_markdown`: that hook strips a
        back-cover credits box / colophon from the *text*, but DETR still crops
        the same box as a Picture and binds it to the last problem. Returning a y
        (in the rendered image's coordinates) makes the pipeline drop any Picture
        whose vertical centre falls below it. `pdf_page` is the born-digital PDF
        page (so the boundary can be read from the exact text layer) and `image`
        the rendered page. Default None keeps every figure.
        """
        return None

    def parse_solutions(self, full_text: str) -> dict:
        """Segment the whole-test solution OCR into {problem_number: text}.

        `full_text` is every solution page's markdown (after
        `clean_solution_markdown`) concatenated in reading order. The default
        splits on this series' problem markers (reusing the nanonets layout
        splitter) and joins each problem's text across page breaks. Override to
        implement a series-specific solution layout; the problem numbers must
        agree with the pipeline's own marker tagging, which assigns the DETR
        figure crops (see pipeline.process_solution_document).
        """
        from ..nanonets import FIGURE_PLACEHOLDER, parse_layout

        opts = self.layout_options()
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(
            full_text,
            self.match_marker(),
            split_marker_table_rows=opts.split_marker_table_rows,
            point_value_list_markers=opts.point_value_list_markers,
            strict_section_restarts=opts.strict_section_restarts,
        ):
            if item["problem"] is None:
                continue
            if item["kind"] == "text":
                grouped.setdefault(item["problem"], []).append(item["text"])
            elif item["kind"] == "image":
                # Keep the figure's reading-order position as a sentinel; the
                # pipeline later aligns these with DETR's crops (see
                # pipeline.inline_solution_figures).
                grouped.setdefault(item["problem"], []).append(FIGURE_PLACEHOLDER)
        return {n: "\n".join(parts) for n, parts in grouped.items()}

    def postprocess_solutions(self, solutions: dict, statements: dict) -> dict:
        """Final series-specific cleanup with parsed statements available.

        Most series can segment worked solutions using printed labels alone.
        A series whose solution PDFs inconsistently omit those labels may use
        the authoritative parsed statement map to remove a verbatim restatement
        without teaching the shared pipeline any competition-specific prose.
        """
        return solutions

    def solution_index_marker(self, text: str):
        """Return a solution index from a solution-block heading, or None.

        Used only for naming solution figure crops. Series with nonstandard
        worked-solution headings can override this while keeping OCR/layout code
        series-agnostic.
        """
        solution = None
        for line in text.splitlines():
            m = _SOLUTION_INDEX_RE.match(line.strip())
            if m is not None:
                solution = int(m.group(1))
        return solution

    # --- Answers ----------------------------------------------------------
    def scrape_answers(self, test: Test) -> dict:
        """Return {problem_number: answer} for `test` without any OCR, or {}.

        For series whose key is already machine-readable (Purple Comet's
        pre-scraped ``answers.txt``). The `solutions` command tries this first;
        when it returns {}, it falls back to OCR-ing `answer_source` and calling
        `parse_answers`. Answers are written to ``problem_answer.json``.
        """
        return {}

    def answer_source(self, test: Test):
        """Return the document (PDF/folder) holding `test`'s answer key, or None.

        May be the same file as `solution_source` (Mandelbrot prints its key in
        a box at the top of the solutions PDF); the `solutions` command reuses
        that document's OCR instead of running it twice.
        """
        return None

    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Turn the answer document's per-page OCR into {problem_number: answer}.

        `pages_markdown` is one *raw* markdown string per rendered page of
        `answer_source` (uncleaned -- see `clean_solution_markdown`), so a
        parser can select its pages (Mathcounts finds its round's pages by
        their header) or its region (Mandelbrot's "Answer Key" box) itself.
        Called only when `answer_source` returned a document. Default: {}.
        """
        return {}


def _safe_name(value: str, label: str) -> str:
    """Validate a name that will become one output-path component."""
    if not _SAFE_NAME_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"invalid {label} {value!r}; use letters, numbers, '.', '_', or '-' "
            "and start with a letter or number"
        )
    return value


class GenericSeries(Series):
    """Runtime-configured series using the conservative base behavior."""

    def __init__(self, name: str):
        self.name = _safe_name(name, "series name")

    def discover_source(self, source, test_name=None):
        """Discover one PDF/image-folder test or a directory batch of PDFs."""
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"source not found: {src}")

        if src.is_file():
            if src.suffix.lower() != ".pdf":
                raise ValueError(f"unsupported source file (expected PDF): {src}")
            name = _safe_name(test_name or src.stem, "test name")
            return [Test(id=name, source=src)]

        pdfs = sorted(
            p
            for p in src.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
        images = _natural_pages(src)
        if pdfs and images:
            raise ValueError(
                f"mixed PDF and page-image directory is ambiguous: {src}"
            )
        if images:
            name = _safe_name(test_name or src.name, "test name")
            return [Test(id=name, source=src)]
        if pdfs:
            if test_name is not None:
                raise ValueError("--test-name cannot be used with a batch directory")
            return [Test(id=_safe_name(p.stem, "test name"), source=p) for p in pdfs]
        raise ValueError(f"no top-level PDFs or page images found in: {src}")
