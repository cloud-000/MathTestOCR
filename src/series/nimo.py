"""NIMO contests collected in ``nimo_all_problems.pdf``.

The source is a 95-page compendium stored beside the OMO source tree.  It is
not one test: Part A contains 45 monthly, summer, April Fun, and Winter
Olympiad contests, while Part B contains short-answer keys and the Winter
Olympiad worked solutions.  Discovery reads the born-digital headings and
records each contest's exact source-page ranges before any OCR client starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from typing_extensions import override

from .. import anchors, config, pdf_io
from .base import ProofProfile, Series, Test, strip_solution_page_furniture


_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
_MONTHLY_TITLE_RE = re.compile(
    rf"(?m)^(\d+)\.\s+(?P<title>(?:{_MONTHS})\s+\d{{1,2}},\s+"
    rf"(?P<year>\d{{4}}))\s*$"
)
_SUMMER_TITLE_RE = re.compile(
    r"(?m)^(\d+)\.\s+(?P<title>Summer\s+(?P<year>\d{4}))\s*$"
)
_APRIL_TITLE_RE = re.compile(
    r"(?m)^(\d+)\.\s+(?P<title>April\s+(?P<year>\d{4}))\s*$"
)
_WINTER_TITLE_RE = re.compile(
    r"(?m)^(\d+)\.\s+(?P<title>Winter Olympiad\s+(?P<year>\d{4}))\s*$"
)
_TITLE_RES = {
    "monthly": _MONTHLY_TITLE_RE,
    "summer": _SUMMER_TITLE_RE,
    "april_fun": _APRIL_TITLE_RE,
    "winter_olympiad": _WINTER_TITLE_RE,
}
_STATEMENT_SECTION_MARKERS = {
    "I. Monthly Contest": "monthly",
    "II. Summer Contest": "summer",
    "III. April Fun Round": "april_fun",
    "IV. Winter Olympiad": "winter_olympiad",
}
_ANSWER_SECTION_MARKERS = {
    "V. Monthly Contest": "monthly",
    "VI. Summer Contest": "summer",
    "VII. April Fun Round": "april_fun",
    "VIII. Winter Olympiad": "winter_olympiad",
}

_RUNNING_FURNITURE_RE = re.compile(
    r"^(?:The NIMO Compendium|"
    r"(?:I|II|III|IV|V|VI|VII|VIII)\.\s*"
    r"(?:Monthly Contest|Summer Contest|April Fun Round|Winter Olympiad)|"
    r"\d+)$",
    re.IGNORECASE,
)
_CONTEST_HEADING_RE = re.compile(
    rf"^\d+\.\s+(?:(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}|"
    r"Summer\s+\d{4}|April\s+\d{4}|Winter Olympiad\s+\d{4})$",
    re.IGNORECASE,
)
_DATE_OR_TIME_RE = re.compile(
    rf"^(?:(?:{_MONTHS})\s+\d{{4}}|"
    r"\d{1,2}:\d{2}\s+[AP]M\s*[-–—]\s*\d{1,2}:\d{2}\s+[AP]M\s+ET)$",
    re.IGNORECASE,
)
_ANSWER_PAIR_RE = re.compile(
    r"\((\d+)\)\s*(.*?)(?=\s*\(\d+\)\s*|\Z)", re.DOTALL
)
_SOLUTION_START_RE = re.compile(
    r"(?m)^\s*(?:\*{1,2})?Solution(?:\s+\d+)?\.?(?:\*{1,2})?\s*"
)
_AUTHOR_RE = re.compile(r"^\s*(?:[*_]+\s*)?\(([^)]{1,80})\)")


@dataclass
class _NimoRecord:
    test: Test
    category: str
    ordinal: int
    title: str
    statement_pages: list[int] = field(default_factory=list)
    answer_pages: list[int] = field(default_factory=list)
    solution_pages: list[int] = field(default_factory=list)


def _section(text: str, markers: dict[str, str]) -> str | None:
    for marker, category in markers.items():
        if marker in text:
            return category
    return None


def _heading(text: str, category: str):
    match = _TITLE_RES[category].search(text)
    if match is None:
        return None
    return int(match.group(1)), match.group("title"), int(match.group("year"))


def _test_id(category: str, title: str, year: int) -> str:
    if category == "monthly":
        month = re.match(r"[A-Za-z]+", title).group(0).lower()
        return f"{year}_{month}"
    if category == "summer":
        return f"{year}_summer"
    if category == "april_fun":
        return f"{year}_april_fun"
    return f"{year}_winter_olympiad"


def _clean_page(markdown: str) -> str:
    return strip_solution_page_furniture(
        markdown,
        line_patterns=(
            _RUNNING_FURNITURE_RE,
            _CONTEST_HEADING_RE,
            _DATE_OR_TIME_RE,
        ),
    )


class NimoSeries(Series):
    name = "nimo"
    has_solutions = True
    has_answers = True
    proof_test_patterns = (r"^\d{4}_winter_olympiad$",)

    def __init__(self):
        self._records: dict[str, _NimoRecord] = {}

    @override
    def discover_tests(self, data_dir):
        root = Path(data_dir)
        candidates = (
            root / "nimo_all_problems.pdf",
            root / "out" / "nimo_all_problems.pdf",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"nimo_all_problems.pdf not found in {root} or {root / 'out'}"
            )

        doc = pymupdf.open(source)
        try:
            texts = [page.get_text("text") for page in doc]
        finally:
            doc.close()

        records: list[_NimoRecord] = []
        by_heading: dict[tuple[str, int], _NimoRecord] = {}
        active: _NimoRecord | None = None
        in_statements = False

        for page_index, text in enumerate(texts):
            if "Part A." in text and "Problems" in text:
                in_statements = True
                active = None
                continue
            if "Part B." in text and "Answers and Solutions" in text:
                break
            if not in_statements:
                continue
            category = _section(text[:400], _STATEMENT_SECTION_MARKERS)
            if category is None:
                if active is not None:
                    active.statement_pages.append(page_index)
                continue
            found = _heading(text[:700], category)
            if found is not None:
                ordinal, title, year = found
                base_id = _test_id(category, title, year)
                test_id = base_id
                suffix = 2
                while any(record.test.id == test_id for record in records):
                    test_id = f"{base_id}_{suffix}"
                    suffix += 1
                active = _NimoRecord(
                    test=Test(id=test_id, source=source),
                    category=category,
                    ordinal=ordinal,
                    title=title,
                )
                records.append(active)
                by_heading[(category, ordinal)] = active
            if active is not None:
                active.statement_pages.append(page_index)

        # Short-answer keys occupy pages 64-69; several contests share a page.
        # Record the page once per heading and parse its compact ``(N) answer``
        # line directly from the exact born-digital text layer.
        for page_index, text in enumerate(texts):
            category = _section(text[:300], _ANSWER_SECTION_MARKERS)
            if category not in {"monthly", "summer", "april_fun"}:
                continue
            for match in _TITLE_RES[category].finditer(text):
                record = by_heading.get((category, int(match.group(1))))
                if record is not None and page_index not in record.answer_pages:
                    record.answer_pages.append(page_index)

        # Only the proof-style Winter Olympiads have worked solutions. Their
        # page ranges run from one contest heading to the next through EOF.
        active = None
        for page_index, text in enumerate(texts):
            if "VIII. Winter Olympiad" not in text[:300]:
                continue
            found = _heading(text[:700], "winter_olympiad")
            if found is not None:
                active = by_heading.get(("winter_olympiad", found[0]))
            if active is not None:
                active.solution_pages.append(page_index)

        self._records = {record.test.id: record for record in records}
        return [record.test for record in records]

    def _record(self, test: Test) -> _NimoRecord:
        try:
            return self._records[test.id]
        except KeyError:
            # Direct hook use outside the CLI still gets deterministic discovery.
            self.discover_tests(Path(test.source).parent)
            return self._records[test.id]

    @override
    def prepare_cached_statement(self, test: Test):
        self._active_test_id = test.id

    @override
    def match_marker(self):
        record = self._records.get(getattr(self, "_active_test_id", ""))
        if record is None or record.category == "winter_olympiad":
            return None

        # Short-answer NIMO problems always print an author immediately after
        # the number. Requiring that signature prevents numbered subproblems,
        # proof steps, and large decimal literals from becoming new problems
        # (especially April 2014 problem 9's embedded 15-leg relay).
        def match(text: str):
            marker = anchors._match_marker(text)
            if marker is None:
                return None
            author = _AUTHOR_RE.match(text[marker[1] :])
            if author is None:
                return None
            byline = author.group(1)
            if not re.search(r"[A-Za-z]", byline) or re.search(r"[=+\\^\d]", byline):
                return None
            return marker

        return match

    @override
    def test_pages(self, test: Test, workdir):
        self._active_test_id = test.id
        record = self._record(test)
        return pdf_io.pdf_pages_to_images(
            record.test.source, workdir, record.statement_pages
        )

    @override
    def solution_pages(self, test: Test, source, workdir):
        self._active_test_id = test.id
        record = self._record(test)
        return pdf_io.pdf_pages_to_images(source, workdir, record.solution_pages)

    @override
    def layout_options(self):
        return config.LayoutOptions(
            inline_figures=True,
            consecutive_problem_markers=True,
            flat_problem_numbering=True,
        )

    @override
    def solution_match_marker(self):
        # Every Winter Olympiad has exactly eight problems. This also fences
        # numbered grading notes in the long compendium solutions from opening
        # a phantom ninth solution block.
        def match(text: str):
            marker = anchors._match_marker(text)
            return marker if marker is not None and marker[0] <= 8 else None

        return match

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        return _clean_page(markdown)

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        return _clean_page(markdown)

    @override
    def solution_source(self, test: Test):
        record = self._record(test)
        return record.test.source if record.solution_pages else None

    @override
    def scrape_answers(self, test: Test) -> dict:
        record = self._record(test)
        if not record.answer_pages:
            return {}

        doc = pymupdf.open(record.test.source)
        try:
            text = "\n".join(doc[index].get_text("text") for index in record.answer_pages)
        finally:
            doc.close()

        heading = re.compile(
            rf"(?m)^{record.ordinal}\.\s+{re.escape(record.title)}\s*$"
        ).search(text)
        if heading is None:
            return {}
        next_heading = _TITLE_RES[record.category].search(text, heading.end())
        block = _clean_page(
            text[heading.end() : next_heading.start() if next_heading else None]
        )

        answers = {}
        for match in _ANSWER_PAIR_RE.finditer(block):
            value = re.sub(r"\s+", " ", match.group(2)).strip()
            if value:
                answers[int(match.group(1))] = value
        return answers

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        solutions = super().parse_solutions(full_text, test=test)
        cleaned = {}
        for number, value in solutions.items():
            values = value if isinstance(value, list) else [value]
            bodies = []
            for candidate in values:
                start = _SOLUTION_START_RE.search(candidate)
                bodies.append(candidate[start.end() :].strip() if start else candidate)
            cleaned[number] = bodies if isinstance(value, list) else bodies[0]
        return cleaned

    @override
    def proof_profile(self, test: Test) -> ProofProfile | None:
        record = self._record(test)
        return ProofProfile() if record.category == "winter_olympiad" else None
