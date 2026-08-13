"""Online Math Open (OMO) tests, solutions, and answer keys.

On-disk layout (data dir is ``OMO`` or ``OMO/out``)::

    out/<year>/<season>/problems.pdf
    out/<year>/<season>/solutions.pdf
    out/<year>/<season>/answers.txt

Test IDs mirror their directory path, joined by underscores: ``2018_spring``,
``2012_fall``. Most tests include a pre-scraped ``answers.txt``; for tests without
one, answers are extracted from ``solutions.pdf``.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test, strip_solution_page_furniture


_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:\*{1,2}|#+\s*|\\textbf\{)?\s*Answer[.:]?"
    r"\s*(?:\*{1,2}|_+|\})?\s*(.*)$",
    re.IGNORECASE,
)
_SOLUTION_LINE_RE = re.compile(
    r"^\s*(?:\*{1,2}|#+\s*|\\textbf\{)?\s*Solution[.:]?"
    r"\s*(?:\*{1,2}|_+|\})?\s*(.*)$",
    re.IGNORECASE,
)
_THE_ANSWER_IS_RE = re.compile(
    r"\bThe\s+answer\s+is\s+([A-Za-z0-9_/\-+\\.\{\}\(\)]+)", re.IGNORECASE
)
_PROPOSED_RE = re.compile(r"^\s*Proposed\s+by\b", re.IGNORECASE)
_OMO_RUNNING_FURNITURE_RE = re.compile(
    r"^(?:"
    r"(?:OMO|online\s+math\s+open)\b.*"
    r"|official\s+solutions?"
    r"|(?:fall|spring|winter)\s+\d{4}"
    r"|(?:fall|spring|winter)\s+OMO\b.*"
    r"|page\s+\d+"
    r")$",
    re.IGNORECASE,
)
_OMO_DATE_FURNITURE_RE = re.compile(
    r"^(?:"
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
    r")(?:/(?:"
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
    r"))?\s+(?:"
    r"\d{1,2}\s*[-–—]\s*(?:(?:"
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
    r")\s+)?\d{1,2},?\s+\d{4}"
    r"|\d{4}(?:\s+(?:(?:fall|spring|winter)\s+)?OMO\b.*)?"
    r")$",
    re.IGNORECASE,
)


def _boxed_answer(block: str) -> str:
    from .smt import _boxed_answer as boxed

    return boxed(block) or ""


class OmoSeries(Series):
    name = "omo"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Discover every OMO test (problems.pdf) under data_dir."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        search_root = root / "out" if (root / "out").is_dir() else root

        tests = []
        for pdf in sorted(search_root.glob("**/problems.pdf")):
            rel_parts = pdf.relative_to(search_root).parts[:-1]
            test_id = "_".join(rel_parts)
            tests.append(Test(id=test_id, source=pdf))
        return tests

    @override
    def skip_page(self, text: str) -> bool:
        """Skip title cover pages, acknowledgements, and contest information."""
        txt = text.strip()
        if not txt:
            return False
        txt_lower = txt.lower()
        if "acknowledgements" in txt_lower or "acknowledgments" in txt_lower:
            return True
        if "contest information" in txt_lower or "team guidelines" in txt_lower:
            return True
        if (
            "the online math open" in txt_lower
            and not any(re.search(r"^\s*1\.\s+", line) for line in txt.splitlines())
            and len(txt) < 400
        ):
            return True
        return False

    @override
    def layout_options(self):
        """Keep statement and solution figures at their reading-order position."""
        return config.LayoutOptions(
            inline_figures=True,
            solution_answer_box_filter=True,
        )

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        """Drop OMO's repeated edition/date masthead before page carry."""
        return strip_solution_page_furniture(
            markdown,
            line_patterns=(
                _OMO_RUNNING_FURNITURE_RE,
                _OMO_DATE_FURNITURE_RE,
            ),
        )

    @override
    def solution_source(self, test: Test):
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def answer_source(self, test: Test):
        return self.solution_source(test)

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        """Drop OMO's repeated edition/"Official Solutions" page furniture."""
        return strip_solution_page_furniture(
            markdown, line_patterns=(_OMO_RUNNING_FURNITURE_RE,)
        )

    @override
    def scrape_answers(self, test: Test) -> dict:
        """Read sibling ``answers.txt`` if present into {problem_number: answer}."""
        answers_file = test.source.parent / "answers.txt"
        if not answers_file.exists():
            return {}
        answers = {}
        for line in answers_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                answers[int(parts[0])] = parts[1]
        return answers

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        """Segment solution document OCR into {problem_number: text}."""
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
            grouped.setdefault(item["problem"], []).append(value)
        return {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Extract answer key from solution document OCR for tests without answers.txt."""
        answers: dict[int, str] = {}
        kept_pages = [page for page in pages_markdown if not self.skip_page(page)]
        full_text = "\n\n".join(kept_pages if kept_pages else pages_markdown)
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None or item["kind"] != "text":
                continue
            grouped.setdefault(item["problem"], []).append(item["text"])

        for n, parts in grouped.items():
            ans = _extract_answer_from_block("\n".join(parts))
            if ans:
                answers[n] = ans
        return answers


def _solution_body(block: str) -> str:
    """Extract worked solution body, dropping restated statement/author/answer headers."""
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_LINE_RE.match(line)
        if match is not None:
            first = re.sub(r"^[*_]+\s*", "", match.group(1).strip())
            kept = ([first] if first else []) + lines[index + 1 :]
            return "\n".join(kept).strip()
    return block.strip()


def _extract_answer_from_block(block: str) -> str:
    """Extract answer string from a solution block."""
    lines = block.splitlines()
    for idx, line in enumerate(lines):
        m = _ANSWER_LINE_RE.match(line)
        if m:
            val = m.group(1).strip()
            if val and not _PROPOSED_RE.match(val) and not _SOLUTION_LINE_RE.match(val):
                return _clean_answer_value(val)
            for following in lines[idx + 1 :]:
                f_str = following.strip()
                if not f_str:
                    continue
                if _PROPOSED_RE.match(f_str) or _SOLUTION_LINE_RE.match(f_str):
                    break
                return _clean_answer_value(f_str)
    m2 = _THE_ANSWER_IS_RE.search(block)
    if m2:
        return _clean_answer_value(m2.group(1))
    return _boxed_answer(block)


def _clean_answer_value(value: str) -> str:
    """Remove sentence punctuation while preserving balanced inline math."""
    return value.strip().rstrip(".").strip()
