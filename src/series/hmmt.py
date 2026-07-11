"""Harvard-MIT Mathematics Tournament tests and solutions.

On-disk layout (data dir is ``HMMT/out``)::

    out/<year>/<feb|nov>/<round>/test.pdf
    out/<year>/<feb|nov>/<round>/solutions.pdf
    out/<year>/hmic/test.pdf
    out/<year>/hmic/solutions.pdf

Test IDs mirror their path, joined by underscores: ``2017_feb_algnt`` and
``2013_hmic``. Most problems use ordinary integer markers (``1. [5] ...``);
older Guts and Oral documents use ``Problem Gu1`` and ``Problem O1``.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test


_ANSWER_RE = re.compile(r"^\s*Answer\s*:\s*(.*)$", re.IGNORECASE)
_PROPOSED_RE = re.compile(r"^\s*Proposed\s+by\s*:", re.IGNORECASE)
_SOLUTION_RE = re.compile(r"^\s*Solution(?:\s+\d+)?\s*:?\s*(.*)$", re.IGNORECASE)
_PREFIXED_MARKER_RE = re.compile(r"^\s*Problem\s+(?:Gu|O)(\d+)\b", re.IGNORECASE)


def _match_marker(text):
    match = _PREFIXED_MARKER_RE.match(text)
    if match is not None:
        return int(match.group(1)), match.end()
    return anchors._match_marker(text)


class HmmtSeries(Series):
    name = "hmmt"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Discover every HMMT ``test.pdf`` recursively.

        The full parent path forms the ID, avoiding collisions between the
        February, November, and HMIC collections and their many round types.
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(
                id="_".join(pdf.relative_to(root).parts[:-1]),
                source=pdf,
            )
            for pdf in sorted(root.glob("**/test.pdf"))
        ]

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        """Keep statement and solution figures at their reading-order position."""
        return config.LayoutOptions(inline_figures=True)

    @override
    def solution_source(self, test):
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def answer_source(self, test):
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text):
        """Drop each restated statement and keep only its worked solution.

        Older HMIC documents put the solution directly after ``Answer:``;
        newer documents add ``Proposed by:`` and ``Solution:`` labels. Figure
        placeholders are retained so DETR crops remain inline.
        """
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
            grouped.setdefault(item["problem"], []).append(value)
        return {number: _solution_body("\n".join(parts)) for number, parts in grouped.items()}

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Extract the explicit ``Answer:`` value from each problem block."""
        answers = {}
        for item in parse_layout("\n\n".join(pages_markdown), self.match_marker()):
            if item["problem"] is None or item["kind"] != "text":
                continue
            answer = _answer_value(item["text"])
            if answer and answer.upper() != "N/A":
                answers[item["problem"]] = answer
        return answers


def _answer_value(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is None:
            continue
        inline = match.group(1).strip()
        if inline:
            return inline
        for following in lines[index + 1 :]:
            if _PROPOSED_RE.match(following) or _SOLUTION_RE.match(following):
                break
            if following.strip():
                return following.strip()
        return ""
    return ""


def _solution_body(block: str) -> str:
    lines = block.splitlines()

    # Modern PDFs explicitly mark the start of each worked solution.
    for index, line in enumerate(lines):
        match = _SOLUTION_RE.match(line)
        if match is not None:
            first = match.group(1).strip()
            kept = ([first] if first else []) + lines[index + 1 :]
            return "\n".join(kept).strip()

    # Older PDFs have no Solution label; their solution begins after Answer.
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is not None:
            start = index + 1
            # When the value is on the next line, omit that line too. If the
            # value is inline, the proof begins immediately on the next line.
            if not match.group(1).strip():
                while start < len(lines) and not lines[start].strip():
                    start += 1
                start += 1
            return "\n".join(lines[start:]).strip()

    # Preserve unexpected layouts rather than silently dropping their text.
    return "\n".join(line for line in lines if not _PROPOSED_RE.match(line)).strip()
