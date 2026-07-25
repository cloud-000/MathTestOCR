"""Carnegie Mellon Informatics and Mathematics Competition.

On-disk layout (data dir may be ``CMIMC`` or ``CMIMC/out``)::

    out/<year>/<division>/<subject>/test.pdf
    out/<year>/<division>/<subject>/solutions.pdf

Test IDs mirror the path below ``out``, joined by underscores, such as
``2025_individual_algebra`` and ``2022_team_team``.  The older power round is
omitted because its hierarchical proof packet does not have a stable flat
problem numbering scheme; the newer three-problem computer-science proof round
is flat and remains discoverable.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test
from .smt import _boxed_answer


_SOLUTION_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Solution(?:\s+\d+)?\s*[*_]{0,2}\s*[.:]?\s*(.*)$",
    re.I,
)
_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Answer\s*[*_]{0,2}\s*[.:]?\s*(.*)$", re.I
)
_THE_ANSWER_IS_RE = re.compile(r"\b(?:the\s+)?answer\s+is\s+(.+?)(?:[.!]\s|$)", re.I)

# CMIMC puts its numbered instructions on the first problem page.  They are
# intentionally filtered here rather than with skip_page, which would also
# discard the first several real problems.
_RULE_START_RE = re.compile(
    r"^(?:"
    r"do not look|"
    r"this test consists|"
    r"write (?:your|answers|legibly)|"
    r"no computational aids|"
    r"all (?:the )?answers (?:are|must|should)|"
    r"answers (?:are|must|should)|"
    r"if you believe|"
    r"in your solution|"
    r"problems are not ordered|"
    r"you have \d+ minutes|"
    r"during the test|"
    r"submit your answers|"
    r"we use the following notation|"
    r"the following notation may be useful|"
    r"in the event of a tie"
    r")\b",
    re.I,
)


def _match_marker(text: str):
    result = anchors._match_marker(text)
    if result is None:
        return None
    if result[0] > 100:
        return None
    if _RULE_START_RE.match(text[result[1] :].strip()):
        return None
    return result


class CmimcSeries(Series):
    name = "cmimc"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        search_root = root / "out" if (root / "out").is_dir() else root
        return [
            Test(
                id="_".join(pdf.relative_to(search_root).parts[:-1]),
                source=pdf,
            )
            for pdf in sorted(search_root.glob("**/test.pdf"))
            if pdf.parent.name != "power"
        ]

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        return config.LayoutOptions(inline_figures=True)

    @override
    def solution_source(self, test: Test):
        solution = test.source.parent / "solutions.pdf"
        return solution if solution.exists() else None

    @override
    def answer_source(self, test: Test):
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text: str) -> dict:
        if _is_answer_key(full_text):
            return {}
        return {
            number: _solution_body(block)
            for number, block in _group_blocks(full_text).items()
        }

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        full_text = "\n\n".join(pages_markdown)
        answer_key = _is_answer_key(full_text)
        answers = {}
        for number, block in _group_blocks(full_text).items():
            value = _answer_value(block)
            if not value and answer_key:
                value = _clean_value(block)
            if value:
                answers[number] = value
        return answers


def _group_blocks(full_text: str) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(full_text, _match_marker):
        if item["problem"] is None:
            continue
        value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
        grouped.setdefault(item["problem"], []).append(value)
    return {number: "\n".join(parts).strip() for number, parts in grouped.items()}


def _is_answer_key(full_text: str) -> bool:
    head = "\n".join(full_text.splitlines()[:12])
    return bool(
        re.search(r"\bIntegration\s+Bee\s+Answers\b", head, re.I)
        and not re.search(r"(?im)^\s*Solution\b", full_text)
    )


def _solution_body(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_LINE_RE.match(line)
        if match is not None:
            first = match.group(1).lstrip("*_ ").strip()
            return "\n".join(([first] if first else []) + lines[index + 1 :]).strip()
    return block.strip()


def _answer_value(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_LINE_RE.match(line)
        if match is None:
            continue
        inline = _clean_value(match.group(1))
        if inline:
            return inline
        for following in lines[index + 1 :]:
            if _SOLUTION_LINE_RE.match(following):
                break
            value = _clean_value(following)
            if value:
                return value
        return ""
    prose = _THE_ANSWER_IS_RE.search(block)
    if prose is not None:
        return _clean_value(prose.group(1))
    return _boxed_answer(block)


def _clean_value(value: str) -> str:
    value = value.strip().strip("*_").strip()
    if value == FIGURE_PLACEHOLDER or value.lower().startswith("proposed by"):
        return ""
    return value.rstrip(".").strip()
