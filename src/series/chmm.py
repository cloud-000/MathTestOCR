"""Caltech Math Meet / Caltech-Harvey Mudd Math Competition.

On-disk layout (data dir is ``CHMM/out``)::

    out/<year>/<season>/<round>/test.pdf
    out/<year>/<season>/<round>/solutions.pdf

Test IDs mirror the directory path, joined by underscores, for example
``2025_annual_individual`` and ``2018_fall_team``.  Power rounds are omitted:
their section/definition numbering is hierarchical and does not fit the
pipeline's one-flat-problem-per-number output.  Other proof-style rounds are
kept when their top-level questions are numbered normally.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test
from .smt import _boxed_answer


# Fall 2012 prefixes each round's problems (IR1, MR2, TR3, TBR4).  The 2015
# packets instead print "Problem 0.1", "Problem 0.2", ...; the leading zero is
# a section number, not part of the stable problem number.
_PREFIXED_MARKER_RE = re.compile(r"^\s*(?:IR|MR|TR|TBR)\s*(\d+)\s*[.)]?", re.I)
_ZERO_SECTION_RE = re.compile(r"^\s*Problem\s+0[.](\d+)\s*[.)]?", re.I)
# Several solution-only packets use "Solution 1." as the sole block marker.
_SOLUTION_MARKER_RE = re.compile(r"^\s*Solution\s+(\d+)\s*[.:)]?", re.I)

_SOLUTION_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Solution(?:\s+\d+)?\s*[*_]{0,2}\s*[.:]?\s*(.*)$",
    re.I,
)
_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Answer\s*[*_]{0,2}\s*[.:]?\s*(.*)$", re.I
)

# Numbered rules on modern cover pages share a page with problem 1.  Rejecting
# these known rule sentences in the matcher avoids turning the real problem 1
# into problem 8 after parse_layout's section-restart offset.
_RULE_START_RE = re.compile(
    r"^(?:"
    r"you have \d+ minutes|"
    r"do not (?:look|flip|turn)|"
    r"this test consists|"
    r"no (?:collaboration|computational aids)|"
    r"you (?:are,?|may|can) (?:however, )?(?:permitted|collaborate|message)|"
    r"you may not collaborate|"
    r"congratulations for scoring|"
    r"there are \d+ questions|"
    r"the top \d+|"
    r"the time limit|"
    r"on the back side|"
    r"write (?:your|answers|legibly)|"
    r"all (?:the )?answers (?:are|must|should)|"
    r"answers (?:are|must|should)|"
    r"if you believe|"
    r"for multi-part problems"
    r")\b",
    re.I,
)


def _match_marker(text: str):
    for pattern in (_PREFIXED_MARKER_RE, _ZERO_SECTION_RE, _SOLUTION_MARKER_RE):
        match = pattern.match(text)
        if match is not None:
            return int(match.group(1)), match.end()
    result = anchors._match_marker(text)
    if result is None:
        return None
    if result[0] > 100:
        return None
    if _RULE_START_RE.match(text[result[1] :].strip()):
        return None
    return result


class ChmmSeries(Series):
    name = "chmm"
    has_solutions = True
    has_answers = True
    ignored_test_substrings = ("math-talk", "tcs")

    @override
    def discover_tests(self, data_dir):
        """Discover flat-numbered rounds and omit hierarchical power packets."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(id="_".join(pdf.relative_to(root).parts[:-1]), source=pdf)
            for pdf in sorted(root.glob("**/test.pdf"))
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
        if solution.exists():
            return solution
        season = test.source.parent.parent
        if season.is_dir():
            candidates = [
                p
                for p in sorted(season.glob("**/solutions.pdf"))
                if p.parent.name != "power"
            ]
            if candidates:
                return candidates[0]
        return None

    @override
    def answer_source(self, test: Test):
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        if test is not None:
            full_text = _filter_round_text(full_text, test.id)
        grouped = _group_blocks(full_text)
        # A few early files are short answer keys, not worked solutions.  They
        # belong in problem_answer.json only.
        if _is_answer_key(full_text):
            return {}
        return {number: _solution_body(block) for number, block in grouped.items()}

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        filtered_pages = [_filter_round_text(md, test.id) for md in pages_markdown]
        full_text = "\n\n".join(filtered_pages)
        grouped = _group_blocks(full_text)
        answer_key = _is_answer_key(full_text)
        answers = {}
        for number, block in grouped.items():
            value = _answer_value(block)
            if not value and answer_key:
                value = _clean_value(block)
            if value:
                answers[number] = value
        return answers


def _filter_round_text(full_text: str, test_id: str) -> str:
    """If full_text contains multi-round section headers, filter to test_id's round."""
    round_keywords = ("individual", "team", "mixer", "tiebreaker")
    target = next((r for r in round_keywords if r in test_id.lower()), None)
    if not target:
        return full_text

    header_pattern = re.compile(
        r"(?im)^\s*(?:#+\s*|\*{1,2}\s*)?(?:Fall|Spring|Winter|Annual)?\s*(?:\d{4})?\s*"
        r"(?:Caltech[- ]Harvey Mudd Math Competition)?\s*"
        r"(Individual|Team|Mixer|Tiebreaker|Power)\s+(?:Round\s*)?(?:Solutions|Answers|Round)?\s*$"
    )
    matches = list(header_pattern.finditer(full_text))
    if not matches:
        return full_text

    sections = []
    for i, m in enumerate(matches):
        r_name = m.group(1).lower()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections.append((r_name, full_text[start:end]))

    matching_text = [sec_text for r_name, sec_text in sections if target in r_name]
    return "\n\n".join(matching_text) if matching_text else full_text


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
        re.search(
            r"\b(?:Individual|Team|Tiebreaker|Mixer)\s+(?:Answers|Round Solutions)\b",
            head,
            re.I,
        )
        and not re.search(r"(?im)^\s*Solution[.:]?\b", full_text)
    )


def _solution_body(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_LINE_RE.match(line)
        if match is not None:
            first = match.group(1).lstrip("*_ ").strip()
            return "\n".join(([first] if first else []) + lines[index + 1 :]).strip()
    # "Solution N." can itself be the marker and is removed by parse_layout.
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
    return _boxed_answer(block)


def _clean_value(value: str) -> str:
    value = value.strip().strip("*_").strip()
    if value == FIGURE_PLACEHOLDER or value.lower().startswith("proposed by"):
        return ""
    return value.rstrip(".").strip()
