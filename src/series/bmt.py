"""BMT (Berkeley Math Tournament) tests and solutions.

On-disk layout (data dir is ``BMT/out``)::

    out/<tournament>/<year>/<subject>/test.pdf
    out/<tournament>/<year>/<subject>/solutions.pdf

``<tournament>`` is ``bmt``, ``bmmt``, or ``bmmt-online``. Test IDs mirror their
path joined by underscores: ``bmt_2024_algebra``, ``bmmt_2018_speed``.

The ``power``, ``relay``, and ``puzzle`` rounds (including puzzle variants like
``puzzle-us-iran``) are skipped by `discover_tests` -- rounds we don't parse.
Every other round numbers its problems plainly (``1.``, ``2.``, ...), so the default
marker matcher fits. Each solutions PDF restates a problem, prints its final
answer on an ``Answer:`` line, then gives the worked solution under a
``Solution:`` label -- the same shape as HMMT, so `parse_answers` reads the
``Answer:`` value and `parse_solutions` keeps only the text from ``Solution:``
onward.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test, numbered_answers_in_line
from .smt import _boxed_answer

# Labels arrive markdown-emphasized from the OCR ("**Answer:** 45",
# "**Answer: 89**", "**Solution:** ..."), so each pattern tolerates leading
# emphasis/heading chars and (for the label itself) a "**" between the word and
# its colon. The captured value is cleaned by _clean_answer.
_EMPH = r"[*_#]{0,3}"
_ANSWER_RE = re.compile(rf"^\s*{_EMPH}\s*(?:Answer|Ans)\b\s*{_EMPH}\s*:?\s*(.*)$", re.IGNORECASE)
_PROPOSED_RE = re.compile(rf"^\s*{_EMPH}\s*Proposed\s+by\b", re.IGNORECASE)
_SOLUTION_RE = re.compile(
    rf"^\s*{_EMPH}\s*Solution(?:\s+\d+)?\b\s*{_EMPH}\s*:?\s*(.*)$", re.IGNORECASE
)

# Rounds we don't parse: power (proof-based team round) and the relay/puzzle
# rounds, whose formats don't fit the per-problem pipeline. Matched on the
# subject folder name, including puzzle variants ("puzzle-us-iran").
_SKIP_SUBJECTS = {"power", "relay"}


def _skip_subject(subject: str) -> bool:
    return subject in _SKIP_SUBJECTS or subject.startswith("puzzle")


def _clean_answer(value: str) -> str:
    """Strip markdown emphasis, surrounding ``$`` math delimiters, and a trailing
    period from a captured answer value ("** 6072", "89**", "$\\frac{2}{3}$")."""
    value = re.sub(r"^[*_\s]+|[*_\s]+$", "", value)
    if len(value) > 1 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value.rstrip(".").strip()


class BmtSeries(Series):
    name = "bmt"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Discover every BMT ``test.pdf`` recursively, minus the skipped rounds.

        The full parent path forms the ID, avoiding collisions across the three
        tournaments (``bmt``, ``bmmt``, ``bmmt-online``) and their many rounds.
        The power/relay/puzzle rounds are excluded (see `_skip_subject`).
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(id="_".join(pdf.relative_to(root).parts[:-1]), source=pdf)
            for pdf in sorted(root.glob("**/test.pdf"))
            if not _skip_subject(pdf.parent.name)
        ]

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

        Figure placeholders are retained so DETR crops stay inline.
        """
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
            grouped.setdefault(item["problem"], []).append(value)
        return {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Extract answers from HTML tables, explicit Answer/Ans lines, boxed values, or numbered lists."""
        full_text = "\n\n".join(pages_markdown)
        answers = {}

        # 1. HTML answer key tables (e.g., Speed Round answer keys)
        td_matches = re.findall(
            r"<td>\s*(\d+)\s*</td>\s*<td>\s*(.*?)\s*</td>", full_text, re.I | re.S
        )
        if td_matches:
            for q_str, ans_str in td_matches:
                num = int(q_str)
                clean_ans = _clean_answer(ans_str)
                if clean_ans:
                    answers[num] = clean_ans
            if len(answers) >= 5:
                return answers
            answers = {}

        # 2. Problem blocks via parse_layout
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None or item["kind"] != "text":
                continue
            answer = _answer_value(item["text"])
            if answer and answer.upper() != "N/A":
                answers[item["problem"]] = answer

        # 3. Line-by-line numbered lists (standalone answer key documents)
        if not answers:
            for line in full_text.splitlines():
                leading, pairs = numbered_answers_in_line(line)
                for num, ans_str in pairs:
                    clean_ans = _clean_answer(ans_str)
                    if clean_ans:
                        answers[num] = clean_ans

        return answers


def _answer_value(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is None:
            continue
        inline = _clean_answer(match.group(1))
        if inline:
            return inline
        # Value carried onto the following non-blank line, before any label.
        for following in lines[index + 1 :]:
            if _PROPOSED_RE.match(following) or _SOLUTION_RE.match(following):
                break
            if following.strip():
                return _clean_answer(following)
        return ""
    return _boxed_answer(block) or ""


def _solution_body(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_RE.match(line)
        if match is not None:
            # Drop a "**" emphasis closer the label's colon left behind.
            first = re.sub(r"^[*_]+\s*", "", match.group(1).strip())
            kept = ([first] if first else []) + lines[index + 1 :]
            return "\n".join(kept).strip()
    # No Solution label (unexpected layout / OCR miss): keep the block minus the
    # answer and proposer furniture so nothing is silently dropped.
    return "\n".join(
        l for l in lines if not (_ANSWER_RE.match(l) or _PROPOSED_RE.match(l))
    ).strip()
