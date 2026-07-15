"""Harvard-MIT Mathematics Tournament tests and solutions.

On-disk layout (data dir is ``HMMT/out``)::

    out/<year>/<feb|nov>/<round>/test.pdf
    out/<year>/<feb|nov>/<round>/solutions.pdf
    out/<year>/hmic/test.pdf
    out/<year>/hmic/solutions.pdf

Test IDs mirror their path, joined by underscores: ``2017_feb_algnt`` and
``2013_hmic``. Most problems use ordinary integer markers (``1. [5] ...``);
older per-subject documents use a round-lettered marker instead --
``Problem A1`` (Algebra), ``C8`` (Calculus), ``G3`` (Geometry),
``T4``/``AT10`` (Team, Advanced Topics), ``Gu1`` (Guts), ``O1`` (Oral).
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, answer_llm, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test


_ANSWER_RE = re.compile(r"^\s*Answer\s*:\s*(.*)$", re.IGNORECASE)
_PROPOSED_RE = re.compile(r"^\s*Proposed\s+by\s*:", re.IGNORECASE)
_SOLUTION_RE = re.compile(r"^\s*Solution(?:\s+\d+)?\s*:?\s*(.*)$", re.IGNORECASE)
# Older rounds print a round-lettered marker instead of a bare integer:
# "Problem A1" (Algebra), "C8" (Calculus), "G3" (Geometry), "T4"/"AT10" (Team,
# Advanced Topics), plus "Gu1" (Guts) and "O1" (Oral). Any 1-3 letter prefix
# glued (or spaced) to the problem number counts.
_PREFIXED_MARKER_RE = re.compile(r"^\s*Problem\s+[A-Za-z]{1,3}\s*(\d+)\b", re.IGNORECASE)
# OCR sometimes renders \boxed{X} as a <box>X</box> tag; both are answer boxes.
_BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)


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

        A Guts round is a special case: its "solutions" document is really an
        answer key -- every block is a bare answer, not a worked solution -- so
        those entries are dropped here (they live only in ``problem_answer.json``
        via `_bare_answer`). This is detected document-wide, never per-problem,
        so a normal round's occasional short one-line solution is still kept.
        """
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
            grouped.setdefault(item["problem"], []).append(value)
        bodies = {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}
        nonempty = [b for b in bodies.values() if b]
        bare = [b for b in nonempty if _bare_answer(b)]
        if nonempty and len(bare) >= _ANSWER_KEY_BARE_FRAC * len(nonempty):
            return {n: b for n, b in bodies.items() if b and not _bare_answer(b)}
        return bodies

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Read each problem's final answer from the solutions document.

        Three tiers, in order: a printed ``Answer:`` line or ``\\boxed{...}``
        (`_answer_value`); a Guts-style answer-key entry, whose whole block is
        just the bare answer with no restated statement or worked solution
        (`_bare_answer`); and, for the older per-subject rounds that bury the
        answer mid-sentence in prose, the answer LLM. Problems that resolve to
        nothing are omitted -- a partial key, never a guessed one.
        """
        answers = {}
        blocks = _group_blocks("\n\n".join(pages_markdown), self.match_marker())
        for number, block in blocks.items():
            answer = _answer_value(block) or _bare_answer(block) or answer_llm.extract(block)
            if answer and answer.upper() != "N/A":
                answers[number] = answer
        return answers


def _group_blocks(full_text: str, match) -> dict:
    """Group the solution document's text into ``{problem_number: block_text}``.

    Uses the same layout splitter and markers as the solution/figure passes so
    every path numbers problems identically. Figure positions are dropped (the
    answer regexes and the LLM never need them)."""
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(full_text, match):
        if item["problem"] is None or item["kind"] != "text":
            continue
        grouped.setdefault(item["problem"], []).append(item["text"])
    return {number: "\n".join(parts).strip() for number, parts in grouped.items()}


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
    # No explicit "Answer:" line -- fall back to the boxed final answer.
    return _boxed_answer(block) or ""


def _iter_boxes(text: str):
    r"""Yield each answer box's inner text: ``\boxed{...}`` (brace-balanced, so a
    nested ``\frac{a}{b}`` survives) and each ``<box>...</box>`` tag, in order."""
    i = 0
    while True:
        j = text.find(r"\boxed", i)
        if j == -1:
            break
        open_brace = text.find("{", j)
        if open_brace == -1:
            break
        depth, k = 0, open_brace
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    yield text[open_brace + 1 : k]
                    break
            k += 1
        i = k + 1
    yield from _BOX_TAG_RE.findall(text)


def _boxed_answer(block: str) -> str:
    """The last boxed value in `block` (the final answer), cleaned, or ""."""
    value = ""
    for box in _iter_boxes(block):
        box = box.strip().strip("$").strip().rstrip(".").strip()
        if box:
            value = box
    return value


# A Guts answer-key entry ("3. 57", "7. $(\frac{x^2+1}{2})^m$.") restates no
# problem and works no solution -- the whole block is the bare answer on one
# short line. A real (prose) solution spans multiple lines or runs long, so
# those are left to the answer LLM instead of being mistaken for the answer.
_BARE_ANSWER_MAX_LEN = 80
# A document is treated as a Guts-style answer key (bare answers, no solutions)
# only when this fraction of its non-empty blocks are bare -- so one short
# solution in an otherwise-worked round never trips it.
_ANSWER_KEY_BARE_FRAC = 0.8


def _bare_answer(block: str) -> str:
    """`block` treated as a verbatim bare answer (Guts key), cleaned, or ""."""
    stripped = block.strip()
    if not stripped or "\n" in stripped or stripped.endswith("?"):
        return ""
    if len(stripped) > _BARE_ANSWER_MAX_LEN:
        return ""
    # Drop surrounding math delimiters and trailing punctuation together, so a
    # "$...$." entry loses both the closing "$" and the period after it.
    return stripped.strip(" $.")


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
