"""PUMaC (Princeton University Math Competition): per-round test/solution PDFs.

On-disk layout (data dir is ``PUMaC/out``)::

    out/<year>/<A|B>/<subject>/test.pdf        # subject rounds
    out/<year>/<A|B>/<subject>/solutions.pdf
    out/<year>/team/test.pdf                    # team round
    out/<year>/team/solutions.pdf

The subject rounds (``algebra``, ``combinatorics``, ``geometry``,
``number_theory``, ``individual_finals``) and the ``team`` round both number
their problems plainly (``1.``, ``2.``, ...), so they fit the default pipeline.
The ``power`` and ``live`` rounds use hierarchical numbering (``Problem 1.1.1``,
``1.1 [5]``) that the integer-keyed pipeline can't represent, so `discover_tests`
deliberately skips them.

Numbering quirk: recent Team rounds print a leading, *unnumbered* ``Bonus:``
question above problem 1 (older ones numbered it ``16.``). The bonus has no
number and sits before problem 1, so `match_marker` maps ``Bonus:`` to problem
``0`` -- it sorts first (0 < 1 < ... < 15), keeping the strictly-increasing
marker guard happy, and is inert on every test that has no ``Bonus:`` label.

Answers live inside the solutions PDF, in a format that drifted over the years,
so `answer_source` reuses that document's OCR and `parse_answers` reads the
answer out of each problem's block. Four printed markers are handled
deterministically, tried in order: a modern ``**Answer:** X`` line (plain, or a
``$\\boxed{X}$`` / ``<box>X</box>`` value); the older ``(ANS: X CB: ...)`` credit
tag, when its leading token is a clean value; a bare ``\\boxed{X}`` in the prose
(distinct boxes kept -- an intended answer plus an accepted alternate, or a boxed
intermediate); and the 2009/2010 ``**Solution.** X. ...`` opener, whose first
token is the answer. Everything else -- proof problems with no closed-form
answer, and older prose that only states the answer inside a sentence -- falls
back to the answer LLM (`answer_llm.extract`).
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, answer_llm, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test

# Subject-round folder names (3-level: <year>/<A|B>/<subject>). Everything else
# two levels deep under a year (power, live) uses numbering we don't support.
_SUBJECTS = {"algebra", "combinatorics", "geometry", "individual_finals", "number_theory"}

# Rounds that are always proof-based ("Prove that ...") and so have no answer
# key: the individual finals. Their worked solutions are still scraped; only the
# answer path is skipped (see answer_source), sparing the OCR + LLM a document
# with no numeric answer to find -- and no hallucinated key to write.
_PROOF_ROUNDS = ("individual_finals",)

# A leading "Bonus:" question (Team round) -> problem 0. Only the label is
# consumed; the statement that follows becomes problem 0's text.
_BONUS_RE = re.compile(r"^\s*Bonus\b\s*:?", re.IGNORECASE)

# Solution-block furniture in the solutions PDF. Each problem is restated with
# its number, credited ("Proposed by ..."), then the worked solution follows a
# "Solution:" label. Only the worked solution is kept.
_SOLUTION_LABEL_RE = re.compile(r"^\**\s*Solution\b\s*:?\**\s*", re.IGNORECASE)
_PROPOSED_RE = re.compile(r"^\**\s*Proposed\s+by\b", re.IGNORECASE)

# --- Answer markers (see module docstring; used by parse_answers) ---
# Modern "**Answer:** X" line (emphasis optional): the value is the rest of the
# line, itself possibly a boxed/tagged span cleaned by _clean_value.
_ANSWER_LABEL_RE = re.compile(r"(?im)^\s*[*_#]{0,3}\s*Answer\b\s*[*_]{0,3}\s*[:.]\s*(.+?)\s*$")
# Older "(ANS: X ... CB: names)" credit tag: capture up to a "CB:" author list or
# the closing paren, then take only a clean leading value (_ANS_VALUE_RE); when
# the tag holds prose instead ("(ANS: Let s_n be ...)") the value match fails and
# the block is left to the LLM fallback.
_ANS_PAREN_RE = re.compile(r"\(\s*ANS\s*[:=]\s*(.*?)\s*(?:\bCB\b\s*[:=].*)?\)", re.I | re.S)
# A clean answer value at the start of an "(ANS: ...)" tag or a "**Solution.** X"
# line: a "$...$" math span, or a plain integer / simple fraction. Anything else
# (a word, a whole sentence) doesn't match, so only unambiguous values are taken
# deterministically.
_ANS_VALUE_RE = re.compile(r"^\s*(\$[^$]+\$|-?\d+(?:\s*/\s*\d+)?)")
# The 2009/2010 format states the answer as the first token of the worked
# solution: "**Solution.** 455. We compute ...". Capture the text after the
# "Solution." / "Solution:" label (emphasis on either side of the punctuation);
# a clean leading value from it (_ANS_VALUE_RE) is the answer, while a label
# followed by prose ("**Solution.** We compute ...") yields no value and is left
# to the fallback. A multi-solution header ("**First Solution:**") doesn't start
# with the label, so it never matches.
_SOLUTION_ANSWER_RE = re.compile(r"(?im)^\s*[*_#>\s]*Solution\b[*_\s]*[.:][*_\s]*(.+)$")
# OCR sometimes renders \boxed{X} as a <box>X</box> tag; both are answer boxes.
_BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)


def _match_marker(text):
    """Match a PUMaC problem marker, mapping a leading ``Bonus:`` to problem 0.

    Falls back to the built-in matcher (``1.``, ``Problem 1``, ...) for every
    numbered problem, so this is a superset of the default behavior.
    """
    m = _BONUS_RE.match(text)
    if m:
        return 0, m.end()
    return anchors._match_marker(text)


class PumacSeries(Series):
    name = "pumac"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Subject rounds (``<year>/<A|B>/<subject>``) and ``<year>/team``.

        The ``power`` and ``live`` rounds are skipped -- their hierarchical
        numbering doesn't fit the integer-keyed pipeline.
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/*/test.pdf")):
            if pdf.parent.name not in _SUBJECTS:
                continue
            year, division, subject = pdf.parent.parent.parent.name, pdf.parent.parent.name, pdf.parent.name
            tests.append(Test(id=f"{year}_{division}_{subject}", source=pdf))
        for pdf in sorted(root.glob("*/team/test.pdf")):
            tests.append(Test(id=f"{pdf.parent.parent.name}_team", source=pdf))
        return tests

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        """Drop the Princeton shield logo (and stylized title banner) DETR reads
        as a Picture in the running header of every page.

        Both sit in the top ~9% of the page (logo center ~0.076, title banner
        ~0.086 of page height); the first real content -- the problem-1 marker --
        is at ~0.17, and a statement figure is always lower still. A 0.12 cutoff
        clears the header furniture without reaching any real figure. Without
        this, the logo binds to problem 1, because the title text above it is a
        left-margin "start" that defeats the drop-above-first-problem guard (see
        pipeline._assign_pictures)."""
        return config.LayoutOptions(header_picture_frac=0.12)

    @override
    def duplicate_scope(self, test_id, across=False):
        """Bucket by year. PUMaC reuses problems across a year's A/B divisions
        (and, less often, its subject rounds), so every ``<year>_*`` test --
        ``2018_A_algebra``, ``2018_B_algebra``, ``2018_team``, ... -- is compared
        together, but never against a different year's tests.

        With ``across=True`` (``dedup --across-years``) the year is dropped and
        every test shares one bucket, catching problems recycled across years."""
        if across:
            return "all"
        return test_id.split("_", 1)[0]

    @override
    def solution_source(self, test):
        """The fixed-name sibling ``solutions.pdf``, or None if absent.

        Roughly a fifth of the tests (notably every 2024 round) ship without a
        solutions PDF; returning None lets the pipeline skip them cleanly.
        """
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def parse_solutions(self, full_text):
        """Segment the solutions document into {problem_number: solution_text}.

        The solutions PDF restates each problem (with its number), credits it
        ("Proposed by ..."), then gives the worked solution under a "Solution:"
        label. We split on the same problem markers the pipeline uses for figure
        assignment, then keep only the text from "Solution:" onward -- dropping
        the restated statement and the proposer credit so the output is the
        solution, not a duplicate of problems.json. When a block has no
        recognizable "Solution:" label (unexpected layout / OCR miss), the whole
        block minus the credit line is kept so nothing is lost. Statement-figure
        crops still bind to the problem geometrically (see
        pipeline.process_solution_document); only the text is trimmed.
        """
        return {n: _solution_body(block) for n, block in _group_blocks(full_text, self.match_marker()).items()}

    @override
    def answer_source(self, test):
        """The solutions PDF (answers live inside it) -- or None for a proof round.

        The individual finals are all "Prove that ..." problems with no numeric
        answer, so there is no key to extract: returning None makes the pipeline
        skip the answer path entirely (no OCR, no LLM) while still scraping the
        round's worked solutions.
        """
        if test.id.endswith(_PROOF_ROUNDS):
            return None
        return self.solution_source(test)

    @override
    def parse_answers(self, test, pages_markdown):
        """Read each problem's final answer from the solutions document.

        The pages are grouped into per-problem blocks by the same markers the
        solution/figure passes use, then each block's answer is taken from its
        printed marker (`_extract_answer`); a block with no recognizable marker
        -- pre-2010 prose that just states the answer in a sentence, or an
        "(ANS: ...)" tag whose value is prose -- falls back to the answer LLM.
        Problems the LLM also can't resolve are omitted (a partial key, never a
        guessed one).
        """
        answers: dict[int, str] = {}
        for n, block in _group_blocks("\n".join(pages_markdown), self.match_marker()).items():
            value = _extract_answer(block) or answer_llm.extract(block)
            if value:
                answers[n] = value
        return answers


def _group_blocks(full_text: str, match) -> dict:
    """Group solution-document text into ``{problem_number: block_text}``.

    Reuses the nanonets layout splitter so figure crops and the solution/answer
    passes all number problems identically; figure positions are kept as a
    sentinel (harmless to the answer regexes, and needed by _solution_body's
    caller for inline figure alignment). Content before problem 1 is dropped.
    """
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(full_text, match):
        if item["problem"] is None:
            continue
        if item["kind"] == "text":
            grouped.setdefault(item["problem"], []).append(item["text"])
        elif item["kind"] == "image":
            grouped.setdefault(item["problem"], []).append(FIGURE_PLACEHOLDER)
    return {n: "\n".join(parts) for n, parts in grouped.items()}


def _solution_body(block: str) -> str:
    """Return just the worked solution from one restated-problem block.

    Everything up to and including the first "Solution:" label is the restated
    statement and proposer credit; drop it. If no label is present, fall back to
    the whole block with any "Proposed by ..." lines removed.
    """
    lines = block.splitlines()
    for i, line in enumerate(lines):
        m = _SOLUTION_LABEL_RE.match(line)
        if m:
            rest = line[m.end():]
            kept = ([rest] if rest.strip() else []) + lines[i + 1:]
            return "\n".join(kept).strip()
    return "\n".join(l for l in lines if not _PROPOSED_RE.match(l)).strip()


def _extract_answer(block: str):
    """Return the answer printed in one problem's block, or None.

    Markers are tried most-authoritative first: the modern "**Answer:** X" line;
    then an "(ANS: X ...)" credit tag with a clean leading value; then any
    ``\\boxed{}`` / ``<box>`` in the prose; then the 2009/2010 "**Solution.** X."
    opener. Returns None when no marker yields a value, leaving the block for the
    LLM fallback. See the module docstring.
    """
    m = _ANSWER_LABEL_RE.search(block)
    if m:
        value = _clean_value(m.group(1))
        if value:
            return value
    m = _ANS_PAREN_RE.search(block)
    if m:
        value = _leading_value(m.group(1))
        if value:
            return value
    boxes = _distinct_boxes(block)
    if boxes:
        # Usually one box (the final answer). Multiple *distinct* boxes are an
        # intended answer plus an accepted alternate or a boxed intermediate
        # (both rare, both indistinguishable without semantics), so all are kept.
        return ", ".join(boxes)
    for m in _SOLUTION_ANSWER_RE.finditer(block):
        value = _leading_value(m.group(1).lstrip("*_ "))
        if value:
            return value
    return None


def _leading_value(text: str):
    """Clean answer value at the start of `text` (a number/fraction or "$...$"
    span), or None when it opens with prose. Used for the "(ANS: ...)" tag and
    the "**Solution.** X" opener, which both put the answer first, then prose."""
    m = _ANS_VALUE_RE.match(text)
    return _clean_value(m.group(1)) if m else None


def _iter_boxes(text: str):
    r"""Yield each answer box's inner text: ``\boxed{...}`` (brace-balanced, so a
    nested ``\frac{a}{b}`` survives) and each ``<box>...</box>`` tag."""
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
                    yield text[open_brace + 1:k]
                    break
            k += 1
        i = k + 1
    yield from _BOX_TAG_RE.findall(text)


def _distinct_boxes(text: str):
    """Boxed values in order, de-duplicated (a restated box isn't a new answer)."""
    out = []
    for box in _iter_boxes(text):
        box = box.strip().strip("$").strip()
        if box and box not in out:
            out.append(box)
    return out


def _clean_value(raw: str):
    """Normalize a marker's raw value to a bare answer string, or None.

    Unwraps a ``\\boxed{}`` / ``<box>`` value (taking the last, the final answer),
    then strips LaTeX ``$`` delimiters, emphasis, and a trailing period.
    """
    boxes = _distinct_boxes(raw)
    value = boxes[-1] if boxes else raw
    value = value.strip().strip("$").strip()
    value = re.sub(r"^\*+|\*+$", "", value).strip().rstrip(".").strip()
    return value or None
