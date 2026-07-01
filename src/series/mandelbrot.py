"""Mandelbrot: several tests per season, each with a sibling solution PDF.

On-disk layout (data dir is ``Mandelbrot/out``)::

    out/<season>/tmctest<n>{N,R}.pdf   individual rounds (National / Regional)
    out/<season>/tmcsoln<n>{N,R}.pdf   their solutions
    out/<season>/mtptest<n>.pdf        team-play rounds
    out/<season>/mtpsoln<n>.pdf        their solutions
    out/<season>/mtptopics<n>.pdf      topic lists (not problems -- skipped)

A test is any ``*test*.pdf`` (id ``<season>_<stem>``, e.g. ``2017-18_tmctest1N``);
``mtptopics*`` is excluded because it has no ``test`` in the name. The solution is
the sibling with ``test`` swapped to ``soln``.

The solution PDF opens with an "Answer Key" box -- short ``N. answer`` entries
in column order (``4. 12  1. 5  5. 3025 ...``) -- before the worked solutions
begin. Its out-of-order markers would corrupt the strictly-increasing problem
segmentation, so `clean_solution_markdown` strips the box before any problem
tagging, and `parse_answers` reads the same box back out as the answer key.
"""

import re
from pathlib import Path

from .. import config
from .base import Series, Test, numbered_answers_in_line

_TAG_RE = re.compile(r"<[^>]+>")
# A line that is only table scaffolding (or blank). The key box is often OCR'd
# as an HTML table; its <table>/<tr> wrapper lines must be stripped along with
# the entries, or a dangling "<table>" left at the top of the page pairs with
# some later table's "</table>" and swallows every solution between them.
_TABLE_MARKUP_ONLY_RE = re.compile(
    r"^\s*(?:</?(?:table|thead|tbody|tr|td|th)\b[^>]*>\s*)*$", re.IGNORECASE
)
# A key entry is a bare value ("12", "9/17", "3025"); solution prose after a
# marker ("1. It is possible to fit...") is far longer. Used to find where the
# box ends.
_MAX_ANSWER_LEN = 24


def _is_answer_key_heading(line: str) -> bool:
    """True when `line` carries the "Answer Key" heading.

    Nanonets renders the box's small-caps title with markup *between the
    letters* ("<strong>A</strong><sub>NSWER</sub> ..."), so tags become nothing
    and all spacing is dropped before matching.
    """
    return "answerkey" in re.sub(r"[^a-z]", "", _TAG_RE.sub("", line).lower())


def _split_answer_key(text: str):
    """Split OCR markdown into ``(answers, text_without_the_key)``.

    The key region starts at the "Answer Key" heading (entries may share that
    line when the box is OCR'd as one row) and ends at the first line that is
    neither blank/markup-only nor made of short ``N. answer`` entries -- in
    practice solution 1's opening prose. Returns ``({}, text)`` unchanged when
    no heading or no entries are found, so pages without a key pass through.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _is_answer_key_heading(ln)), None)
    if start is None:
        return {}, text
    answers = {}

    def consume(line, allow_leading=False):
        """Record the line's entries; False if it isn't a pure answer-key line."""
        leading, pairs = numbered_answers_in_line(line)
        if (leading and not allow_leading) or not pairs:
            return False
        if any(len(a) > _MAX_ANSWER_LEN for _, a in pairs):
            return False
        for n, a in pairs:
            if a:
                answers.setdefault(n, a)
        return True

    # Entries can share the heading line (the box OCR'd as one row); the
    # heading text itself is the leading part, so it is allowed there.
    consume(lines[start], allow_leading=True)
    end = start + 1
    for i in range(start + 1, len(lines)):
        if _TABLE_MARKUP_ONLY_RE.match(lines[i]):
            end = i + 1  # blank / table-scaffolding line inside the box
            continue
        if not consume(lines[i]):
            break
        end = i + 1
    if not answers:
        return {}, text  # heading matched but no entries -- don't strip anything
    while start > 0 and _TABLE_MARKUP_ONLY_RE.match(lines[start - 1]):
        start -= 1  # the box's own <table>/<tr> wrapper lines above the heading
    return answers, "\n".join(lines[:start] + lines[end:])


class MandelbrotSeries(Series):
    name = "mandelbrot"
    has_solutions = True
    has_answers = True

    def layout_options(self):
        """Nudge the OCR temperature off greedy for Mandelbrot's grid pages.

        Greedy decoding (temperature 0.0) sometimes gets stuck repeating a
        ``<table>`` row when a page shows a grid/diagram, blowing past the
        runaway guard or padding the transcription. A small bump breaks the loop
        while keeping transcription faithful; the layout heuristics stay at the
        conservative base defaults. Raise further only if grids still loop.
        """
        return config.LayoutOptions(nanonets_temperature=0.1)

    def discover_tests(self, data_dir):
        """One test per ``*test*.pdf`` inside each season folder."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for season in sorted(p for p in root.iterdir() if p.is_dir()):
            for pdf in sorted(season.glob("*test*.pdf")):
                tests.append(Test(id=f"{season.name}_{pdf.stem}", source=pdf))
        return tests

    def solution_source(self, test):
        """Sibling solution PDF: the test name with ``test`` swapped to ``soln``."""
        src = test.source
        sol = src.with_name(src.name.replace("test", "soln"))
        return sol if sol.exists() else None

    def answer_source(self, test):
        """The key lives inside the solutions PDF itself (its front box)."""
        return self.solution_source(test)

    def clean_solution_markdown(self, page_index, markdown):
        """Strip the front answer-key box so problem tagging sees only solutions.

        Left in place, the box's out-of-order entries ("4. 12" before "1. 5")
        would be read as problem markers, mis-numbering every solution and
        figure after them. The answers themselves are recovered separately by
        `parse_answers` from the raw markdown.
        """
        return _split_answer_key(markdown)[1]

    def parse_answers(self, test, pages_markdown):
        """Read the "Answer Key" box from the first page that carries one."""
        for markdown in pages_markdown:
            answers, _ = _split_answer_key(markdown)
            if answers:
                return answers
        return {}

    # match_marker stays default: rounds number "1." / "1)" normally. Add an
    # override here (mirroring UsamtsSeries) if a real run reveals a quirk.
