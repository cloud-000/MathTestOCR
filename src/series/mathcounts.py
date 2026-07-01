"""Mathcounts: one test per round.

On-disk layout (data dir is ``Mathcounts/out``)::

    out/<year>/<level>/<round>.pdf

A ``<year>/<level>`` folder mixes problem rounds (sprint, target, team,
countdown, ...) with a single shared ``solutions.pdf`` and ``answers.pdf`` that
cover several rounds at once. Each problem round is its own test (id
``<year>_<level>_<round>``, e.g. ``2025_state_sprint``); the round whitelist is
``config.MATHCOUNTS_TEST_ROUNDS``.

Worked solutions are **deferred**: the shared ``solutions.pdf`` restarts
numbering across rounds, so mapping it back to individual rounds needs a real
OCR run to inspect the alignment first; `has_solutions` stays False.

**Answers** are wired: the shared ``answers.pdf`` is a title page followed by a
page run per round, each page headed "<Round> Round" ("Sprint Round" spans two
pages, then Target, Team, and the remaining pages Countdown). `parse_answers`
selects the parsed test's pages by that header in the OCR markdown -- some
years are scanned with no text layer, so the header must come from OCR, not
the PDF -- and reads the ``N. ____ answer`` blank lines off them.
"""

import re
from pathlib import Path

from .. import config
from .base import Series, Test, numbered_answers_in_line

# Round stem (the test PDF's filename) -> the header its answer-key pages carry.
# Practice rounds (warmups/workouts/masters) publish separate keys with a
# different layout and are not wired yet.
_ANSWER_ROUND_HEADERS = {
    "sprint": "sprint round",
    "target": "target round",
    "team": "team round",
    "countdown": "countdown round",
    "cdr": "countdown round",
}

# MATHCOUNTS fill-in-the-blank problems print "____ cm" or "____ factors" (a
# blank plus an optional unit) right after the problem number, before the
# statement itself ("In the figure, ..."). Both engines already strip the
# printed problem number and collapse the blank rule; what is left dangling at
# the front of the statement is just the unit word(s). Strip the blank run
# first (in case an engine left literal underscores behind), then up to a few
# lowercase unit words -- but only when a capitalized statement follows, so a
# blank-less problem's real text is never touched.
_LEADING_BLANK_RE = re.compile(r"^_+\s*")
_LEADING_UNIT_RE = re.compile(r"^(?:[a-z][\w°²³./-]*\s+){1,3}(?=[A-Z(])")

# Boilerplate that marks a page as having no problems worth parsing: the cover
# sheet in front of every round, the divider MATHCOUNTS reprints before each
# pair of Target Round problems ("every other page" in that round), and the
# Forms of Answers page some rounds append at the end. All three carry this
# exact instructional text regardless of year/level, so a plain substring
# check on the PDF's own text layer is enough -- no OCR needed to skip them.
_SKIP_PAGE_PHRASES = ("do not begin until you are instructed", "forms of answers")


class MathcountsSeries(Series):
    name = "mathcounts"
    has_solutions = False  # shared per-level solutions.pdf deferred -- see module docstring
    has_answers = True

    def layout_options(self):
        """Opt into the MATHCOUNTS-tuned nanonets figure/table heuristics.

        MATHCOUNTS pages need all three (see `config.LayoutOptions`): a recurring
        whole-page false-positive Picture box (filtered by area), problems packed
        into a single answer-blank ``<table>`` (marker rows unpacked to text), and
        faint number boxes that sometimes miss detection and drop a problem's
        left-margin start (recovered by the gap-based fallback). Other series keep
        the conservative base defaults.
        """
        return config.LayoutOptions(
            max_picture_area_frac=config.NANONETS_MAX_PICTURE_AREA_FRAC,
            gap_based_picture_fallback=True,
            split_marker_table_rows=True,
        )

    def skip_page(self, text):
        # Older booklets double-space this boilerplate ("DO  NOT  BEGIN...")
        # and can wrap it across a line break; collapse all whitespace runs
        # (including newlines) to a single space before matching.
        collapsed = re.sub(r"\s+", " ", text).strip().lower()
        return any(phrase in collapsed for phrase in _SKIP_PAGE_PHRASES)

    def discover_tests(self, data_dir):
        """One test per whitelisted ``<year>/<level>/<round>.pdf``."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/*.pdf")):
            if pdf.stem not in config.MATHCOUNTS_TEST_ROUNDS:
                continue
            test_id = f"{pdf.parent.parent.name}_{pdf.parent.name}_{pdf.stem}"
            tests.append(Test(id=test_id, source=pdf))
        return tests

    def answer_source(self, test):
        """The shared ``answers.pdf`` sibling covering every round, or None."""
        if test.source.stem not in _ANSWER_ROUND_HEADERS:
            return None  # practice rounds -- separate key files, not wired yet
        src = test.source.parent / "answers.pdf"
        return src if src.exists() else None

    def parse_answers(self, test, pages_markdown):
        """Pull this round's answers out of the shared answer-key document.

        Pages are selected by the round header ("Sprint Round", ...) appearing
        anywhere in their OCR markdown; the title page carries no round header
        and drops out on its own. Entries are ``N. ____ answer`` blank lines,
        several per OCR'd line when the key is laid out in columns; unit words
        printed under a blank sit on their own line and are ignored. First
        occurrence of a number wins.
        """
        header = _ANSWER_ROUND_HEADERS[test.source.stem]
        pages = [
            md for md in pages_markdown
            if header in re.sub(r"\s+", " ", md).lower()
        ]
        answers = {}
        for markdown in pages:
            for line in markdown.splitlines():
                _, pairs = numbered_answers_in_line(line)
                for n, answer in pairs:
                    if answer:
                        answers.setdefault(n, answer)
        return answers

    def postprocess(self, problems):
        """Drop the leaked answer-blank unit from each problem's first text element."""
        for problem in problems:
            for element in problem.elements:
                if element.kind != "text":
                    continue
                text = _LEADING_BLANK_RE.sub("", element.text)
                element.text = _LEADING_UNIT_RE.sub("", text)
                break
        return problems
