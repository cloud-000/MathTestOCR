"""Mathcounts: one test per round.

On-disk layout (data dir is ``Mathcounts/out``)::

    out/<year>/<level>/<round>.pdf

A ``<year>/<level>`` folder mixes problem rounds (sprint, target, team,
countdown, ...) with a single shared ``solutions.pdf`` and ``answers.pdf`` that
cover several rounds at once. Each problem round is its own test (id
``<year>_<level>_<round>``, e.g. ``2025_state_sprint``); the round whitelist is
``config.MATHCOUNTS_TEST_ROUNDS``.

Solutions are **deferred**: the shared ``solutions.pdf``/``answers.pdf`` restart
numbering across rounds, so mapping them back to individual rounds needs a real
OCR run to inspect the alignment first. Until then `has_solutions` is False and
the `solutions` command cleanly skips this series.
"""

import re
from pathlib import Path

from .. import config
from .base import Series, Test

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
    has_solutions = False  # shared per-level solutions.pdf/answers.pdf -- see module docstring

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
