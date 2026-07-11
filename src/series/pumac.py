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
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test

# Subject-round folder names (3-level: <year>/<A|B>/<subject>). Everything else
# two levels deep under a year (power, live) uses numbering we don't support.
_SUBJECTS = {"algebra", "combinatorics", "geometry", "individual_finals", "number_theory"}

# A leading "Bonus:" question (Team round) -> problem 0. Only the label is
# consumed; the statement that follows becomes problem 0's text.
_BONUS_RE = re.compile(r"^\s*Bonus\b\s*:?", re.IGNORECASE)

# Solution-block furniture in the solutions PDF. Each problem is restated with
# its number, credited ("Proposed by ..."), then the worked solution follows a
# "Solution:" label. Only the worked solution is kept.
_SOLUTION_LABEL_RE = re.compile(r"^\**\s*Solution\b\s*:?\**\s*", re.IGNORECASE)
_PROPOSED_RE = re.compile(r"^\**\s*Proposed\s+by\b", re.IGNORECASE)


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
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            if item["kind"] == "text":
                grouped.setdefault(item["problem"], []).append(item["text"])
            elif item["kind"] == "image":
                grouped.setdefault(item["problem"], []).append(FIGURE_PLACEHOLDER)
        return {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}


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
