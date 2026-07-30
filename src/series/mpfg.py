"""Math Prize for Girls (MPfG): test PDF and solution PDF.

On-disk layout (data dir is ``MPfG/out``)::

    out/<year>/<division>/test.pdf
    out/<year>/<division>/solutions.pdf

so each ``<year>/<division>`` (division is typically ``mathprize`` or ``olympiad``)
is one test, id ``<year>_<division>``.
"""

import re
from pathlib import Path

from .. import config
from .base import Series, Test, strip_solution_page_furniture


_SOLUTION_FURNITURE_RE = re.compile(
    r"^(?:"
    # Older solution PDFs use the full "AT MATH PRIZE FOR GIRLS ..."
    # masthead; newer ones omit the leading preposition.
    r"(?:at\s+)?math\s+prize\s+for\s+girls\b.*"
    r"|(?:the\s+)?advantage\s+testing(?:\s+foundation)?\b.*"
    r"|(?:math\s+prize|olympiad)\s+\d{4}\s+solutions?"
    r"|page\s+\d+"
    r")$",
    re.IGNORECASE,
)


class MpfgSeries(Series):
    name = "mpfg"
    has_solutions = True
    has_answers = True

    def discover_tests(self, data_dir):
        """One test per ``<year>/<division>/test.pdf`` under the data dir."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/test.pdf")):
            test_id = f"{pdf.parent.parent.name}_{pdf.parent.name}"
            tests.append(Test(id=test_id, source=pdf))
        return tests

    def layout_options(self):
        """Inline statement figures: MPfG problems interleave prose and figures,
        so a figure's position in the text carries meaning (see
        pipeline.inline_problem_figures)."""
        return config.LayoutOptions(inline_figures=True)

    def skip_page(self, text: str) -> bool:
        """Drop the directions/title page that precedes the problems.

        The ``mathprize`` division opens with a "Directions" page whose numbered
        instructions ("1. Do not open this test...", "2. Fill out the top of
        your answer sheet", ...) would otherwise be parsed as problems 1-4 and
        collide with the real problems. The ``olympiad`` division has no such
        page (its first page is problem 1), so key on the directions signature
        rather than a page index.
        """
        return "Do not open this test" in text

    def solution_source(self, test):
        """The solutions.pdf file in the same folder as test.pdf."""
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    def answer_source(self, test):
        """The answers live in the solutions PDF (mathprize only).

        The olympiad division is proof-based and prints no short answers, so its
        solution pages carry no "Answer:" label and `parse_answers` yields {}.
        """
        return self.solution_source(test)

    def clean_solution_markdown(self, page_index, markdown):
        """Remove MPfG's repeated solution masthead before page joining."""
        return strip_solution_page_furniture(
            markdown, line_patterns=(_SOLUTION_FURNITURE_RE,)
        )

    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Parse the short answer for each problem from the solutions OCR.

        Every mathprize problem block prints ``Answer: <value>`` immediately
        before ``Solution:``; the value is the cleanest source of the final
        answer (cleaner than the in-text ``\\boxed{...}``, which the OCR
        occasionally drops, and than the PDF text layer, which flattens stacked
        fractions like ``27/25`` into an ambiguous ``27 25``). The label carries
        proper ``$...$`` LaTeX, matching the answer-key convention.

        Markers and the ``Answer:``/``Solution:`` labels may be wrapped in
        markdown bold (``**Problem 7**``), so every anchor tolerates surrounding
        ``*``. When a page break falls between the answer and ``Solution:``, the
        running header/footer and page number leak into the captured span, so
        the value is taken as the first line that is real answer content.
        """
        full_text = "\n".join(pages_markdown)
        answers = {}
        marks = list(self._MARKER_RE.finditer(full_text))
        for m, nxt in zip(marks, marks[1:] + [None]):
            end = nxt.start() if nxt is not None else len(full_text)
            block = full_text[m.end():end]
            am = re.search(
                r"Answer:\*{0,2}\s*(.*?)\s*(?:\*{0,2}Solution:|\Z)",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if am is None:
                continue
            value = self._clean_answer(am.group(1))
            if value:
                answers[int(m.group(1))] = value
        return answers

    # Problem marker as it appears in the solutions OCR, tolerating markdown bold.
    _MARKER_RE = re.compile(r"\*{0,2}Problem\s+(\d+)\*{0,2}", re.IGNORECASE)
    # Running header/footer ("... Math Prize for Girls YYYY Solutions ...").
    _HEADER_RE = re.compile(r"math prize|advantage testing|for girls", re.IGNORECASE)

    @classmethod
    def _clean_answer(cls, raw: str) -> str:
        """Return the answer value from the span after ``Answer:``.

        The value is the first line with real content. On a page break the
        running header/logo/page number splice into the captured span, but only
        *after* the value (the label and its value are printed together), so
        returning at the first content line skips that junk without a
        content-based filter that might misfire on a short numeric answer. The
        header/logo skips guard only against a stray blank-then-header ordering.
        """
        for line in raw.splitlines():
            s = line.strip().strip("*").strip()
            if not s:
                continue
            if s.startswith("<img") or cls._HEADER_RE.search(s):
                continue
            return s.rstrip(".").strip()
        return ""
