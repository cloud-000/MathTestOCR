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
from .base import Series, Test


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
        """The answers live in the solutions PDF."""
        return self.solution_source(test)

    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Parse answers from the solutions PDF pages.

        Each problem block contains "Answer: <answer>" followed by "Solution:".
        """
        answers = {}
        full_text = "\n\n".join(pages_markdown)
        pattern = re.compile(
            r"Problem\s+(\d+)\s+.*?Answer:\s*(.*?)\s*(?:Solution:|\Z)",
            re.DOTALL | re.IGNORECASE
        )
        for p_num, ans in pattern.findall(full_text):
            ans_lines = []
            for line in ans.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip out headers/footers that might appear from OCR
                if "Math Prize" in line or "Solutions" in line:
                    continue
                if line.isdigit() and (not ans_lines or len(line) <= 2):
                    # Skip page numbers
                    continue
                ans_lines.append(line)
            answers[int(p_num)] = " ".join(ans_lines).strip()
        return answers
