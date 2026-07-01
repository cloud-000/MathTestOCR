"""Purple Comet: test PDF plus a pre-downloaded answer key.

On-disk layout (data dir is ``PurpleComet/out``)::

    out/<year>/<division>/test.pdf
    out/<year>/<division>/answers.txt

so each ``<year>/<division>`` (division is ``MS`` or ``HS``) is one test, id
``<year>_<division>``. Purple Comet publishes no worked solutions, only an answer
key -- and the data repo already scraped it into ``answers.txt`` (a TSV with a
``Problem #\tAnswer`` header), so `scrape_answers` just reads that file: no
network, no extra deps. The `solutions` command writes these as
``problem_<n>_answer.txt``.
"""

from pathlib import Path

from .. import config
from .base import Series, Test


class PurpleCometSeries(Series):
    name = "purplecomet"
    has_solutions = False  # no worked solutions -- answer key only (see scrape_answers)
    has_answers = True

    def layout_options(self):
        # Purple Comet prints each problem number on its own "Problem N" heading
        # line above the statement, so DETR sees two left-margin boxes per problem.
        # Take the problem start from the heading alone, or figure assignment
        # drifts down the page (a problem-3 figure lands on problem 6).
        return config.LayoutOptions(problem_start_from_headers=True)

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

    def scrape_answers(self, test):
        """Read the sibling ``answers.txt`` (TSV) into {problem_number: answer}."""
        answers_file = test.source.parent / "answers.txt"
        if not answers_file.exists():
            return {}
        answers = {}
        for line in answers_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            number, answer = parts[0].strip(), parts[1].strip()
            if not number.isdigit():  # skips the "Problem #\tAnswer" header
                continue
            answers[int(number)] = answer
        return answers

    def postprocess(self, problems):
        # TODO: merge problem 19's nested "1. 2. 3." conditions back into its
        # statement instead of treating them as separate problems (needs a real
        # OCR run to validate the heuristic before implementing).
        return problems
