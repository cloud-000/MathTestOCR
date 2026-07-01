"""Purple Comet: scaffold.

Purple Comet has no published solutions -- only an *answer key* on a web page
(URL to be provided). Source-folder format is also TBD; for now it inherits the
default "one PDF per test" discovery. Two extension points are stubbed:

  * scrape_answers -- fetch the answer key from the answers URL and return
    {problem_number: answer}; the solutions command will write these as
    ``problem_<n>_answer.txt``.
  * postprocess -- Purple Comet problem 19 prints nested conditions "1. 2. 3."
    that the marker logic can mistake for new problems (TODOS.txt); clean that up
    here once the parsing is exercised on real pages.
"""

from .base import Series

# Filled in once provided, e.g. "https://purplecomet.org/views/data/...".
ANSWERS_URL = None


class PurpleCometSeries(Series):
    name = "purplecomet"
    has_solutions = False  # answers only, fetched from the web (see scrape_answers)

    def scrape_answers(self, test):
        """Return {problem_number: answer} from the Purple Comet answer key.

        TODO: implement once ANSWERS_URL and the page format are known.
        """
        raise NotImplementedError(
            "Purple Comet answer scraping not implemented yet -- provide ANSWERS_URL"
        )

    def postprocess(self, problems):
        # TODO: merge problem 19's nested "1. 2. 3." conditions back into its
        # statement instead of treating them as separate problems.
        return problems
