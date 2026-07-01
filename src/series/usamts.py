"""USAMTS: one PDF per test, one solution PDF per test.

On-disk layout (data dir is ``USAMTS/out``)::

    out/<year>/<round>/test.pdf
    out/<year>/<round>/solutions.pdf

so each ``<year>/<round>`` is one test (id ``<year>_<round>``) and its solution
is the fixed-name sibling ``solutions.pdf``.

Numbering quirk: USAMTS prints problems as ``1/3/37.`` (problem / round / year).
The default matcher captures the *year* (the last group), which collapses every
problem on a round to the same number. Here we capture the *first* component --
the problem index within the round -- so problems come out 1, 2, 3, ... (see
TODOS.txt).
"""

import functools
import re
from pathlib import Path

from typing_extensions import override

from .. import anchors
from .base import Series, Test

# USAMTS-specific marker patterns. The "N/R/Y." form must come first and capture
# the leading problem index; the other forms mirror the defaults as a fallback.
_USAMTS_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*/\s*\d+\s*/\s*\d+\s*\."),  # "1/3/37." -> 1
    re.compile(r"^\s*(?:Problem|Question|Prob\.?)\s+(\d+)", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*[.)]"),  # "1." / "1)"
]

# USAMTS closes the problem set with a rule of asterisks ("**************")
# followed by submission instructions and a mailing address. That trailing
# furniture has no problem marker, so it binds to the last problem -- cut it.
_SEPARATOR_RE = re.compile(r"^\s*\*{5,}\s*$")


class UsamtsSeries(Series):
    name = "usamts"
    has_solutions = True

    def discover_tests(self, data_dir):
        """One test per ``<year>/<round>/test.pdf`` under the data dir."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/test.pdf")):
            test_id = f"{pdf.parent.parent.name}_{pdf.parent.name}"
            tests.append(Test(id=test_id, source=pdf))
        return tests

    def match_marker(self):
        return functools.partial(anchors._match_marker, patterns=_USAMTS_PATTERNS)

    def solution_source(self, test):
        """The fixed-name sibling ``solutions.pdf``, or None if absent."""
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    def postprocess(self, problems):
        """Drop the trailing submission-instructions/address footer.

        Everything from the ``**************`` separator onward (within whichever
        problem it landed in) is page furniture, not problem content.
        """
        for p in problems:
            kept = []
            cut = False
            for el in p.elements:
                if cut:
                    continue  # drop all furniture after the separator
                if el.kind == "text":
                    lines = el.text.splitlines()
                    for i, line in enumerate(lines):
                        if _SEPARATOR_RE.match(line):
                            el.text = "\n".join(lines[:i]).rstrip()
                            cut = True
                            break
                    if not el.text.strip():
                        continue  # element became empty
                kept.append(el)
            p.elements = kept
        return problems
