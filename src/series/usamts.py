"""USAMTS: one PDF per test, one solution PDF per test.

Numbering quirk: USAMTS prints problems as ``1/3/37.`` (problem / round / year).
The default matcher captures the *year* (the last group), which collapses every
problem on a round to the same number. Here we capture the *first* component --
the problem index within the round -- so problems come out 1, 2, 3, ... (see
TODOS.txt).
"""

import functools
import re

from .. import anchors, config
from .base import Series

# USAMTS-specific marker patterns. The "N/R/Y." form must come first and capture
# the leading problem index; the other forms mirror the defaults as a fallback.
_USAMTS_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*/\s*\d+\s*/\s*\d+\s*\."),  # "1/3/37." -> 1
    re.compile(r"^\s*(?:Problem|Question|Prob\.?)\s+(\d+)", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*[.)]"),  # "1." / "1)"
]


class UsamtsSeries(Series):
    name = "usamts"
    has_solutions = True

    def match_marker(self):
        return functools.partial(anchors._match_marker, patterns=_USAMTS_PATTERNS)

    def solution_source(self, test):
        """Per-test solution PDF.

        Looked up as ``<stem><USAMTS_SOLUTION_SUFFIX>.pdf`` next to the test, then
        as ``solutions/<stem>.pdf`` in a sibling folder. Returns None if neither
        exists (the test is then skipped by the solutions command).
        """
        src = test.source
        candidates = [
            src.with_name(f"{src.stem}{config.USAMTS_SOLUTION_SUFFIX}.pdf"),
            src.parent / "solutions" / f"{src.stem}.pdf",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None
