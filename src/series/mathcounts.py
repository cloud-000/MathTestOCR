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

from pathlib import Path

from .. import config
from .base import Series, Test


class MathcountsSeries(Series):
    name = "mathcounts"
    has_solutions = False  # shared per-level solutions.pdf/answers.pdf -- see module docstring

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
