"""Mandelbrot: several tests per season, each with a sibling solution PDF.

On-disk layout (data dir is ``Mandelbrot/out``)::

    out/<season>/tmctest<n>{N,R}.pdf   individual rounds (National / Regional)
    out/<season>/tmcsoln<n>{N,R}.pdf   their solutions
    out/<season>/mtptest<n>.pdf        team-play rounds
    out/<season>/mtpsoln<n>.pdf        their solutions
    out/<season>/mtptopics<n>.pdf      topic lists (not problems -- skipped)

A test is any ``*test*.pdf`` (id ``<season>_<stem>``, e.g. ``2017-18_tmctest1N``);
``mtptopics*`` is excluded because it has no ``test`` in the name. The solution is
the sibling with ``test`` swapped to ``soln``.
"""

from pathlib import Path

from .. import config
from .base import Series, Test


class MandelbrotSeries(Series):
    name = "mandelbrot"
    has_solutions = True

    def layout_options(self):
        """Nudge the OCR temperature off greedy for Mandelbrot's grid pages.

        Greedy decoding (temperature 0.0) sometimes gets stuck repeating a
        ``<table>`` row when a page shows a grid/diagram, blowing past the
        runaway guard or padding the transcription. A small bump breaks the loop
        while keeping transcription faithful; the layout heuristics stay at the
        conservative base defaults. Raise further only if grids still loop.
        """
        return config.LayoutOptions(nanonets_temperature=0.1)

    def discover_tests(self, data_dir):
        """One test per ``*test*.pdf`` inside each season folder."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for season in sorted(p for p in root.iterdir() if p.is_dir()):
            for pdf in sorted(season.glob("*test*.pdf")):
                tests.append(Test(id=f"{season.name}_{pdf.stem}", source=pdf))
        return tests

    def solution_source(self, test):
        """Sibling solution PDF: the test name with ``test`` swapped to ``soln``."""
        src = test.source
        sol = src.with_name(src.name.replace("test", "soln"))
        return sol if sol.exists() else None

    # match_marker stays default: rounds number "1." / "1)" normally. Add an
    # override here (mirroring UsamtsSeries) if a real run reveals a quirk.
