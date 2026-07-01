"""Mandelbrot: scaffold.

Mandelbrot has solutions, but the exact on-disk layout (a single solution PDF vs.
a solutions folder, and its naming relative to the test) still needs confirming.
Discovery inherits the default "one PDF per test"; `solution_source` is stubbed.
"""

from .base import Series


class MandelbrotSeries(Series):
    name = "mandelbrot"
    has_solutions = True

    def solution_source(self, test):
        # TODO: locate the Mandelbrot solution PDF/folder for `test` once the
        # naming convention is confirmed, mirroring UsamtsSeries.solution_source.
        return None
