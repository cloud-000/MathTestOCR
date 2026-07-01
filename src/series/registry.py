"""Series registry: maps a series name to its `Series` instance."""

from .mandelbrot import MandelbrotSeries
from .mathcounts import MathcountsSeries
from .purplecomet import PurpleCometSeries
from .usamts import UsamtsSeries

SERIES = {
    s.name: s
    for s in (UsamtsSeries(), PurpleCometSeries(), MandelbrotSeries(), MathcountsSeries())
}


def series_names():
    """Registered series names, for CLI choices."""
    return sorted(SERIES)


def get_series(name):
    """Return the `Series` for `name`, or raise KeyError with the valid options."""
    try:
        return SERIES[name]
    except KeyError:
        raise KeyError(
            f"unknown series {name!r}; choose from {', '.join(series_names())}"
        )
