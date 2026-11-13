"""The shape every rank test returns: scalar readouts to calibrate and curves to draw."""

from typing import NamedTuple

import numpy as np
import pandas as pd

CURVE_COLUMNS = ["curve", "series", "x", "y", "n"]


class RankResult(NamedTuple):
    """What one rank test measures on one mouse (or one replay of it).

    Attributes:
        readouts (dict[str, float]): ``{name: value}``, each a scalar the null calibration can
            compare with replays; NaN where the data cannot give it.
        curves (pd.DataFrame): ``curve, series, x, y, n`` rows to draw (``n`` is the count behind
            ``y``); ``series`` names a line within a ``curve`` (``""`` for one).
    """

    readouts: dict
    curves: pd.DataFrame


def curve_rows(curve: str, x, y, n, series: str = "") -> pd.DataFrame:
    """A table of curve points: one row per ``x``."""
    x = np.atleast_1d(x)
    return pd.DataFrame(
        dict(curve=curve, series=series, x=x, y=np.atleast_1d(y), n=np.atleast_1d(n)),
        columns=CURVE_COLUMNS,
    )


def merge(*results: RankResult) -> RankResult:
    """Combine results: readouts joined (names must differ), curves stacked."""
    readouts: dict = {}
    for result in results:
        clash = readouts.keys() & result.readouts.keys()
        if clash:
            raise ValueError(f"readouts named twice: {sorted(clash)}")
        readouts |= result.readouts
    curves = [r.curves for r in results if len(r.curves)]
    return RankResult(readouts, pd.concat(curves, ignore_index=True) if curves else empty_curves())


def empty_curves() -> pd.DataFrame:
    """A curve table without rows."""
    return pd.DataFrame(columns=CURVE_COLUMNS)
