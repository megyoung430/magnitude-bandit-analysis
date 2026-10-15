"""Choice probabilities against option values: heatmap tables and psychometric curves.

Everything here is pure: it takes the per-cell counts of
:func:`src.behavior_analysis.get_choice_probs_by_value.count_choice_probs` (one
:class:`MouseCounts` per mouse) and returns arrays and DataFrames. Mice are the replicates:
every mean and standard error is taken across mice, never across trials.

Two weightings give a mouse's value at one x (e.g. best value 4):

- ``pool_trials=True``: the mouse's trials in all cells sharing that x are pooled,
  ``p = chosen / trials``, so a cell with more trials counts for more;
- ``pool_trials=False``: the mouse's cell probabilities are averaged with equal weight, so
  a cell with few trials counts as much as one with many.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis.get_choice_probs_by_value import (
    BEST_VALS,
    SECOND_VALS,
    MouseCounts,
    choice_probs_from_counts,
    init_matrix,
)
from src.behavior_analysis.stats import nan_mean, nan_standard_error

CHOICES = ("best", "second")
X_AXES = ("best_val", "second_val", "diff")

# Cells in the order of ``DataFrame.to_numpy().ravel()`` of an ``init_matrix`` grid.
_CELLS = [(b, s) for b in BEST_VALS for s in SECOND_VALS]
# best - second; the cell (best 2, second 3) has best below second, which cannot occur.
_DIFFS = sorted({b - s for b, s in _CELLS if b >= s})


@dataclass(frozen=True)
class Curve:
    """One psychometric curve: a value per mouse and x, and the mean and SE across mice.

    ``per_mouse`` is ``(n_mice, n_x)`` with NaN where a mouse has no trials at that x.
    ``se`` is NaN where fewer than two mice have a value. ``n_mice`` counts the mice with a
    value at each x and ``n_trials`` the trials of all mice there.
    """

    x: np.ndarray
    mice: list[str]
    per_mouse: np.ndarray
    mean: np.ndarray
    se: np.ndarray
    n_mice: np.ndarray
    n_trials: np.ndarray


class HeatmapTables(NamedTuple):
    """Cell probabilities and trial counts (best value x second value) for the heatmaps.

    The ``p_*`` and ``n_trials`` tables are over all mice: the mean of the mice's cell
    probabilities (every mouse equal, empty cells ignored) and the summed trial counts.
    """

    p_best: pd.DataFrame
    p_second: pd.DataFrame
    n_trials: pd.DataFrame
    per_mouse_p_best: dict[str, pd.DataFrame]
    per_mouse_p_second: dict[str, pd.DataFrame]
    per_mouse_n: dict[str, pd.DataFrame]


def _membership(x_values: list, x_of_cell) -> np.ndarray:
    """``(n_x, 9)`` 0/1 matrix: row i marks the cells whose x is ``x_values[i]``."""
    return np.array([[float(x_of_cell(b, s) == x) for b, s in _CELLS] for x in x_values])


_MEMBERSHIP = {
    "best_val": (BEST_VALS, _membership(BEST_VALS, lambda b, s: b)),
    "second_val": (SECOND_VALS, _membership(SECOND_VALS, lambda b, s: s)),
    "diff": (_DIFFS, _membership(_DIFFS, lambda b, s: b - s)),
}


def _check_no_impossible_cell(counts: MouseCounts, subject: str) -> None:
    """Raise if a mouse has trials in the cell with best below second."""
    if counts.denom.loc[2, 3] > 0:
        raise ValueError(
            f"{subject}: {counts.denom.loc[2, 3]:g} trials have best value 2 below second "
            "value 3, which cannot occur; the best and second values are mislabelled"
        )


def _mouse_values(
    member: np.ndarray, numerator: np.ndarray, denom: np.ndarray, *, pool_trials: bool
) -> np.ndarray:
    """One mouse's choice probability at each x, NaN where it has no trials there.

    Args:
        member (np.ndarray): ``(n_x, 9)`` cell membership of each x.
        numerator (np.ndarray): Flattened chosen counts per cell.
        denom (np.ndarray): Flattened trial counts per cell.
        pool_trials (bool): Pool the trials of the cells sharing an x, or average their
            probabilities with equal weight.

    Returns:
        np.ndarray: ``(n_x,)`` probabilities.
    """
    if pool_trials:
        chosen, trials = member @ numerator, member @ denom
        return np.where(trials > 0, chosen / np.maximum(trials, 1), np.nan)
    has_trials = denom > 0
    cell_p = np.where(has_trials, numerator / np.maximum(denom, 1), 0.0)
    n_cells = member @ has_trials.astype(float)
    return np.where(n_cells > 0, (member @ cell_p) / np.maximum(n_cells, 1), np.nan)


def _curve(
    counts_by_mouse: dict[str, MouseCounts], choice: str, x_axis: str, *, pool_trials: bool
) -> Curve:
    """The :class:`Curve` of *choice* ("best"/"second") against *x_axis*."""
    x_values, member = _MEMBERSHIP[x_axis]
    mice = sorted(counts_by_mouse)
    values, trials = [], []
    for subject in mice:
        counts = counts_by_mouse[subject]
        _check_no_impossible_cell(counts, subject)
        numerator = getattr(counts, f"num_{choice}").to_numpy(dtype=float).ravel()
        denom = counts.denom.to_numpy(dtype=float).ravel()
        values.append(_mouse_values(member, numerator, denom, pool_trials=pool_trials))
        trials.append(member @ denom)
    per_mouse = np.array(values).reshape(len(mice), len(x_values))
    n_trials = np.array(trials).reshape(len(mice), len(x_values)).sum(axis=0)
    se = nan_standard_error(per_mouse) if mice else np.full(len(x_values), np.nan)
    return Curve(
        x=np.array(x_values),
        mice=mice,
        per_mouse=per_mouse,
        mean=nan_mean(per_mouse, axis=0),
        se=se,
        n_mice=np.sum(~np.isnan(per_mouse), axis=0),
        n_trials=n_trials,
    )


def psychometric_curves(
    counts_by_mouse: dict[str, MouseCounts], *, pool_trials: bool
) -> dict[str, dict[str, Curve]]:
    """P(best) and P(second) against best value, second value and best - second value.

    The three x axes use the same trials: every cell with best >= second (ties, where best
    equals second, are on the difference axis at 0 as well as on both value axes).

    Args:
        counts_by_mouse (dict[str, MouseCounts]): Output of ``count_choice_probs``.
        pool_trials (bool): Within each mouse, pool trials across the cells sharing an x
            (``True``) or average the cell probabilities equally (``False``). Mice are the
            replicates for the mean and standard error either way.

    Returns:
        dict[str, dict[str, Curve]]: ``curves[choice][x_axis]`` for choice in
        ``("best", "second")`` and x_axis in ``("best_val", "second_val", "diff")``.

    Raises:
        ValueError: If a mouse has trials in the impossible cell best 2 / second 3.
    """
    return {
        choice: {
            axis: _curve(counts_by_mouse, choice, axis, pool_trials=pool_trials)
            for axis in X_AXES
        }
        for choice in CHOICES
    }


def heatmap_tables(counts_by_mouse: dict[str, MouseCounts]) -> HeatmapTables:
    """Cell probabilities and trial counts for the overall and per-mouse heatmaps.

    Args:
        counts_by_mouse (dict[str, MouseCounts]): Output of ``count_choice_probs``.

    Returns:
        HeatmapTables: Probabilities are NaN, and trial counts 0, in cells with no trials.
    """
    p_best, p_second, per_mouse_best, per_mouse_second = choice_probs_from_counts(counts_by_mouse)
    per_mouse_n = {subject: counts.denom.copy() for subject, counts in counts_by_mouse.items()}
    total = init_matrix(fill=0.0)
    for n in per_mouse_n.values():
        total = total + n
    return HeatmapTables(
        p_best, p_second, total, per_mouse_best, per_mouse_second, per_mouse_n
    )
