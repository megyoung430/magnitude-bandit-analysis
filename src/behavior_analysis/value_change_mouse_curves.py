"""A tidy table of the per-mouse curves averaged in the mouse-weighted value-change plot.

:func:`mouse_curves_table` lays out, for one :class:`ValueChangeCurves`, every mouse's choice
probability per label and t (the arrays whose mean across mice the figure draws), how many of
the mouse's events have a trial at each t (the denominators of those probabilities) and the
mean across mice, so the plotted curve can be checked by eye. Pure: no file access.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis.value_change_windows import ValueChangeCurves

MEAN_ROW = "mean_across_mice"
N_EVENTS_LABEL = "n_events_at_t"
METADATA_COLUMNS = ("scope", "config", "mode", "mouse", "n_events", "label")


def t_column_name(t: float) -> str:
    """Signed column name of trial offset *t*, e.g. ``"t-5"``, ``"t+0"``, ``"t+15"``."""
    return f"t{int(t):+d}"


def mode_name(truncated: bool) -> str:
    """The truncation mode as written in the table's ``mode`` column."""
    return "truncated_at_rank_swap" if truncated else "untruncated"


def _rows_of_mouse(
    mouse: str,
    n_events: int,
    probabilities: dict[str, np.ndarray],
    n_events_at_t: np.ndarray,
    labels: tuple[str, ...],
) -> list[tuple[str, int, str, np.ndarray]]:
    """``(mouse, n_events, label, values per t)`` for each label, then the per-t event counts."""
    rows = [(mouse, n_events, label, probabilities[label]) for label in labels]
    return [*rows, (mouse, n_events, N_EVENTS_LABEL, n_events_at_t)]


def _check_mouse_weighted(curves: ValueChangeCurves) -> None:
    if curves.weighting != "subject":
        raise ValueError(
            f"the per-mouse table needs mouse-weighted curves (weighting='subject'), "
            f"got weighting={curves.weighting!r}"
        )
    if curves.per_subject_n_events is None or curves.per_subject_total_events is None:
        raise ValueError("curves lack per-mouse event counts; build them with summarize_windows")


def mouse_curves_table(curves: ValueChangeCurves, *, scope: str) -> pd.DataFrame:
    """One row per mouse and label, plus a ``mean_across_mice`` row per label.

    For each mouse the rows are its choice probability per label (the array that enters the
    mouse-weighted mean), then ``n_events_at_t``: how many of its events have a trial at each t.
    A probability is NaN where no event has a trial. The final rows hold the mean across mice,
    equal to what the mouse-weighted figure plots, and the events summed over mice per t.

    Args:
        curves (ValueChangeCurves): Mouse-weighted curves from ``summarize_windows``.
        scope (str): Name of the pooled problems, written in the ``scope`` column.

    Returns:
        pd.DataFrame: Columns ``scope``, ``config``, ``mode``, ``mouse``, ``n_events`` (the
        mouse's total events; all events on the mean rows), ``label`` and one column per t
        named like ``t-5``. Empty (columns only) when no mouse has an event.

    Raises:
        ValueError: If *curves* are event-weighted or lack the per-mouse event counts.
    """
    _check_mouse_weighted(curves)
    t_columns = [t_column_name(t) for t in curves.x]
    columns = [*METADATA_COLUMNS, *t_columns]
    rows = []
    for mouse, probabilities in curves.per_subject_mean.items():
        rows += _rows_of_mouse(
            mouse,
            curves.per_subject_total_events[mouse],
            probabilities,
            curves.per_subject_n_events[mouse],
            curves.labels,
        )
    if rows:
        rows += _rows_of_mouse(
            MEAN_ROW, curves.num_events, curves.mean, curves.n_events_per_bin, curves.labels
        )
    mode = mode_name(curves.truncate_at_rank_swap)
    records = [
        [scope, curves.config.label, mode, mouse, n_events, label, *values]
        for mouse, n_events, label, values in rows
    ]
    return pd.DataFrame(records, columns=columns)
