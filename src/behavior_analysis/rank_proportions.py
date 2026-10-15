"""Pure summaries of arm-rank choice proportions: per-subject and per-block means and errors.

Input is ``{subject: list[rank_count_dict]}`` as built by
:func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`:
one dict per block with ``best/second/third`` counts, ``total`` and the matching
``*_prop`` proportions. A "unit" below is ``{rank: value}`` for one mouse or one
block; the summaries are the mean and standard error across units.

Every row is a real block (``get_rank_counts_by_good_reversal`` appends no
summary row), so all rows of every subject are used throughout.

Each mean is paired with a standard error across the same units it averages:
blocks for an unpooled mouse, trials for a pooled mouse, mice or blocks for the
across-mouse and across-block summaries. NaN units count towards neither the
mean nor n, and fewer than two units give a NaN standard error, never 0.
"""

import warnings
from collections.abc import Hashable

import numpy as np

from src.behavior_analysis.stats import nan_standard_error

RANKS = ("best", "second", "third")

RankRow = dict[str, float]  # one count dict: best/second/third, total and the *_prop values
RankCounts = dict[Hashable, list[RankRow]]  # {subject (name or id): one count dict per block}
Unit = dict[str, float]  # {rank: value} for one mouse or block


def subject_rank_proportions(
    rows: list[RankRow], *, pool_trials_within_subject: bool = False
) -> Unit:
    """Return one proportion per rank while keeping the subject as the unit.

    Pooled, each rank's count is summed over rows and divided by the summed
    ``total`` (NaN if there are no trials); unpooled, it is the mean of the
    rows' ``*_prop`` values (NaN rows ignored).
    """
    if pool_trials_within_subject:
        total = sum(r.get("total", 0) for r in rows)
        if total <= 0:
            return {rank: np.nan for rank in RANKS}
        return {rank: sum(r.get(rank, 0) for r in rows) / total for rank in RANKS}
    return {rank: np.nanmean([r.get(f"{rank}_prop", np.nan) for r in rows]) for rank in RANKS}


def standard_error(values: list[float]) -> float:
    """Standard error of the mean over the non-NaN *values*.

    With fewer than two non-NaN values it is NaN and a ``RuntimeWarning`` is emitted.
    """
    column = np.asarray(values, dtype=float)[:, None]
    if np.sum(~np.isnan(column)) < 2:
        warnings.warn(
            "fewer than two non-NaN values: the standard error is NaN",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(nan_standard_error(column)[0])


def mean_by_rank(units: list[Unit], *, guard_empty: bool) -> list[float]:
    """Mean per rank (best, second, third) across *units*.

    With *guard_empty* an empty *units* gives NaN silently; without it numpy
    emits a ``RuntimeWarning``. The figures that warned before still warn,
    because the warning is the only sign that an across-mice SE is NaN, so
    callers choose per figure.
    """
    if guard_empty and not units:
        return [np.nan] * len(RANKS)
    return [np.nanmean([u[rank] for u in units]) for rank in RANKS]


def se_by_rank(units: list[Unit], *, guard_single: bool) -> list[float]:
    """Standard error per rank across *units* (see :func:`standard_error`).

    With *guard_single* a rank with fewer than two non-NaN units gets a NaN
    silently; without it :func:`standard_error` emits a ``RuntimeWarning`` for that
    rank. Callers choose per figure, because the warning is the only sign that an
    across-mice SE is NaN.
    """
    ses = []
    for rank in RANKS:
        values = [u[rank] for u in units]
        if guard_single and np.sum(~np.isnan(np.asarray(values, dtype=float))) < 2:
            ses.append(np.nan)
        else:
            ses.append(standard_error(values))
    return ses


def row_props(row: RankRow, *, strict: bool = False) -> list[float]:
    """The three ``*_prop`` values of a count row; *strict* raises on a missing key."""
    if strict:
        return [row[f"{rank}_prop"] for rank in RANKS]
    return [row.get(f"{rank}_prop", np.nan) for rank in RANKS]


def prop_units(rows: list[RankRow]) -> list[Unit]:
    """One ``{rank: proportion}`` unit per count row."""
    return [dict(zip(RANKS, row_props(row), strict=True)) for row in rows]


def pooled_standard_errors(rows: list[RankRow]) -> list[float]:
    """Per-rank SE across the pooled trials of *rows*, in :data:`RANKS` order.

    A pooled proportion is the mean over trials of a 0/1 indicator (the trial
    chose that rank), so its SE is that indicator's sample SD over the trials,
    ``sqrt(p * (1 - p) / (n - 1))``. NaN with fewer than two trials.

    Caveats: trials are treated as independent, but choices are serially
    correlated within a block and blocks can differ systematically, so this
    understates the uncertainty. It is therefore not comparable with the
    across-block SE of the unpooled bars, which measures block-to-block spread.
    For believed ranks (``get_rank_counts_by_value``) ties give fractional
    weights and ``total`` counts eligible choices, so the indicator is not 0/1;
    ``p * (1 - p)`` is then an upper bound of its variance and the formula is only
    approximate.
    """
    n = sum(r.get("total", 0) for r in rows)
    if n < 2:
        return [np.nan] * len(RANKS)
    proportions = [sum(r.get(rank, 0) for r in rows) / n for rank in RANKS]
    return [float(np.sqrt(p * (1 - p) / (n - 1))) for p in proportions]


def subject_rank_ses(
    rows: list[RankRow], *, pool_trials_within_subject: bool = False
) -> list[float]:
    """SE of :func:`subject_rank_proportions`, across the units that mean averages.

    Unpooled, the mean averages blocks, so this is the SE across the rows'
    ``*_prop`` values. Pooled, it averages trials, so this is the SE across trials
    (:func:`pooled_standard_errors`).
    """
    if pool_trials_within_subject:
        return pooled_standard_errors(rows)
    return se_by_rank(prop_units(rows), guard_single=True)


def proportions_by_mouse(
    counts: RankCounts, *, pool_trials_within_subject: bool = False
) -> dict[Hashable, Unit]:
    """``{subject: {rank: proportion}}`` for subjects that have any rows (all rows used)."""
    return {
        subject: subject_rank_proportions(
            rows, pool_trials_within_subject=pool_trials_within_subject
        )
        for subject, rows in counts.items()
        if rows
    }


def rows_in_block(counts: RankCounts, block_idx: int) -> list[RankRow]:
    """Each subject's row for *block_idx*, for the subjects that have that block."""
    return [rows[block_idx] for rows in counts.values() if block_idx < len(rows)]


def proportions_by_block(counts: RankCounts) -> list[Unit]:
    """One ``{rank: proportion}`` unit per block, averaged over the subjects that have it.

    Empty when there are no subjects or every subject has no rows.
    """
    block_rows = (rows_in_block(counts, b) for b in range(n_blocks(counts)))
    return [
        {rank: np.nanmean([r.get(f"{rank}_prop", np.nan) for r in rows]) for rank in RANKS}
        for rows in block_rows
        if rows
    ]


def n_blocks(counts: RankCounts) -> int:
    """The number of blocks of the longest subject (0 if there are no rows)."""
    return max((len(rows) for rows in counts.values()), default=0)
