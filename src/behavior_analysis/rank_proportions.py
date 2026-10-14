"""Pure summaries of arm-rank choice proportions: per-subject and per-block means and errors.

Input is ``{subject: list[rank_count_dict]}`` as built by
:func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`:
one dict per block with ``best/second/third`` counts, ``total`` and the matching
``*_prop`` proportions. A "unit" below is ``{rank: value}`` for one mouse or one
block; the summaries are the mean and standard error across units.

Known issue (shared by :func:`block_rows_for_subject`, :func:`rows_in_block`,
:func:`proportions_by_block` and :func:`n_blocks_excluding_last`): these helpers
treat each subject's last row as a "summary" entry and leave it out, but
``get_rank_counts_by_good_reversal`` appends no summary row. Every subject's last
real block is therefore missing from them, whereas :func:`proportions_by_mouse`
uses all rows.
"""

import numpy as np

RANKS = ("best", "second", "third")

RankRow = dict  # one count dict: best/second/third, total and the *_prop values
RankCounts = dict[str, list[dict]]  # {subject: one count dict per block}
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
    """Standard error of the mean, ``nanstd(ddof=1) / sqrt(len(values))``.

    Known issue: NaNs are left out of the spread but still count towards n, so
    a unit with no data widens the denominator and the error is understated.
    With fewer than two non-NaN values numpy returns NaN and emits a
    ``RuntimeWarning``.
    """
    return np.nanstd(values, ddof=1) / np.sqrt(len(values))


def mean_by_rank(units: list[Unit], *, guard_empty: bool) -> list[float]:
    """Mean per rank (best, second, third) across *units*.

    With *guard_empty* an empty *units* gives NaN silently; without it numpy
    emits its ``Mean of empty slice`` warning. The figures differ in which of
    the two they want, so callers choose.
    """
    if guard_empty and not units:
        return [np.nan] * len(RANKS)
    return [np.nanmean([u[rank] for u in units]) for rank in RANKS]


def se_by_rank(units: list[Unit], *, guard_single: bool) -> list[float]:
    """Standard error per rank across *units* (see :func:`standard_error`).

    With *guard_single* fewer than two units give NaN silently; without it numpy
    emits its ``Degrees of freedom <= 0`` warning. The figures differ in which
    of the two they want, so callers choose.
    """
    if guard_single and len(units) < 2:
        return [np.nan] * len(RANKS)
    return [standard_error([u[rank] for u in units]) for rank in RANKS]


def row_props(row: RankRow, *, strict: bool = False) -> list[float]:
    """The three ``*_prop`` values of a count row; *strict* raises on a missing key."""
    if strict:
        return [row[f"{rank}_prop"] for rank in RANKS]
    return [row.get(f"{rank}_prop", np.nan) for rank in RANKS]


def prop_units(rows: list[RankRow]) -> list[Unit]:
    """One ``{rank: proportion}`` unit per count row."""
    return [dict(zip(RANKS, row_props(row), strict=True)) for row in rows]


def block_rows_for_subject(
    rows: list[RankRow], *, pool_trials_within_subject: bool = False
) -> list[RankRow]:
    """The rows of one subject that count as its blocks.

    The last row is dropped unless trials are pooled, as if it were a summary
    entry. Known issue: ``get_rank_counts_by_good_reversal`` appends no summary
    entry, so this drops a real block.
    """
    return rows if pool_trials_within_subject else rows[:-1]


def proportions_by_mouse(
    counts: RankCounts, *, pool_trials_within_subject: bool = False
) -> dict[str, Unit]:
    """``{subject: {rank: proportion}}`` for subjects that have any rows (all rows used)."""
    return {
        subject: subject_rank_proportions(
            rows, pool_trials_within_subject=pool_trials_within_subject
        )
        for subject, rows in counts.items()
        if rows
    }


def rows_in_block(counts: RankCounts, block_idx: int) -> list[RankRow]:
    """Each subject's row for *block_idx*, leaving out every subject's last row.

    Known issue: same dropped last real block as :func:`block_rows_for_subject`.
    """
    return [rows[block_idx] for rows in counts.values() if block_idx < len(rows) - 1]


def proportions_by_block(counts: RankCounts) -> list[Unit]:
    """One ``{rank: proportion}`` unit per block, averaged over the subjects that have it.

    Known issue: raises ``ValueError: max() arg is an empty sequence`` when
    *counts* has subjects but every list is empty. Also leaves out each
    subject's last row (see :func:`rows_in_block`).
    """
    n_blocks = max((len(v) - 1) for v in counts.values() if len(v) > 0) if counts else 0
    block_rows = (rows_in_block(counts, b) for b in range(n_blocks))
    return [
        {rank: np.nanmean([r.get(f"{rank}_prop", np.nan) for r in rows]) for rank in RANKS}
        for rows in block_rows
        if rows
    ]


def n_blocks_excluding_last(counts: RankCounts) -> int:
    """The longest subject list minus its last row (0 if there are no rows)."""
    return max((max(0, len(rows) - 1) for rows in counts.values()), default=0)
