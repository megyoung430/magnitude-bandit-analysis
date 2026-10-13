"""Pure summaries of arm-rank choice proportions, ready to plot.

Input is ``{subject: list[rank_count_dict]}`` as built by
:func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`:
one dict per block with ``best/second/third`` counts, ``total`` and the matching
``*_prop`` proportions. A "unit" below is ``{rank: value}`` for one mouse or one
block; the bars in the plots are the mean and standard error across units.
"""

import numpy as np

RANKS = ("best", "second", "third")


def subject_rank_proportions(rows, pool_trials_within_subject=False):
    """Return one proportion per rank while keeping the subject as the unit."""
    if pool_trials_within_subject:
        total = sum(r.get("total", 0) for r in rows)
        if total <= 0:
            return {rank: np.nan for rank in RANKS}
        return {rank: sum(r.get(rank, 0) for r in rows) / total for rank in RANKS}
    return {rank: np.nanmean([r.get(f"{rank}_prop", np.nan) for r in rows]) for rank in RANKS}


def standard_error(values):
    """Standard error of the mean; NaN for fewer than two values.

    NaNs are left out of the spread but still count towards n, so a unit with
    no data widens the denominator.
    """
    if len(values) < 2:
        return np.nan
    return np.nanstd(values, ddof=1) / np.sqrt(len(values))


def mean_and_se_by_rank(units):
    """Mean and standard error per rank (best, second, third) across *units*."""
    means = [np.nanmean([u[rank] for u in units]) if units else np.nan for rank in RANKS]
    ses = [standard_error([u[rank] for u in units]) for rank in RANKS]
    return means, ses


def row_props(row, *, strict=False):
    """The three ``*_prop`` values of a count row; *strict* raises on a missing key."""
    if strict:
        return [row[f"{rank}_prop"] for rank in RANKS]
    return [row.get(f"{rank}_prop", np.nan) for rank in RANKS]


def prop_units(rows):
    """One ``{rank: proportion}`` unit per count row."""
    return [dict(zip(RANKS, row_props(row))) for row in rows]


def mouse_block_rows(rows, pool_trials_within_subject=False):
    """Rows drawn as block lines for one mouse.

    The last row is dropped unless trials are pooled, matching the "last entry
    is a summary" convention of the plots' docstrings. Note that
    ``get_rank_counts_by_good_reversal`` appends no summary entry, so this
    drops a real block.
    """
    return rows if pool_trials_within_subject else rows[:-1]


def proportions_by_mouse(counts, pool_trials_within_subject=False):
    """``{subject: {rank: proportion}}`` for subjects that have any rows."""
    return {
        subject: subject_rank_proportions(rows, pool_trials_within_subject)
        for subject, rows in counts.items()
        if rows
    }


def rows_in_block(counts, block_idx):
    """Each subject's row for *block_idx*, excluding every subject's last row.

    Same convention (and caveat) as :func:`mouse_block_rows`.
    """
    return [rows[block_idx] for rows in counts.values() if block_idx < len(rows) - 1]


def proportions_by_block(counts):
    """One ``{rank: proportion}`` unit per block, averaged over the subjects that have it."""
    # max() raises ValueError when counts has subjects but every list is empty
    n_blocks = max((len(v) - 1) for v in counts.values() if len(v) > 0) if counts else 0
    block_rows = (rows_in_block(counts, b) for b in range(n_blocks))
    return [
        {rank: np.nanmean([r.get(f"{rank}_prop", np.nan) for r in rows]) for rank in RANKS}
        for rows in block_rows
        if rows
    ]


def n_plotted_blocks(counts):
    """Number of block panels: the longest subject list minus its last row."""
    return max((max(0, len(rows) - 1) for rows in counts.values()), default=0)
