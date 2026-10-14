"""Across-mouse summary of best/second/third choice proportions at good reversals.

Consumes the per-reversal proportions from
:func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`.
"""

import numpy as np

RANKS = ("best", "second", "third")


def mean_rank_proportions_by_subject(rank_counts_by_good_reversal: dict) -> dict:
    """Each subject's mean best/second/third proportion over its reversals.

    Subjects with no reversals are left out; a reversal missing a proportion
    counts as NaN, which the mean skips.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[reversal_dict]}`` with
            ``"best_prop"``, ``"second_prop"``, ``"third_prop"`` in each dict.

    Returns:
        ``{subject: {"best": float, "second": float, "third": float}}``.
    """
    return {
        subj: {
            rank: np.nanmean([r.get(f"{rank}_prop", float("nan")) for r in rows]) for rank in RANKS
        }
        for subj, rows in rank_counts_by_good_reversal.items()
        if rows
    }


def rank_mean_and_se(per_mouse: dict) -> tuple[list[float], list[float]]:
    """Across-mouse mean and standard error for each rank.

    NaNs are skipped in the mean and in the standard deviation.

    Known issue: the standard error divides by ``sqrt(len(per_mouse))``, which
    counts mice whose proportion is NaN although the standard deviation skips
    them, so the error is understated whenever a mouse is NaN. With a single
    mouse the sample standard deviation (``ddof=1``) is NaN, so the error is NaN.

    Args:
        per_mouse: ``{mouse: {"best", "second", "third"}}`` from
            :func:`mean_rank_proportions_by_subject`.

    Returns:
        ``(means, ses)``, each a list of three values in :data:`RANKS` order.
    """
    columns = [[v[rank] for v in per_mouse.values()] for rank in RANKS]
    means = [np.nanmean(col) for col in columns]
    ses = [np.nanstd(col, ddof=1) / np.sqrt(len(per_mouse)) for col in columns]
    return means, ses
