"""Across-mouse summary of best/second/third choice proportions at good reversals.

Consumes the per-reversal proportions from
:func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`.
"""

import warnings

import numpy as np

RANKS = ("best", "second", "third")


def nan_standard_error(stack: np.ndarray) -> np.ndarray:
    """Per-column standard error of the mean over the non-NaN rows of *stack*.

    NaN in columns with fewer than two non-NaN rows (the sample SD needs two).
    """
    n_valid = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        # nanstd warns for columns with < 2 values; those columns are set to NaN below.
        warnings.filterwarnings(
            "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
        )
        se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_valid)
    return np.where(n_valid >= 2, se, np.nan)


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

    NaNs are skipped in the mean, the standard deviation and the count, so a
    mouse without data for a rank does not count towards that rank's n. A rank
    with fewer than two non-NaN mice has a NaN standard error (with a
    ``RuntimeWarning``), never 0.

    Args:
        per_mouse: ``{mouse: {"best", "second", "third"}}`` from
            :func:`mean_rank_proportions_by_subject`.

    Returns:
        ``(means, ses)``, each a list of three values in :data:`RANKS` order.
    """
    columns = [[v[rank] for v in per_mouse.values()] for rank in RANKS]
    means = [np.nanmean(col) for col in columns]
    ses = [float(nan_standard_error(np.asarray(col, dtype=float)[:, None])[0]) for col in columns]
    if np.any(np.isnan(ses)):
        # the NaN error is drawn as a bar without an error bar; the warning says why
        warnings.warn(
            "fewer than two non-NaN mice for a rank: its standard error is NaN",
            RuntimeWarning,
            stacklevel=2,
        )
    return means, ses
