"""Statistical tests comparing rank-choice proportions across subjects.

Functions here compute paired t-test p-values that compare mean choice
proportions (best vs second vs third arm) across subjects, using per-subject
averages as the unit of observation.
"""
import numpy as np
from scipy.stats import ttest_rel


def pvalue_paired_t_best_vs_second_vs_third(rank_counts_by_good_reversal):
    """Run paired t-tests comparing rank-choice proportions across all rank pairs.

    Averages each subject's ``best_prop``, ``second_prop``, and ``third_prop``
    across reversals, then runs three paired t-tests:
    best vs second, best vs third, and second vs third.

    Args:
        rank_counts_by_good_reversal: ``{subject: list[rank_count_dict]}`` as
            returned by
            :func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`.
            Each inner dict must contain ``"best_prop"``, ``"second_prop"``,
            and ``"third_prop"`` keys.

    Returns:
        Dict with keys ``"best_vs_second"``, ``"best_vs_third"``, and
        ``"second_vs_third"``, each mapping to the corresponding two-sided
        p-value (float).
    """
    best = []
    second = []
    third = []
    for subj, rows in rank_counts_by_good_reversal.items():
        if not rows:
            continue
        best.append(np.nanmean([r["best_prop"] for r in rows]))
        second.append(np.nanmean([r["second_prop"] for r in rows]))
        third.append(np.nanmean([r["third_prop"] for r in rows]))
    best = np.array(best, float)
    second = np.array(second, float)
    third = np.array(third, float)
    t_bs, p_bs = ttest_rel(best, second)
    t_bt, p_bt = ttest_rel(best, third)
    t_st, p_st = ttest_rel(second, third)
    return {
        "best_vs_second": p_bs,
        "best_vs_third": p_bt,
        "second_vs_third": p_st
    }

def pvalue_paired_t_new_vs_third(per_subject_counts, alternative="greater"):
    """Run a paired t-test comparing new-best vs third-arm first-leave proportions.

    Uses each subject's fraction of reversals where the animal first left the
    previous-best arm toward the new-best arm vs toward the third arm.  Handles
    the degenerate case where all differences are zero (returns 0.0 or 1.0
    deterministically).

    Args:
        per_subject_counts: ``{subject: {"new_best": int, "third": int, "total":
            int}}`` as returned by
            :func:`src.behavior_analysis.get_first_leave_after_good_reversals.get_first_leave_after_good_reversals`.
        alternative: Tail of the t-test — ``"greater"``, ``"less"``, or
            ``"two-sided"`` (default: ``"greater"``).

    Returns:
        Float p-value, or ``numpy.nan`` if fewer than 2 subjects have valid
        data.
    """
    new_vals = []
    third_vals = []

    for subj, c in per_subject_counts.items():
        tot = c.get("total", 0)
        if tot is None or tot <= 0:
            continue
        new_vals.append(c["new_best"] / tot)
        third_vals.append(c["third"] / tot)

    new_vals = np.asarray(new_vals, dtype=float)
    third_vals = np.asarray(third_vals, dtype=float)

    mask = np.isfinite(new_vals) & np.isfinite(third_vals)
    new_vals = new_vals[mask]
    third_vals = third_vals[mask]

    if new_vals.size < 2:
        return np.nan

    diffs = new_vals - third_vals
    if np.isclose(np.std(diffs, ddof=1), 0.0):
        return 1.0 if np.mean(diffs) <= 0 else 0.0

    result = ttest_rel(new_vals, third_vals, alternative=alternative)
    return float(result.pvalue)