import numpy as np
from scipy.stats import ttest_rel

def pvalue_paired_t_best_vs_second_vs_third(rank_counts_by_good_reversal):
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