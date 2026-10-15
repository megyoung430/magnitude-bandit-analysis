"""Choice-probability curves around good reversals, per problem and averaged across problems.

No plotting. :mod:`src.behavior_visualization.plot_averaged_choice_probs` re-exports the
public functions for existing callers.
"""

from __future__ import annotations

import numpy as np

from src.behavior_analysis.get_choice_probs_around_good_reversals import nan_standard_error

# The three choice-probability curves computed around each good reversal.
CHOICE_KEYS = ("prev_best", "next_best", "third")


def _check_same_x(curves_by_problem: dict, probs: list) -> np.ndarray:
    """Return the shared x-array, raising if any problem's x differs from the first."""
    x0 = curves_by_problem[probs[0]]["x"]
    for p in probs[1:]:
        xp = curves_by_problem[p]["x"]
        if x0.shape != xp.shape or not np.allclose(x0, xp, equal_nan=True):
            raise ValueError(
                f"x-array mismatch between problems {probs[0]} and {p}. "
                "Ensure pre/post/skip_n are consistent."
            )
    return x0


def _problem_means(curves_by_problem: dict, probs: list, key: str) -> dict:
    """``{problem: mean curve for key}`` for the problems that have *key*."""
    means = {p: curves_by_problem[p]["across"]["mean"].get(key) for p in probs}
    return {p: np.asarray(y, dtype=float) for p, y in means.items() if y is not None}


def _mean_and_se_over_problems(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and standard error across the rows (problems) of *stacked*.

    The SE uses the number of non-NaN problems at each offset (NaN with fewer than
    two) and is zero for a single problem.
    """
    mean = np.nanmean(stacked, axis=0)
    se = nan_standard_error(stacked) if stacked.shape[0] > 1 else np.zeros_like(mean)
    return mean, se


def _problem_totals(curves_by_problem: dict, probs: list) -> tuple[int, int]:
    """Number of distinct subjects and total reversals summed over problems."""
    subjects: set = set()
    reversals = 0
    for p in probs:
        subjects |= curves_by_problem[p]["subjects"]
        reversals += int(curves_by_problem[p]["num_reversals"] or 0)
    return len(subjects), reversals


def compute_curve_from_windows(
    reversal_windows: dict,
    *,
    pre: int,
    post: int,
    skip_n: int = 0,
    do_moving_avg: bool = False,
    moving_avg_window: int = 4,
) -> tuple:
    """Compute choice-probability curves from a single problem's reversal windows.

    Args:
        reversal_windows: ``{subject: list[reversal_dict]}`` for one problem.
        pre: Number of trials before the reversal to include.
        post: Number of trials after the reversal to include.
        skip_n: Number of trials immediately after the reversal to mark as
            missing (default: 0).
        do_moving_avg: If ``True``, apply a centred moving average to the
            curves (default: False).
        moving_avg_window: Window size for the moving average (default: 4).

    Returns:
        ``(x, per_subject, across)`` as returned by
        :func:`src.behavior_analysis.get_choice_probs_around_good_reversals.get_choice_probs_around_good_reversals`.
    """
    from src.behavior_analysis.get_choice_probs_around_good_reversals import (
        apply_moving_average_to_choice_probs,
        get_choice_probs_around_good_reversals,
    )

    x, per_subject, across = get_choice_probs_around_good_reversals(
        reversal_windows,
        pre=pre,
        post=post,
        skip_n_trials_after_reversal=skip_n,
    )
    if do_moving_avg:
        x, per_subject, across = apply_moving_average_to_choice_probs(
            x, per_subject, moving_avg_window=moving_avg_window, mode="centered"
        )
    return x, per_subject, across


def curves_by_problem_from_windows_dict(
    windows_dict: dict,
    *,
    pre: int,
    post: int,
    skip_n: int = 0,
    do_moving_avg: bool = False,
    moving_avg_window: int = 4,
) -> dict:
    """Compute per-problem choice-probability curves from a dict of windows.

    Args:
        windows_dict: ``{problem: reversal_windows}``
        pre: Trials before reversal.
        post: Trials after reversal.
        skip_n: Trials to skip after reversal (default: 0).
        do_moving_avg: Apply moving average (default: False).
        moving_avg_window: Moving-average window size (default: 4).

    Returns:
        ``{problem: {"x": ..., "across": ..., "subjects": set, "num_reversals": int}}``
        for each problem that had at least one subject contributing data.
    """
    out: dict = {}
    for p, rw in windows_dict.items():
        if not rw:
            continue
        x, per_subject, across = compute_curve_from_windows(
            rw,
            pre=pre,
            post=post,
            skip_n=skip_n,
            do_moving_avg=do_moving_avg,
            moving_avg_window=moving_avg_window,
        )
        if across.get("num_subjects", 0) == 0:
            continue
        out[p] = {
            "x": np.asarray(x, float),
            "across": across,
            "subjects": set(per_subject.keys()),
            "num_reversals": int(across.get("num_reversals", 0) or 0),
        }
    return out


def average_across_problem_curves(curves_by_problem: dict) -> tuple:
    """Average per-problem choice-probability curves across problems.

    Each problem contributes equally (one replicate) to the mean.

    Args:
        curves_by_problem: Dict as returned by
            :func:`curves_by_problem_from_windows_dict`.

    Returns:
        ``(x, across_avg)`` where *across_avg* contains keys
        ``"mean"``, ``"se"``, ``"num_subjects"``, ``"num_reversals"``,
        ``"num_problems"``, and ``"problems_used"`` (only the problems that
        supplied at least one curve).

    Raises:
        ValueError: If *curves_by_problem* is empty or if the x-arrays do not
            match across problems.
    """
    probs = sorted(curves_by_problem.keys())
    if not probs:
        raise ValueError("No problems to average.")
    x0 = _check_same_x(curves_by_problem, probs)

    mean_out: dict = {}
    se_out: dict = {}
    contributing: set = set()
    for key in CHOICE_KEYS:
        means = _problem_means(curves_by_problem, probs, key)
        contributing |= means.keys()
        if not means:
            mean_out[key] = se_out[key] = None
        else:
            stacked = np.vstack(list(means.values()))
            mean_out[key], se_out[key] = _mean_and_se_over_problems(stacked)
    problems_used = [p for p in probs if p in contributing]

    num_subjects, num_reversals = _problem_totals(curves_by_problem, probs)
    across_avg = {
        "mean": mean_out,
        "se": se_out,
        "num_subjects": num_subjects,
        "num_reversals": num_reversals,
        "num_problems": len(problems_used),
        "problems_used": problems_used,
    }
    return x0, across_avg
