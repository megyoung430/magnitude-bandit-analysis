"""Average choice-probability curves across problems (no plotting)."""

from __future__ import annotations

import numpy as np

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


def _stack_problem_means(curves_by_problem: dict, probs: list, key: str) -> np.ndarray | None:
    """Stack each problem's mean curve for *key* into ``(n_problems, T)``.

    Problems that lack *key* are left out; returns ``None`` if no problem has it.
    """
    rows = []
    for p in probs:
        y = curves_by_problem[p]["across"]["mean"].get(key)
        if y is not None:
            rows.append(np.asarray(y, dtype=float))
    return np.vstack(rows) if rows else None


def _mean_and_se_over_problems(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and standard error across the rows (problems) of *stacked*."""
    mean = np.nanmean(stacked, axis=0)
    # Known issue: the denominator is sqrt(n_problems) even where some problems
    # are NaN at that offset, so the SE is too small there.
    se = (
        np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        if stacked.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, se


def _problem_totals(curves_by_problem: dict, probs: list) -> tuple[int, int]:
    """Number of distinct subjects and total reversals summed over problems."""
    subjects: set = set()
    reversals = 0
    for p in probs:
        subjects |= curves_by_problem[p]["subjects"]
        reversals += int(curves_by_problem[p]["num_reversals"] or 0)
    return len(subjects), reversals


def average_across_problem_curves(curves_by_problem: dict) -> tuple:
    """Average per-problem choice-probability curves across problems.

    Each problem contributes equally (one replicate) to the mean.

    Args:
        curves_by_problem: Dict as returned by
            :func:`src.behavior_visualization.plot_averaged_choice_probs.curves_by_problem_from_windows_dict`.

    Returns:
        ``(x, across_avg)`` where *across_avg* contains keys
        ``"mean"``, ``"se"``, ``"num_subjects"``, ``"num_reversals"``,
        ``"num_problems"``, and ``"problems_used"``.

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
    for key in CHOICE_KEYS:
        stacked = _stack_problem_means(curves_by_problem, probs, key)
        if stacked is None:
            mean_out[key] = se_out[key] = None
        else:
            mean_out[key], se_out[key] = _mean_and_se_over_problems(stacked)

    num_subjects, num_reversals = _problem_totals(curves_by_problem, probs)
    across_avg = {
        "mean": mean_out,
        "se": se_out,
        "num_subjects": num_subjects,
        "num_reversals": num_reversals,
        "num_problems": len(probs),
        "problems_used": probs,
    }
    return x0, across_avg
