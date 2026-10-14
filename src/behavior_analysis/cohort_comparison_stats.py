"""Statistics behind the two-cohort comparison figures (no plotting, no file output)."""

from __future__ import annotations

import numpy as np

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_good_reversal
from src.behavior_analysis.matched_trial_budget import match_trial_budget
from src.behavior_analysis.trial_block_summary import build_trial_block_tables

RANKS = ("best", "second", "third")


def require_problems(label: str, data: dict, problems) -> None:
    """Raise ``ValueError`` if cohort *label* lacks any of *problems*."""
    missing = set(problems) - set(data)
    if missing:
        raise ValueError(f"{label}: missing problems {sorted(missing)}")


def cohort_tables(data: dict, subject_numbers: dict, problems) -> dict:
    """Session and block tables for one cohort restricted to *problems*."""
    selected = {p: data[p] for p in problems}
    sessions, blocks = build_trial_block_tables(selected, subject_numbers)
    return {"session": sessions, "block": blocks}


def good_reversal_rank_counts(subjects: dict) -> dict:
    """Per-mouse, per-reversal rank counts, with the first block counted as a reversal."""
    windows = get_good_reversal_info(subjects, include_first_block=True)
    return get_rank_counts_by_good_reversal(windows)


def rank_proportions(rows: list[dict]) -> tuple[int, list[float] | None]:
    """Pool a mouse's reversals into ``(eligible_trials, [best, second, third] proportions)``.

    The proportions are ``None`` when the mouse has no eligible trials.
    """
    n = sum(r["total"] for r in rows)
    if not n:
        return n, None
    return n, [sum(r[rank] for r in rows) / n for rank in RANKS]


def reversal_values(totals: dict) -> list[float]:
    """``[good, bad, all]`` reversal counts; good/bad are NaN when not recorded."""
    return [
        totals.get("good_reversals", np.nan),
        totals.get("bad_reversals", np.nan),
        totals["total_reversals"],
    ]


def rank_record(label: str, problem, mouse: str, eligible_trials: int, values: list[float]) -> dict:
    """CSV row for one mouse's pooled rank proportions (*values* are best, second, third)."""
    return dict(
        cohort=label,
        problem=problem,
        mouse=mouse,
        eligible_trials=eligible_trials,
        **dict(zip(RANKS, values, strict=True)),
    )


def reversal_record(label: str, problem, mouse: str, totals: dict) -> dict:
    """CSV row for one mouse's reversal totals; *totals* keys become columns."""
    return dict(cohort=label, problem=problem, mouse=mouse, **totals)


def mean_and_sem(values: list[list[float]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean and SEM across mice for each of the three columns; ``None`` without mice.

    NaN entries are ignored; the SEM is zero for a single mouse.
    """
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return None
    mean = np.nanmean(arr, axis=0)
    sem = (
        np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(arr), axis=0))
        if len(arr) > 1
        else np.zeros(3)
    )
    return mean, sem


def label_height(value: float, error: float) -> float:
    """Height for a mean's text label: just above the top of its error bar."""
    return value + (error if np.isfinite(error) else 0) + 0.035


def cap_problem_two(
    cohorts: dict, subject_numbers: dict, labels: list[str], trial_budget: int
) -> tuple[dict, list[dict]]:
    """Problem 2 of each cohort cut to its first *trial_budget* trials per mouse.

    Returns:
        ``(capped, reports)``: ``{label: {2: subjects}}`` and one report row per
        mouse and cohort.
    """
    capped: dict = {}
    reports: list[dict] = []
    for label in labels:
        data, report = match_trial_budget(
            cohorts[label][2], subject_numbers[label], trial_budget=trial_budget
        )
        capped[label] = {2: data}
        reports.extend(dict(cohort=label, **row) for row in report["mice"])
    return capped, reports
