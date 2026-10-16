"""Statistics behind the two-cohort comparison figures (no plotting, no file output)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_good_reversal
from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.matched_trial_budget import match_trial_budget
from src.behavior_analysis.stats import nan_standard_error
from src.behavior_analysis.trial_block_summary import build_trial_block_tables

RANKS = ("best", "second", "third")


def require_problems(label: str, data: dict, problems) -> None:
    """Raise ``ValueError`` if cohort *label* lacks any of *problems*.

    Args:
        label (str): Cohort name, used in the error message.
        data (dict): The cohort's data, keyed by problem.
        problems (Iterable): Problems that must be keys of *data*.

    Raises:
        ValueError: If any of *problems* is not a key of *data*.
    """
    missing = set(problems) - set(data)
    if missing:
        raise ValueError(f"{label}: missing problems {sorted(missing)}")


def cohort_tables(data: dict, subject_numbers: dict, problems) -> dict:
    """Session and block tables for one cohort restricted to *problems*.

    Args:
        data (dict): ``{problem: {mouse: {session_key: session_dict}}}`` for the cohort.
        subject_numbers (dict): ``{mouse: subject number (int)}``.
        problems (Iterable): Problems to keep; each must be a key of *data*.

    Returns:
        dict: ``{"session": pd.DataFrame, "block": pd.DataFrame}`` as built by
        :func:`src.behavior_analysis.trial_block_summary.build_trial_block_tables`
        (one row per session and per block).
    """
    selected = {p: data[p] for p in problems}
    sessions, blocks = build_trial_block_tables(selected, subject_numbers)
    return {"session": sessions, "block": blocks}


def good_reversal_rank_counts(subjects: dict) -> dict:
    """Per-mouse, per-reversal rank counts, with the first block counted as a reversal.

    Args:
        subjects (dict): ``{mouse: {session_key: session_dict}}`` for one problem.

    Returns:
        dict: ``{mouse: list[rank_count_dict]}`` with one entry per reversal, as returned by
        :func:`src.behavior_analysis.get_rank_counts_by_good_reversal.get_rank_counts_by_good_reversal`
        (keys ``best``, ``second``, ``third``, ``total``, ``n_trials`` and ``*_prop``).
    """
    windows = get_good_reversal_info(subjects, include_first_block=True)
    return get_rank_counts_by_good_reversal(windows)


def rank_proportions(rows: list[dict]) -> tuple[int, list[float] | None]:
    """Pool a mouse's reversals into ``(eligible_trials, [best, second, third] proportions)``.

    The proportions are ``None`` when the mouse has no eligible trials.

    Args:
        rows (list[dict]): One rank-count dict per reversal of the mouse, each with
            ``"total"``, ``"best"``, ``"second"`` and ``"third"`` counts.

    Returns:
        tuple[int, list[float] | None]: ``(eligible_trials, proportions)``, where
        ``eligible_trials`` is the summed ``"total"`` and ``proportions`` is
        ``[best, second, third]`` summing to 1, or ``None`` if ``eligible_trials`` is 0.
    """
    n = sum(r["total"] for r in rows)
    if not n:
        return n, None
    return n, [sum(r[rank] for r in rows) / n for rank in RANKS]


def reversal_values(totals: dict) -> list[float]:
    """``[good, bad, all]`` reversal counts; good/bad are NaN when not recorded.

    Args:
        totals (dict): Output of
            :func:`src.behavior_analysis.get_total_reversals.get_total_reversals`;
            must contain ``"total_reversals"``, and may contain ``"good_reversals"``
            and ``"bad_reversals"``.

    Returns:
        list[float]: ``[good, bad, all]`` counts.
    """
    return [
        totals.get("good_reversals", np.nan),
        totals.get("bad_reversals", np.nan),
        totals["total_reversals"],
    ]


def rank_record(label: str, problem, mouse: str, eligible_trials: int, values: list[float]) -> dict:
    """CSV row for one mouse's pooled rank proportions (*values* are best, second, third).

    Args:
        label (str): Cohort name.
        problem (int): Problem number.
        mouse (str): Mouse identifier.
        eligible_trials (int): Number of trials the proportions are computed over.
        values (list[float]): ``[best, second, third]`` proportions.

    Returns:
        dict: Row with keys ``cohort``, ``problem``, ``mouse``, ``eligible_trials``,
        ``best``, ``second`` and ``third``.
    """
    return dict(
        cohort=label,
        problem=problem,
        mouse=mouse,
        eligible_trials=eligible_trials,
        **dict(zip(RANKS, values, strict=True)),
    )


def reversal_record(label: str, problem, mouse: str, totals: dict) -> dict:
    """CSV row for one mouse's reversal totals; *totals* keys become columns.

    Args:
        label (str): Cohort name.
        problem (int): Problem number.
        mouse (str): Mouse identifier.
        totals (dict): Output of
            :func:`src.behavior_analysis.get_total_reversals.get_total_reversals`.

    Returns:
        dict: Row with keys ``cohort``, ``problem``, ``mouse`` and every key of *totals*.
    """
    return dict(cohort=label, problem=problem, mouse=mouse, **totals)


class RankCell(NamedTuple):
    """One mouse's pooled rank proportions in a cohort-by-problem cell, with its CSV row.

    Attributes:
        proportions (list[float]): ``[best, second, third]`` proportions.
        record (dict): The matching row from :func:`rank_record`.
    """

    proportions: list[float]
    record: dict


class ReversalCell(NamedTuple):
    """One mouse's ``[good, bad, all]`` reversal counts in a cell, with its CSV row.

    Attributes:
        values (list[float]): ``[good, bad, all]`` reversal counts.
        record (dict): The matching row from :func:`reversal_record`.
    """

    values: list[float]
    record: dict


def mouse_rank_cell(label: str, problem, mouse: str, rows: list[dict]) -> RankCell | None:
    """Rank cell from a mouse's per-reversal rank counts; ``None`` without eligible trials.

    Args:
        label (str): Cohort name.
        problem (int): Problem number.
        mouse (str): Mouse identifier.
        rows (list[dict]): The mouse's per-reversal rank-count dicts, as for :func:`rank_proportions`.

    Returns:
        RankCell | None: The cell, or ``None`` if the mouse has no eligible trials.
    """
    n, proportions = rank_proportions(rows)
    if proportions is None:
        return None
    return RankCell(proportions, rank_record(label, problem, mouse, n, proportions))


def mouse_reversal_cell(label: str, problem, mouse: str, sessions: dict) -> ReversalCell:
    """Reversal cell for one mouse from its ``{session_key: session}`` data.

    Args:
        label (str): Cohort name.
        problem (int): Problem number.
        mouse (str): Mouse identifier.
        sessions (dict): ``{session_key: session_dict}`` for the mouse.

    Returns:
        ReversalCell: Counts from
        :func:`src.behavior_analysis.get_total_reversals.get_total_reversals` and their CSV row.
    """
    totals = get_total_reversals(sessions)
    return ReversalCell(reversal_values(totals), reversal_record(label, problem, mouse, totals))


def mean_and_sem(values: list[list[float]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean and SEM across mice for each of the three columns; ``None`` without mice.

    NaN entries are ignored (also in the count); a column with fewer than two
    values has a NaN SEM, never 0.

    Args:
        values (list[list[float]]): One ``[good, bad, all]``-style row per mouse; all
            rows have the same length.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: ``(mean, sem)``, each of shape
        ``(n_columns,)``, or ``None`` if *values* is empty.
    """
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return None
    mean = np.nanmean(arr, axis=0)
    return mean, nan_standard_error(arr)


def cap_problem_two(
    cohorts: dict, subject_numbers: dict, labels: list[str], trial_budget: int
) -> tuple[dict, list[dict]]:
    """Problem 2 of each cohort cut to its first *trial_budget* trials per mouse.

    Args:
        cohorts (dict): ``{label: {problem: {mouse: {session_key: session_dict}}}}``;
            each listed cohort must contain problem ``2``.
        subject_numbers (dict): ``{label: {mouse: subject number}}``.
        labels (list[str]): Cohort labels to process.
        trial_budget (int): Maximum trials kept per mouse.

    Returns:
        tuple[dict, list[dict]]: ``(capped, reports)``: ``{label: {2: subjects}}`` and
        one report row per mouse and cohort (the ``"mice"`` rows of
        :func:`src.behavior_analysis.matched_trial_budget.match_trial_budget` plus a
        ``cohort`` key).
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
