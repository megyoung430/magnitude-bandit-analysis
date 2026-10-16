"""Summaries of per-session / per-block trial counts for the cohort figures (no plotting).

The tables are those built by
:func:`src.behavior_analysis.trial_block_summary.build_trial_block_tables`: one row
per observed session or block, with at least ``mouse``, ``problem``, ``date`` and
``trials`` columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mice_by_subject_number(sessions: pd.DataFrame, blocks: pd.DataFrame) -> list[str]:
    """Mice seen in either table, ordered by subject number.

    Known issue: mice sharing a subject number come out in an arbitrary order that
    can change between runs (string hashes are salted per process).

    Known issue: a mouse that appears only in *blocks* has no subject number in
    *sessions*, so this raises a bare ``IndexError``.

    Args:
        sessions (pd.DataFrame): Session table with ``mouse`` and ``subject_number`` columns.
        blocks (pd.DataFrame): Block table with a ``mouse`` column.

    Returns:
        list[str]: Mouse ids sorted by subject number.
    """
    return sorted(
        set(sessions.mouse) | set(blocks.mouse),
        key=lambda m: int(sessions.loc[sessions.mouse == m, "subject_number"].iloc[0]),
    )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and their empirical cumulative fractions ``1/n .. 1``.

    Args:
        values (np.ndarray): 1-D array of numbers (non-empty).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x, y)``: *values* sorted ascending and the fraction of
        values at or below each, both of shape ``(n,)``.
    """
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def daily_mean_trials(table: pd.DataFrame) -> pd.DataFrame:
    """Mean trials per (mouse, date); several observations on one date are averaged.

    Args:
        table (pd.DataFrame): A session or block table with ``mouse``, ``date`` and ``trials``.

    Returns:
        pd.DataFrame: One row per mouse and date; columns ``mouse``, ``date``, ``trials``
        (the mean).
    """
    return table.groupby(["mouse", "date"], as_index=False).trials.mean()


def mean_trials_by_problem_and_mouse(table: pd.DataFrame) -> pd.DataFrame:
    """Each mouse's mean trials within each problem, so every mouse weighs equally.

    Args:
        table (pd.DataFrame): A session or block table with ``problem``, ``mouse`` and
            ``trials``.

    Returns:
        pd.DataFrame: One row per problem and mouse; columns ``problem``, ``mouse``, ``trials``
        (the mean).
    """
    return table.groupby(["problem", "mouse"], as_index=False).trials.mean()


def summarize_trials_by_problem(per_mouse: pd.DataFrame) -> pd.DataFrame:
    """Mean, SEM and number of mice (``n_mice``) per problem from per-mouse means.

    Args:
        per_mouse (pd.DataFrame): Output of :func:`mean_trials_by_problem_and_mouse`.

    Returns:
        pd.DataFrame: Indexed by ``problem``, with columns ``mean``, ``sem`` (NaN with a single
        mouse) and ``n_mice``.
    """
    return per_mouse.groupby("problem").trials.agg(mean="mean", sem="sem", n_mice="count")
