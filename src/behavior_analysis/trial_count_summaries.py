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
    """
    return sorted(
        set(sessions.mouse) | set(blocks.mouse),
        key=lambda m: int(sessions.loc[sessions.mouse == m, "subject_number"].iloc[0]),
    )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and their empirical cumulative fractions ``1/n .. 1``."""
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def daily_mean_trials(table: pd.DataFrame) -> pd.DataFrame:
    """Mean trials per (mouse, date); several observations on one date are averaged."""
    return table.groupby(["mouse", "date"], as_index=False).trials.mean()


def mean_trials_by_problem_and_mouse(table: pd.DataFrame) -> pd.DataFrame:
    """Each mouse's mean trials within each problem, so every mouse weighs equally."""
    return table.groupby(["problem", "mouse"], as_index=False).trials.mean()


def summarize_trials_by_problem(per_mouse: pd.DataFrame) -> pd.DataFrame:
    """Mean, SEM and number of mice (``n_mice``) per problem from per-mouse means."""
    return per_mouse.groupby("problem").trials.agg(mean="mean", sem="sem", n_mice="count")
