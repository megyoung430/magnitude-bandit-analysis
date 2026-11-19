"""Choice probabilities around block changes early and late in training (Samborska Fig. 1f).

Samborska et al. (2022, Fig. 1f) show P(choose the new good port) around reversals in the first
and in a late problem: late in training the mice switch to the new best option faster. Here the
same curves come from the good reversals of each problem
(:func:`~src.behavior_analysis.reversals.get_good_reversal_info.get_good_reversal_info`, with the
trials after a bad reversal removed and the reversals whose best arm did not change left out, as
in the problem figures), and are compared between the three training stages of
:mod:`src.behavior_analysis.data.training_stages`: the first two structured problems, the first
two drifting problems and the last two drifting problems.

- **Per problem** (:func:`problem_reversal_curves`): each mouse's mean P(previous best),
  P(new best) and P(third) at every trial from ``-pre`` to ``post - 1`` around the block change
  (0 is the first trial of the new block), averaged over the problem's reversals.
- **Per stage** (:func:`stage_curves`): each mouse's curves averaged over the problems of a stage,
  each problem weighted equally.
- **Readouts** (:func:`adaptation_readouts`): per mouse and stage, the mean P(new best) and
  P(previous best) over trials ``ADAPT_START`` to ``ADAPT_STOP`` after the change (how fast the
  mouse leaves the old best arm and finds the new one), compared between stages with paired
  t-tests (:func:`~src.behavior_analysis.data.training_stages.stage_tests`).

Pure: session dicts and tables in, tables out.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.behavior_analysis.data.training_stages import PROBLEM_KEYS
from src.behavior_analysis.reversals.get_choice_probs_around_good_reversals import (
    NoReversalsError,
    get_choice_probs_around_good_reversals,
    remove_trials_after_bad_rev,
)
from src.behavior_analysis.reversals.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.reversals.get_total_reversals import get_reversal_indices_by_kind

PRE, POST = 10, 40
PREVIOUS_BEST, NEW_BEST, THIRD = "prev_best", "next_best", "third"
ROLES = (NEW_BEST, PREVIOUS_BEST, THIRD)
ADAPT_START, ADAPT_STOP = 1, 10  # trials after the change of the readout window, both included
CURVE_COLUMNS = ["mouse", "role", "offset", "probability", "n_reversals"]


def problem_reversal_curves(
    subjects_trials: Mapping[str, Mapping[str, Mapping]], pre: int = PRE, post: int = POST
) -> pd.DataFrame:
    """Each mouse's mean choice probabilities around the good reversals of one problem.

    Args:
        subjects_trials (Mapping): ``{mouse: {session_key: session_dict}}`` of one problem.
        pre (int): Trials before the change. Default ``PRE``.
        post (int): Trials from the change on (the first trial of the new block is 0). Default
            ``POST``.

    Returns:
        pd.DataFrame: ``CURVE_COLUMNS``: one row per mouse, role (``ROLES``) and offset; empty
        if no mouse has a good reversal whose best arm changed.
    """
    windows = get_good_reversal_info(subjects_trials, pre=pre, post=post)
    kinds = get_reversal_indices_by_kind(subjects_trials)
    windows = remove_trials_after_bad_rev(windows, kinds["good"], kinds["bad"])
    try:
        x, per_subject, _ = get_choice_probs_around_good_reversals(windows, pre=pre, post=post)
    except NoReversalsError:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    rows = [
        pd.DataFrame(
            dict(
                mouse=mouse,
                role=role,
                offset=x,
                probability=summary[f"{role}_mean"],
                n_reversals=summary["num_reversals"],
            )
        )  # fmt: skip
        for mouse, summary in per_subject.items()
        for role in ROLES
    ]
    return pd.concat(rows, ignore_index=True)[CURVE_COLUMNS]


def stage_curves(curves: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Each mouse's curves averaged over the problems of each stage.

    Args:
        curves (pd.DataFrame): :func:`problem_reversal_curves` of several problems, with
            ``phase`` and ``phase_problem`` added.
        stages (pd.DataFrame): :func:`~src.behavior_analysis.data.training_stages.training_stages`.

    Returns:
        pd.DataFrame: ``mouse, stage, role, offset, probability, n_problems, n_reversals``:
        ``n_problems`` counts the problems with a value at that offset, ``n_reversals`` sums
        their reversals.
    """
    staged = curves.merge(stages, on=PROBLEM_KEYS)
    grouped = staged.dropna(subset=["probability"]).groupby(
        ["mouse", "stage", "role", "offset"], sort=False
    )
    return grouped.agg(
        probability=("probability", "mean"),
        n_problems=("probability", "size"),
        n_reversals=("n_reversals", "sum"),
    ).reset_index()


def adaptation_readouts(
    curves: pd.DataFrame, start: int = ADAPT_START, stop: int = ADAPT_STOP
) -> pd.DataFrame:
    """Per mouse and stage: mean P(new best) and P(previous best) over trials *start* to *stop*.

    Args:
        curves (pd.DataFrame): :func:`stage_curves`.
        start (int): First trial after the change in the window. Default ``ADAPT_START``.
        stop (int): Last trial in the window (included). Default ``ADAPT_STOP``.

    Returns:
        pd.DataFrame: ``mouse, stage, role, probability`` for the roles new best and previous
        best.
    """
    window = curves[curves["offset"].between(start, stop) & curves["role"].isin(ROLES[:2])]
    return window.groupby(["mouse", "stage", "role"], as_index=False)["probability"].mean()


def nan_mean_se(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of the finite values (se NaN with fewer than two)."""
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return float(values.mean()), float(se)
