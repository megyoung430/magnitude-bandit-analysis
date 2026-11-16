"""Load structured and drifting problems, fit the choice logit per problem, save tables and figures.

The analysis is :mod:`src.behavior_analysis.choice_logit.logit_by_problem`; the figures are
:mod:`src.behavior_visualization.choice_logit.plot_logit_by_problem`. Cohorts are loaded
cache-first, as in :func:`src.pipelines.choice_logit.choice_logit_values.load_drifting_problems`:
the newest import-cache snapshot is used without reading the data share.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import logit_by_problem as lbp
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_import.task_types import number_within_task_type, regroup_by_task_type
from src.behavior_visualization.choice_logit import plot_logit_by_problem as plots
from src.pipelines.common.all_cohorts_drifting import check_cohort_ids
from src.pipelines.common.cohort_analysis import load_cohort


def cohort_trial_table(cohort: int, trials_by_problem: dict) -> pd.DataFrame:
    """One cohort's trials of both task types, each problem numbered within its type and overall.

    Args:
        cohort (int): The cohort id, written into ``cohort`` and the ``problem`` key.
        trials_by_problem (dict): ``{overall problem: {mouse: sessions}}`` as imported with the
            task types classified.

    Returns:
        pd.DataFrame: The columns of :func:`drifting_trial_table` (``problem`` is ``(cohort,
        task, k)``) plus ``cohort``, ``task``, ``k`` and ``overall_problem``.
    """
    numbering = number_within_task_type(trials_by_problem)
    overall = {(mouse, *typed): problem for (mouse, problem), typed in numbering.items()}
    tables = []
    for (task, k), subjects in regroup_by_task_type(trials_by_problem, numbering).items():
        table = drifting_trial_table({(cohort, task, k): subjects})
        tables.append(table.assign(cohort=cohort, task=task, k=k))
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    keys = zip(out["mouse"], out["task"], out["k"], strict=True)
    return out.assign(overall_problem=[overall[key] for key in keys])


def load_training_trial_table(
    cohort_ids: Sequence[int], data_root: Path | None = None, trials_by_cohort: dict | None = None
) -> pd.DataFrame:
    """The trials of both task types of *cohort_ids*, stacked (see :func:`cohort_trial_table`).

    Args:
        cohort_ids (Sequence[int]): Cohorts to load, e.g. ``[1, 3]``.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``. Default ``None``:
            the ``BANDIT_DATA_ROOT`` setting.
        trials_by_cohort (dict | None): Already-loaded ``{cohort id: {overall problem: {mouse:
            sessions}}}``; skips loading. Default ``None``.

    Returns:
        pd.DataFrame: All cohorts' trials.
    """
    ids = check_cohort_ids(cohort_ids)
    if trials_by_cohort is None:
        trials_by_cohort = {
            c: load_cohort(c, data_root, use_cache=True, from_cache=True).trials_by_problem
            for c in ids
        }
    return pd.concat([cohort_trial_table(c, trials_by_cohort[c]) for c in ids], ignore_index=True)


class LogitByProblem(NamedTuple):
    """What :func:`run_logit_by_problem` computed (the files are already written).

    Attributes:
        weights (pd.DataFrame): :func:`logit_by_problem.fit_by_problem`.
        summaries (pd.DataFrame): :func:`logit_by_problem.weight_summaries`.
        trends (pd.DataFrame): :func:`logit_by_problem.phase_trends`.
        tests (pd.DataFrame): :func:`logit_by_problem.trend_tests`.
    """

    weights: pd.DataFrame
    summaries: pd.DataFrame
    trends: pd.DataFrame
    tests: pd.DataFrame


def run_logit_by_problem(
    trials: pd.DataFrame,
    max_drifting_by_cohort: dict,
    out_dir: str | Path,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
) -> LogitByProblem:
    """Fit every problem, test the trends and write the tables and figures into *out_dir*.

    Args:
        trials (pd.DataFrame): :func:`load_training_trial_table`.
        max_drifting_by_cohort (dict): As :func:`logit_by_problem.training_trials`.
        out_dir (str | Path): Folder of the figures; the tables go into its ``tables/``.
        n_lags (int): Reward and choice lags. Default ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty weight. Default ``choice_logit.DEFAULT_L2``.

    Returns:
        LogitByProblem: The tables.
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    selected = lbp.training_trials(trials, max_drifting_by_cohort)
    weights = lbp.fit_by_problem(selected, n_lags, l2)
    summaries = lbp.weight_summaries(weights)
    trends = lbp.phase_trends(summaries)
    tests = lbp.trend_tests(trends)
    for name, table in dict(weights=weights, summaries=summaries, trends=trends,
                            tests=tests).items():  # fmt: skip
        table.to_csv(tables_dir / f"logit_by_problem_{name}.csv", index=False)
    plots.plot_logit_by_problem_figures(weights, summaries, tests, out_dir, n_lags=n_lags, l2=l2)
    return LogitByProblem(weights, summaries, trends, tests)
