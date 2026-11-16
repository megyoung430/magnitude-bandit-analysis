"""Each mouse's training problems and stages, read from imported cohorts.

The analysis is :mod:`src.behavior_analysis.data.training_stages`; this module only turns
imported ``{overall problem: {mouse: sessions}}`` dicts into its problem table, and groups a
cohort's sessions by problem.
"""

from collections.abc import Mapping

import pandas as pd

from src.behavior_analysis.data import training_stages as stages
from src.behavior_import.task_types import number_within_task_type, regroup_by_task_type


def problem_table(trials_by_cohort: Mapping[int, Mapping]) -> pd.DataFrame:
    """One row per cohort, mouse and problem: ``mouse, cohort, task, k, overall_problem``.

    Args:
        trials_by_cohort (Mapping[int, Mapping]): ``{cohort id: {overall problem: {mouse:
            sessions}}}``, imported with the task types classified.

    Returns:
        pd.DataFrame: The problems of every mouse, ``task`` and ``k`` from
        ``number_within_task_type``.
    """
    rows = [
        dict(mouse=mouse, cohort=cohort, task=task, k=k, overall_problem=problem)
        for cohort, trials in trials_by_cohort.items()
        for (mouse, problem), (task, k) in number_within_task_type(trials).items()
    ]
    return pd.DataFrame(rows, columns=["mouse", "cohort", "task", "k", "overall_problem"])


def staged_problems(
    trials_by_cohort: Mapping[int, Mapping], max_drifting_by_cohort: dict
) -> pd.DataFrame:
    """The training problems of every mouse with their stage (problems outside a stage dropped).

    Args:
        trials_by_cohort (Mapping[int, Mapping]): As :func:`problem_table`.
        max_drifting_by_cohort (dict): As ``training_stages.training_problems``.

    Returns:
        pd.DataFrame: ``training_problems`` columns plus ``stage``.
    """
    problems = stages.training_problems(problem_table(trials_by_cohort), max_drifting_by_cohort)
    return problems.merge(stages.training_stages(problems), on=stages.PROBLEM_KEYS)


def sessions_by_problem(trials_by_problem: Mapping) -> dict:
    """``{(task, k): {mouse: sessions}}`` of one cohort (``regroup_by_task_type``)."""
    return regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
