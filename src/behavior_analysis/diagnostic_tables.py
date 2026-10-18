"""The diagnostic tables of the analyses, stacked over problems for one file each.

The analysis entry points return DataFrames next to their figures instead of writing CSVs into
the figure folders. The cohort runner collects them per problem; :func:`stack_diagnostics` makes
one table per kind, with the problem each row came from:

- ``poke_trial_summary``: ``run_poke_analysis``'s ``summary``, one row per mouse and session;
- ``reversal_aligned_arms_counts``: ``run_reversal_aligned_arm_analysis``'s ``counts``;
- ``value_change_events``: ``run_value_change_analysis``'s ``contamination_table``;
- ``value_change_bin_counts``: its ``bin_counts_table``;
- ``value_change_symmetric_exclusions``: its ``symmetric_exclusions_table``.

Each row gets ``task_type`` and ``problem_in_type`` in front. A table with a ``mouse`` column
(the poke summary) also gets that mouse's ``overall_problem``, which differs between mice; the
others pool mice, so they carry no overall number.

All five kinds are always returned. A kind that no problem returned (the value analyses of a
cohort without drifting problems) has only its leading columns, so that writing it replaces a
file left by an earlier run instead of leaving it to look current.
"""

import pandas as pd

DIAGNOSTIC_TABLES = (
    "poke_trial_summary",
    "reversal_aligned_arms_counts",
    "value_change_events",
    "value_change_bin_counts",
    "value_change_symmetric_exclusions",
)


_LEADING_COLUMNS = {
    "poke_trial_summary": ("task_type", "problem_in_type", "overall_problem", "mouse"),
    "reversal_aligned_arms_counts": ("task_type", "problem_in_type"),
    "value_change_events": ("task_type", "problem_in_type"),
    "value_change_bin_counts": ("task_type", "problem_in_type"),
    "value_change_symmetric_exclusions": ("task_type", "problem_in_type"),
}


def _overall_problems(frame: pd.DataFrame, task_type: str, k: int, problem_map: pd.DataFrame):
    """The overall problem number of each row's mouse in problem *k* of *task_type*."""
    rows = problem_map[
        (problem_map["task_type"] == task_type) & (problem_map["problem_in_type"] == k)
    ]
    overall = dict(zip(rows["mouse"], rows["overall_problem"], strict=True))
    missing = sorted(set(frame["mouse"]) - set(overall))
    if missing:
        raise ValueError(f"{task_type} problem {k}: no entry in the problem map for {missing}")
    return frame["mouse"].map(overall)


def _with_problem(frame: pd.DataFrame, task_type: str, k: int, problem_map: pd.DataFrame):
    labelled = frame.copy()
    if "mouse" in labelled.columns:
        labelled.insert(0, "overall_problem", _overall_problems(frame, task_type, k, problem_map))
    labelled.insert(0, "problem_in_type", k)
    labelled.insert(0, "task_type", task_type)
    return labelled


def stack_diagnostics(by_problem: dict, problem_map: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One table per diagnostic kind, concatenated over the problems.

    Args:
        by_problem (dict): ``{(task_type, k): {kind: DataFrame}}``, ``kind`` one of
            :data:`DIAGNOSTIC_TABLES`; a problem may lack some kinds.
        problem_map (pd.DataFrame): The cohort's problem map (``mouse``, ``task_type``,
            ``problem_in_type``, ``overall_problem``).

    Returns:
        dict[str, pd.DataFrame]: ``{kind: stacked table}`` for every kind of
        :data:`DIAGNOSTIC_TABLES`, rows in the order of *by_problem*; a kind no problem returned
        has no rows and only its leading columns.

    Raises:
        ValueError: For a kind not in :data:`DIAGNOSTIC_TABLES`, or a mouse that the problem
            map does not have in that problem.
    """
    parts: dict[str, list[pd.DataFrame]] = {}
    for (task_type, k), tables in by_problem.items():
        for kind, frame in tables.items():
            if kind not in DIAGNOSTIC_TABLES:
                raise ValueError(f"unknown diagnostic table {kind!r}; expected {DIAGNOSTIC_TABLES}")
            parts.setdefault(kind, []).append(_with_problem(frame, task_type, k, problem_map))
    return {
        kind: pd.concat(parts[kind], ignore_index=True)
        if kind in parts
        else pd.DataFrame(columns=list(_LEADING_COLUMNS[kind]))
        for kind in DIAGNOSTIC_TABLES
    }
