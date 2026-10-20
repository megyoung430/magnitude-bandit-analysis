"""How many sessions each mouse needed to complete each drifting problem (no files, no plotting).

:func:`sessions_per_problem_table` counts, per mouse and drifting problem, the sessions with
bandit trials (the ``_part-NN`` parts of one session count once) and, in separate columns, the
port-finding-only sessions assigned to the problem
(:mod:`src.behavior_analysis.port_finding_sessions`) and the total.

A mouse's latest problem, of any task type, may still be running, so its row is marked
``complete=False``; every problem the mouse has moved on from is complete. A mouse whose latest
problem is blocked has only complete drifting problems.

:func:`summarize_sessions_per_problem` arranges a table for the figure: one series per mouse and
the mean and standard error across mice of the completed problems only (a running problem has
not reached its final count).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis.port_finding_sessions import (
    PortFindingAssignment,
    port_finding_sessions_for_problem,
)
from src.behavior_analysis.stats import mean_se_n
from src.behavior_import.session_order import ordered_session_keys, session_without_part
from src.behavior_import.task_types import DRIFTING, number_within_task_type

TABLE_COLUMNS = (
    "mouse",
    "task_type",
    "problem_in_type",
    "overall_problem",
    "n_sessions",
    "n_port_finding_only_sessions",
    "n_sessions_total",
    "first_session",
    "last_session",
    "complete",
)


class SessionsPerProblem(NamedTuple):
    """Sessions per drifting problem, arranged for plotting.

    ``mean``, ``se`` and ``n`` are across mice and use the completed problems only; ``mean`` and
    ``se`` are NaN where no (``se``: fewer than two) mouse has completed the problem.

    Attributes:
        mice (list[str]): Mouse ids, sorted.
        problems (list[int]): Drifting problem numbers (``problem_in_type``) on the x axis.
        by_mouse (dict[str, np.ndarray]): ``{mouse: float array over problems}``; NaN where the
            mouse did not run the problem.
        complete (dict[str, np.ndarray]): ``{mouse: bool array over problems}``; ``False`` for a
            running problem and where the mouse has no row.
        mean (np.ndarray): Float array over problems; mean across mice of completed problems.
        se (np.ndarray): Float array over problems; standard error across those mice.
        n (np.ndarray): Int array over problems; the number of mice behind ``mean``.
        counts_port_finding (bool): Whether the values include port-finding-only sessions.
    """

    mice: list[str]
    problems: list[int]
    by_mouse: dict[str, np.ndarray]
    complete: dict[str, np.ndarray]
    mean: np.ndarray
    se: np.ndarray
    n: np.ndarray
    counts_port_finding: bool


def _problem_row(
    mouse: str,
    overall_problem: int,
    k: int,
    sessions: dict,
    assignment: PortFindingAssignment | None,
    latest: int,
) -> dict:
    # dict.fromkeys drops the repeats the _part-NN segments of one session leave, keeping order
    keys = list(
        dict.fromkeys(session_without_part(key) for key in ordered_session_keys(mouse, sessions))
    )
    found = (
        port_finding_sessions_for_problem(assignment, {mouse: sessions}).get(mouse, {})
        if assignment is not None
        else {}
    )
    n_found = len({session_without_part(key) for key in found})
    return {
        "mouse": mouse,
        "task_type": DRIFTING,
        "problem_in_type": k,
        "overall_problem": overall_problem,
        "n_sessions": len(keys),
        "n_port_finding_only_sessions": n_found,
        "n_sessions_total": len(keys) + n_found,
        "first_session": keys[0],
        "last_session": keys[-1],
        "complete": overall_problem != latest,
    }


def sessions_per_problem_table(
    trials_by_problem: dict, port_finding: PortFindingAssignment | None = None
) -> pd.DataFrame:
    """One row per mouse and drifting problem with the number of sessions it took.

    Args:
        trials_by_problem (dict): ``{overall problem: {mouse: {session_key: session_dict}}}`` of
            one cohort with the task types classified (blocked problems too: they decide which
            problem is a mouse's latest).
        port_finding (PortFindingAssignment | None): Where the cohort's port-finding-only
            sessions go, from ``assign_port_finding_sessions``. Default ``None``: the cohort
            has none.

    Returns:
        pd.DataFrame: Columns :data:`TABLE_COLUMNS`, sorted by mouse then ``problem_in_type``;
        header only if the cohort has no drifting problem. ``n_sessions`` counts the sessions
        with bandit trials, ``n_port_finding_only_sessions`` those assigned to the problem and
        ``n_sessions_total`` their sum; ``first_session`` and ``last_session`` are keys of the
        bandit sessions (without ``_part-NN``); ``complete`` is ``False`` only for a mouse's
        latest problem overall.

    Raises:
        ValueError: If a session's task name is not recognised or the sessions of a mouse cannot
            be ordered (see ``number_within_task_type`` and ``ordered_session_keys``).
    """
    numbering = number_within_task_type(trials_by_problem)
    latest: dict[str, int] = {}
    for mouse, problem in numbering:
        latest[mouse] = max(latest.get(mouse, problem), problem)
    rows = [
        _problem_row(
            mouse, problem, k, trials_by_problem[problem][mouse], port_finding, latest[mouse]
        )
        for (mouse, problem), (task_type, k) in numbering.items()
        if task_type == DRIFTING
    ]
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    return table.sort_values(["mouse", "problem_in_type"], ignore_index=True)


def with_cohort(table: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """*table* with a ``cohort`` column in front, for stacking the tables of several cohorts.

    Args:
        table (pd.DataFrame): A :func:`sessions_per_problem_table`.
        cohort (str): The cohort's name, e.g. ``"cohort-03"``.

    Returns:
        pd.DataFrame: A copy with the extra first column.
    """
    return table.assign(cohort=cohort)[["cohort", *table.columns]]


def _check_table(table: pd.DataFrame, value_column: str) -> None:
    missing = [c for c in ("mouse", "problem_in_type", "complete", value_column) if c not in table]
    if missing:
        raise ValueError(f"sessions-per-problem table lacks the column(s) {missing}")
    if table.empty:
        raise ValueError("the sessions-per-problem table has no drifting problems to summarize")
    repeated = table[table.duplicated(["mouse", "problem_in_type"], keep=False)]
    if not repeated.empty:
        first = repeated.iloc[0]
        raise ValueError(
            f"{first['mouse']} has more than one row for drifting problem "
            f"{first['problem_in_type']}"
        )


def _mouse_series(
    rows: pd.DataFrame, problems: list[int], value_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(values, complete)`` of a mouse over *problems*; NaN and ``False`` where it has no row."""
    values = np.full(len(problems), np.nan)
    complete = np.zeros(len(problems), dtype=bool)
    position = {k: i for i, k in enumerate(problems)}
    for k, value, done in zip(
        rows["problem_in_type"], rows[value_column], rows["complete"], strict=True
    ):
        values[position[k]] = value
        complete[position[k]] = bool(done)
    return values, complete


def summarize_sessions_per_problem(
    table: pd.DataFrame, *, include_port_finding: bool = False, problems: list[int] | None = None
) -> SessionsPerProblem:
    """Per-mouse series and the across-mice mean and SE of the completed problems.

    Args:
        table (pd.DataFrame): A :func:`sessions_per_problem_table` (a stacked one may carry
            extra columns).
        include_port_finding (bool): Count ``n_sessions_total`` (bandit plus port-finding-only
            sessions) instead of ``n_sessions``. Default ``False``.
        problems (list[int] | None): The problem numbers of the axis; must include every problem
            of *table*. Default ``None``: the problems present in *table*, ascending.

    Returns:
        SessionsPerProblem: The series on the common problem axis.

    Raises:
        ValueError: If *table* is empty, lacks a column, has a mouse and problem twice, or
            *problems* misses a problem of *table*.
    """
    value_column = "n_sessions_total" if include_port_finding else "n_sessions"
    _check_table(table, value_column)
    present = sorted(int(k) for k in table["problem_in_type"].unique())
    if problems is None:
        problems = present
    elif not set(present) <= set(problems):
        raise ValueError(f"problems {problems} do not include all problems of the table {present}")
    mice = sorted(table["mouse"].unique())
    series = {m: _mouse_series(table[table["mouse"] == m], problems, value_column) for m in mice}
    by_mouse = {m: values for m, (values, _) in series.items()}
    complete = {m: done for m, (_, done) in series.items()}
    stats = [
        mean_se_n([by_mouse[m][i] for m in mice if complete[m][i]]) for i in range(len(problems))
    ]
    return SessionsPerProblem(
        mice=mice,
        problems=list(problems),
        by_mouse=by_mouse,
        complete=complete,
        mean=np.array([mean for mean, _, _ in stats]),
        se=np.array([se for _, se, _ in stats]),
        n=np.array([n for _, _, n in stats], dtype=int),
        counts_port_finding=include_port_finding,
    )
