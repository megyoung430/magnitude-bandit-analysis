"""Write the sessions-per-drifting-problem figures of a table (the edge between table and plots).

Both runners (:mod:`src.pipelines.cohort_analysis` for one cohort,
:mod:`src.pipelines.all_cohorts_drifting` for several) call
:func:`write_sessions_per_problem_figures` with the table they have already built, so the figures
are the same wherever they are drawn.
"""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.sessions_per_problem import summarize_sessions_per_problem
from src.behavior_visualization.plot_sessions_per_problem import (
    plot_sessions_per_problem,
    plot_sessions_per_problem_by_cohort,
)

SESSIONS_PER_PROBLEM_FOLDER = "sessions-per-problem"
TABLE_NAME = "sessions_per_problem"
FIGURE_NAME = "Sessions per Drifting Problem"
BY_COHORT_FIGURE_NAME = "Sessions per Drifting Problem by Cohort"


def _cohort_summaries(table: pd.DataFrame, include_port_finding: bool) -> dict:
    """``{cohort: summary}`` on the problem axis shared by all cohorts, in cohort order."""
    problems = sorted(int(k) for k in table["problem_in_type"].unique())
    return {
        cohort: summarize_sessions_per_problem(
            rows, include_port_finding=include_port_finding, problems=problems
        )
        for cohort, rows in table.groupby("cohort", sort=True)
    }


def write_sessions_per_problem_figures(
    table: pd.DataFrame,
    save_dir: Path,
    scope_label: str,
    *,
    mouse_legend: bool = True,
    include_port_finding: bool = False,
    by_cohort: bool = False,
) -> None:
    """Save the sessions-per-drifting-problem figure(s) of *table* into *save_dir*.

    Writes ``Sessions per Drifting Problem.{png,pdf}`` and, with *by_cohort*,
    ``Sessions per Drifting Problem by Cohort.{png,pdf}``.

    Args:
        table (pd.DataFrame): A ``sessions_per_problem_table``; for *by_cohort* a stack of them
            with a ``cohort`` column (``with_cohort``).
        save_dir (Path): Folder that receives the figures (created if missing).
        scope_label (str): The cohort(s) named in the captions, e.g. ``"cohort 3"``.
        mouse_legend (bool): Name the mice in the legend; their lines are drawn either way.
            Default True.
        include_port_finding (bool): Plot the total of bandit and port-finding-only sessions
            instead of the bandit sessions alone. Default False.
        by_cohort (bool): Also draw the figure of one mean line per cohort. Default False.
            Keyword-only.

    Raises:
        ValueError: If *table* has no drifting problems, or (with *by_cohort*) no ``cohort``
            column.
    """
    save_dir = Path(save_dir)
    plot_sessions_per_problem(
        summarize_sessions_per_problem(table, include_port_finding=include_port_finding),
        save_dir / FIGURE_NAME,
        scope_label,
        mouse_legend=mouse_legend,
    )
    if by_cohort:
        if "cohort" not in table:
            raise ValueError("a figure by cohort needs a 'cohort' column in the table")
        plot_sessions_per_problem_by_cohort(
            _cohort_summaries(table, include_port_finding),
            save_dir / BY_COHORT_FIGURE_NAME,
            scope_label,
        )
