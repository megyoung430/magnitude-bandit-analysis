"""Port-category fractions per training stage: tables and the figure.

The analysis is :mod:`src.behavior_analysis.pokes.port_finding_by_stage`; the figure is
:mod:`src.behavior_visualization.pokes.plot_port_finding_by_stage`. The per-problem fractions
come from the port-finding summary of every problem
(:func:`~src.behavior_analysis.pokes.port_finding_summary.cohort_port_finding_summaries`), with
the port-finding-only sessions assigned from each cohort's import log.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.data.training_stages import stage_tests
from src.behavior_analysis.pokes import port_finding_by_stage as pfs
from src.behavior_analysis.pokes.port_finding_sessions import assign_port_finding_sessions
from src.behavior_analysis.pokes.port_finding_summary import (
    category_fraction_table,
    cohort_port_finding_summaries,
)
from src.behavior_visualization.pokes import plot_port_finding_by_stage as plots


class PortFindingByStage(NamedTuple):
    """What :func:`run_port_finding_by_stage` computed (the files are already written).

    Attributes:
        problem_fractions (pd.DataFrame): ``category_fraction_table`` of every problem.
        fractions (pd.DataFrame): :func:`port_finding_by_stage.stage_fractions`.
        tests (pd.DataFrame): ``stage_tests`` of the in-sequence fraction by kind and window.
    """

    problem_fractions: pd.DataFrame
    fractions: pd.DataFrame
    tests: pd.DataFrame


def problem_fractions(
    trials_by_cohort: Mapping[int, Mapping], port_finding_sessions: Mapping[int, Mapping]
) -> pd.DataFrame:
    """The port-category fractions of every mouse and problem of the cohorts.

    Args:
        trials_by_cohort (Mapping[int, Mapping]): ``{cohort id: {overall problem: {mouse:
            sessions}}}``.
        port_finding_sessions (Mapping[int, Mapping]): ``{cohort id: ImportLog
            .port_finding_sessions}`` (``{}`` for a cohort without them).

    Returns:
        pd.DataFrame: ``category_fraction_table`` with ``cohort, task_type, k`` labels.
    """
    entries = []
    for cohort, trials in trials_by_cohort.items():
        assignment = assign_port_finding_sessions(trials, port_finding_sessions[cohort])
        for (task, k), mice in cohort_port_finding_summaries(trials, assignment).items():
            labels = dict(cohort=cohort, task_type=task, k=k)
            entries += [(labels, summary) for summary in mice.values()]
    return category_fraction_table(entries)


def run_port_finding_by_stage(
    trials_by_cohort: Mapping[int, Mapping],
    port_finding_sessions: Mapping[int, Mapping],
    staged: pd.DataFrame,
    out_dir: str | Path,
) -> PortFindingByStage:
    """Compute the fractions per stage and write the tables and the figure.

    Args:
        trials_by_cohort (Mapping[int, Mapping]): As :func:`problem_fractions`.
        port_finding_sessions (Mapping[int, Mapping]): As :func:`problem_fractions`.
        staged (pd.DataFrame): ``training_stages.staged_problems``.
        out_dir (str | Path): Folder of the figure; the tables go into its ``tables/``.

    Returns:
        PortFindingByStage: The tables.
    """
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    per_problem = pfs.problem_keys(problem_fractions(trials_by_cohort, port_finding_sessions))
    fractions = pfs.stage_fractions(
        per_problem, staged[["mouse", "phase", "phase_problem", "stage"]]
    )
    in_sequence = pfs.in_sequence_fractions(fractions)
    tests = stage_tests(in_sequence, "fraction", by=["kind", "window"])
    result = PortFindingByStage(per_problem, fractions, tests)
    for name, table in result._asdict().items():
        table.to_csv(out_dir / "tables" / f"port_finding_by_stage_{name}.csv", index=False)
    plots.plot_port_finding_by_stage(fractions, tests, out_dir / plots.PORT_FINDING_BY_STAGE)
    return result
