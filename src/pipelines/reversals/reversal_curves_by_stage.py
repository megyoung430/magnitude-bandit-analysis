"""Choice probabilities around block changes per training stage: tables and the figure.

The analysis is :mod:`src.behavior_analysis.reversals.reversal_curves_by_stage`; the figure is
:mod:`src.behavior_visualization.reversals.plot_reversal_curves_by_stage`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.data.training_stages import TASK_PHASE, stage_tests
from src.behavior_analysis.reversals import reversal_curves_by_stage as rcs
from src.behavior_visualization.reversals import plot_reversal_curves_by_stage as plots
from src.pipelines.common.training_stages import sessions_by_problem


class ReversalCurvesByStage(NamedTuple):
    """What :func:`run_reversal_curves_by_stage` computed (the files are already written).

    Attributes:
        curves (pd.DataFrame): :func:`reversal_curves_by_stage.stage_curves`.
        readouts (pd.DataFrame): :func:`reversal_curves_by_stage.adaptation_readouts`.
        tests (pd.DataFrame): ``stage_tests`` of the readouts by role.
    """

    curves: pd.DataFrame
    readouts: pd.DataFrame
    tests: pd.DataFrame


def staged_problem_curves(
    trials_by_cohort: Mapping[int, Mapping], staged: pd.DataFrame
) -> pd.DataFrame:
    """:func:`reversal_curves_by_stage.problem_reversal_curves` of every staged problem.

    Args:
        trials_by_cohort (Mapping[int, Mapping]): ``{cohort id: {overall problem: {mouse:
            sessions}}}``.
        staged (pd.DataFrame): ``training_stages.staged_problems`` (``cohort, mouse, task, k,
            phase, phase_problem, stage``).

    Returns:
        pd.DataFrame: The curves, with ``cohort, phase, phase_problem`` added; only the mice
        whose problem is staged.
    """
    curves = []
    for cohort, trials in trials_by_cohort.items():
        mine = staged[staged["cohort"] == cohort]
        for (task, k), subjects in sessions_by_problem(trials).items():
            mice = set(mine.loc[(mine["task"] == task) & (mine["k"] == k), "mouse"])
            if not mice:
                continue
            table = rcs.problem_reversal_curves({m: subjects[m] for m in sorted(mice)})
            curves.append(table.assign(cohort=cohort, phase=TASK_PHASE[task], phase_problem=k))
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def run_reversal_curves_by_stage(
    trials_by_cohort: Mapping[int, Mapping], staged: pd.DataFrame, out_dir: str | Path
) -> ReversalCurvesByStage:
    """Compute the curves and readouts per stage and write the tables and the figure.

    Args:
        trials_by_cohort (Mapping[int, Mapping]): As :func:`staged_problem_curves`.
        staged (pd.DataFrame): ``training_stages.staged_problems``.
        out_dir (str | Path): Folder of the figure; the tables go into its ``tables/``.

    Returns:
        ReversalCurvesByStage: The tables.
    """
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    problem_curves = staged_problem_curves(trials_by_cohort, staged)
    curves = rcs.stage_curves(problem_curves, staged[["mouse", "phase", "phase_problem", "stage"]])
    readouts = rcs.adaptation_readouts(curves)
    tests = stage_tests(readouts, "probability", by=["role"])
    result = ReversalCurvesByStage(curves, readouts, tests)
    for name, table in result._asdict().items():
        table.to_csv(out_dir / "tables" / f"reversal_curves_by_stage_{name}.csv", index=False)
    plots.plot_reversal_curves_by_stage(
        curves, readouts, tests, out_dir / plots.REVERSAL_CURVES_BY_STAGE
    )
    return result
