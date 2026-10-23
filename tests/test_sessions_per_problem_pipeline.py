# ruff: noqa: F811, F401  (fixtures imported from another test module are used as arguments)
"""The cohort runner writes the sessions-per-problem table and figure.

The fixture cohort (``tests/cohort_fixtures.py``) has, for both mice, blocked problem 1 (ses-1,
ses-2) and drifting problems 1 (ses-3, ses-4) and 2 (ses-5, ses-6); the drifting-only entry points
are stubbed as in ``tests/test_cohort_runner_integration.py``, whose fixtures are reused.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src.behavior_import.import_log import ImportLog
from src.behavior_import.task_types import BLOCKED, DRIFTING
from src.pipelines.cohort_analysis import run_cohort, run_cohort_incremental
from src.pipelines.sessions_per_problem import (
    SESSIONS_PER_PROBLEM_FOLDER,
    write_sessions_per_problem_figures,
)
from tests.cohort_fixtures import PROBLEMS, blocked_then_two_drifting, cohort_with_runs
from tests.test_cohort_port_finding_sessions import port_finding_only
from tests.test_cohort_runner_integration import (
    calls,
    close_figures,
    cohort_dir,
    cohort_extras,
    single_sessions,
)
from tests.test_sessions_per_problem import ROWS, table_of

FIGURE_FILES = {
    f"Sessions per Drifting Problem.{ext}" for ext in ("caption.txt", "pdf", "png", "svg")
}


@pytest.fixture
def stubbed(calls, cohort_extras, single_sessions):
    return None


@pytest.fixture
def figure_dir(cohort_dir):
    return cohort_dir / DRIFTING / "all-drifting-problems" / SESSIONS_PER_PROBLEM_FOLDER


def table_file(tmp_path):
    return tmp_path / "tables" / "open-field" / "cohort-01" / "sessions_per_problem.csv"


def run(tmp_path, **kwargs):
    return run_cohort(
        1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), **kwargs
    )


# --- the table -------------------------------------------------------------------------------


def test_the_table_lists_each_mouse_and_drifting_problem_with_its_sessions(tmp_path, stubbed):
    result = run(tmp_path)
    table = pd.read_csv(table_file(tmp_path))
    assert table[["mouse", "problem_in_type", "overall_problem", "n_sessions"]].values.tolist() == [
        ["m1", 1, 2, 2],
        ["m1", 2, 3, 2],
        ["m2", 1, 2, 2],
        ["m2", 2, 3, 2],
    ]
    # the last problem of each mouse is the latest one, so it may still be running
    assert table["complete"].tolist() == [True, False, True, False]
    assert list(result.review_tables["sessions_per_problem"]["mouse"]) == list(table["mouse"])


def test_a_blocked_last_problem_leaves_every_drifting_problem_complete(tmp_path, stubbed):
    problems = {**PROBLEMS, 4: PROBLEMS[1]}  # blocked again after the two drifting problems
    cohort = cohort_with_runs(problems, {"m1": [1, 2, 3, 4], "m2": [1, 2, 3, 4]})
    run_cohort(1, results_root=tmp_path, trials_by_problem=cohort)
    assert pd.read_csv(table_file(tmp_path))["complete"].all()


def test_port_finding_only_sessions_of_the_log_are_counted_in_their_own_column(tmp_path, stubbed):
    # m1 runs the blocked problem (ses-1, ses-2), then two port-finding-only sessions on the
    # towers of overall problem 3, then that drifting problem (ses-5, ses-6)
    cohort = cohort_with_runs(PROBLEMS, {"m1": [1, 3]})
    towers = PROBLEMS[3][1]
    log = ImportLog(
        port_finding_sessions={
            "m1": {
                "ses-3_date-20260103": port_finding_only(towers),
                "ses-4_date-20260104": port_finding_only(towers),
            }
        }
    )
    with pytest.warns(UserWarning):  # the port-finding check flags the 5-trial sessions
        run_cohort(1, results_root=tmp_path, trials_by_problem=cohort, import_log=log)
    (row,) = pd.read_csv(table_file(tmp_path)).to_dict("records")
    assert (row["n_sessions"], row["n_port_finding_only_sessions"], row["n_sessions_total"]) == (
        2,
        2,
        4,
    )


def test_a_cohort_without_drifting_problems_writes_the_table_header_and_no_figure(
    tmp_path, stubbed
):
    cohort = cohort_with_runs(PROBLEMS, {"m1": [1], "m2": [1]})
    run_cohort(1, results_root=tmp_path, trials_by_problem=cohort)
    assert pd.read_csv(table_file(tmp_path)).empty
    assert not list((tmp_path / "figures").rglob(SESSIONS_PER_PROBLEM_FOLDER))


# --- the figure ------------------------------------------------------------------------------


def test_the_figure_goes_into_a_sessions_per_problem_folder_of_all_drifting_problems(
    tmp_path, figure_dir, stubbed
):
    run(tmp_path)
    assert {p.name for p in figure_dir.iterdir()} == FIGURE_FILES


def test_a_partial_run_regenerates_it_when_a_drifting_problem_is_touched(
    tmp_path, figure_dir, stubbed
):
    run(tmp_path, only_problems={(DRIFTING, 2)})
    assert {p.name for p in figure_dir.iterdir()} == FIGURE_FILES


def test_an_incremental_run_of_the_last_sessions_regenerates_it_and_the_table(
    tmp_path, figure_dir, stubbed
):
    run_cohort_incremental(
        1,
        last_sessions=1,
        results_root=tmp_path,
        trials_by_problem=blocked_then_two_drifting(),
    )
    assert {p.name for p in figure_dir.iterdir()} == FIGURE_FILES
    assert len(pd.read_csv(table_file(tmp_path))) == 4


def test_a_blocked_only_partial_run_redraws_the_figure_with_the_table(
    tmp_path, figure_dir, stubbed
):
    # a new blocked session can complete a drifting problem, so the figure follows the table
    run(tmp_path, only_problems={(BLOCKED, 1)})
    assert {p.name for p in figure_dir.iterdir()} == FIGURE_FILES
    assert len(pd.read_csv(table_file(tmp_path))) == 4


# --- the writer ---------------------------------------------------------------------------------


def test_the_total_can_be_plotted_instead_of_the_bandit_count(tmp_path):
    write_sessions_per_problem_figures(
        table_of(ROWS), tmp_path, "cohort 3", include_port_finding=True
    )
    assert {p.name for p in tmp_path.iterdir()} == FIGURE_FILES
