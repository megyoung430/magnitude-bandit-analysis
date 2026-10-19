# ruff: noqa: F811, F401  (fixtures imported from another test module are used as arguments)
"""run_cohort and run_cohort_incremental hand the import log's port-finding-only sessions on.

The fixture cohort (``tests/cohort_fixtures.py``) has, for both mice, blocked problem 1 (choice
ports A1, B1, C1; ses-1, ses-2). A port-finding-only session of m1 dated before it belongs to that
problem. The drifting-only entry points and cohort-level plots are stubbed as in
``tests/test_cohort_runner_integration.py``, whose fixtures are reused.
"""

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src.behavior_import.import_log import ImportLog
from src.behavior_import.task_types import BLOCKED, DRIFTING
from src.pipelines.cohort_analysis import run_cohort, run_cohort_incremental
from tests.cohort_fixtures import PROBLEMS, blocked_then_two_drifting
from tests.poke_fixtures import Recording
from tests.test_cohort_runner_integration import (
    calls,
    close_figures,
    cohort_dir,
    cohort_extras,
    single_sessions,
)

EARLIER = "ses-0_date-20251231"
PORT_FINDING_FILES = (
    "Maze Heatmaps - Pokes After Choice (Port Finding).png",
    "Maze Heatmaps - Pokes After Initiation (Port Finding).png",
)


def port_finding_only(towers, n_trials=5):
    rec = Recording(towers=towers)
    for _ in range(n_trials):
        rec.port_finding_trial(towers[0], long_before=["C3"])
    rec.rows.insert(1, (0.001, "variable", "run_start", json.dumps({"found_t_thresh": 4})))
    return {"df": rec.to_frame(), "choice_towers": frozenset(towers), "initiation_tower": "B2"}


def log_with(mouse, key, record):
    return ImportLog(port_finding_sessions={mouse: {key: record}})


def tables(tmp_path):
    folder = tmp_path / "tables" / "open-field" / "cohort-01"
    return {
        name: pd.read_csv(folder / f"{name}.csv")
        for name in ("poke_trial_summary", "port_finding_check")
    }


def long_pokes_dir(cohort_dir, kind, k):
    return cohort_dir / kind / f"problem-{k:02d}" / "long-pokes"


@pytest.fixture
def stubbed(calls, cohort_extras, single_sessions):
    return None


def test_the_sessions_of_the_log_reach_the_port_finding_maps_of_their_problem(
    tmp_path, cohort_dir, stubbed
):
    log = log_with("m1", EARLIER, port_finding_only(PROBLEMS[1][1]))
    run_cohort(
        1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), import_log=log
    )
    blocked = long_pokes_dir(cohort_dir, BLOCKED, 1)
    assert all((blocked / name).exists() for name in PORT_FINDING_FILES)
    # the fixture's task sessions have no port finding: only m1's other session draws a map
    for k in (1, 2):
        assert not (long_pokes_dir(cohort_dir, DRIFTING, k) / PORT_FINDING_FILES[0]).exists()
    summary = tables(tmp_path)["poke_trial_summary"]
    marked = summary[summary["port_finding_only"]]
    assert marked[["task_type", "problem_in_type", "mouse", "session"]].values.tolist() == [
        [BLOCKED, 1, "m1", EARLIER]
    ]
    assert marked["n_port_finding_trials"].tolist() == [5]


def test_the_check_table_is_written_with_the_problem_and_mouse(tmp_path, stubbed):
    log = log_with("m1", EARLIER, port_finding_only(PROBLEMS[1][1], n_trials=5))
    with pytest.warns(UserWarning, match="below the expected minimum"):
        run_cohort(
            1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), import_log=log
        )
    check = tables(tmp_path)["port_finding_check"]
    (row,) = check.to_dict("records")
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"], row["mouse"]) == (
        BLOCKED,
        1,
        1,
        "m1",
    )
    assert row["n_port_finding_trials"] == 5 and row["below_expected_min"]
    folder = tmp_path / "tables" / "open-field" / "cohort-01"
    warnings_log = pd.read_csv(folder / "warnings_log.csv")
    assert warnings_log["message"].str.contains("below the expected minimum").any()


def test_a_session_with_other_towers_is_not_used_and_warns(tmp_path, cohort_dir, stubbed):
    log = log_with("m1", EARLIER, port_finding_only(["A2", "B3", "C3"]))
    with pytest.warns(UserWarning, match=r"m1 ses-0_date-20251231.*A2.*B3.*C3"):
        run_cohort(
            1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), import_log=log
        )
    assert not (long_pokes_dir(cohort_dir, BLOCKED, 1) / PORT_FINDING_FILES[0]).exists()
    assert not tables(tmp_path)["poke_trial_summary"]["port_finding_only"].any()


def test_a_run_without_port_finding_only_sessions_is_unchanged(tmp_path, stubbed):
    def summary_of(root, **kwargs):
        run_cohort(
            1, results_root=root, trials_by_problem=blocked_then_two_drifting(), **kwargs
        )
        return tables(root)["poke_trial_summary"]

    with pytest.warns(UserWarning, match="without an import_log"):
        without_log = summary_of(tmp_path / "a")
    with_log = summary_of(tmp_path / "b", import_log=ImportLog())
    assert with_log.equals(without_log)
    assert not with_log["port_finding_only"].any()
    assert len(with_log) == 12  # two mice x six sessions


def test_the_incremental_run_uses_the_log_for_the_problems_it_regenerates(tmp_path, stubbed):
    log = log_with("m1", EARLIER, port_finding_only(PROBLEMS[1][1]))
    run_cohort_incremental(
        1,
        last_sessions=1,
        results_root=tmp_path,
        trials_by_problem=blocked_then_two_drifting(),
        import_log=log,
    )
    summary = tables(tmp_path)["poke_trial_summary"]
    # the last session of each mouse is in drifting problem 2: the blocked problem is not rerun
    assert not summary["port_finding_only"].any()
    run_cohort_incremental(
        1,
        since="2026-01-01",
        results_root=tmp_path,
        trials_by_problem=blocked_then_two_drifting(),
        import_log=log,
    )
    assert tables(tmp_path)["poke_trial_summary"]["port_finding_only"].sum() == 1
