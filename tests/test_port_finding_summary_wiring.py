# ruff: noqa: F811, F401  (fixtures imported from another test module are used as arguments)
"""Where the runners write the port-finding summary, with the drawing stubbed.

- ``run_problem``: the per-layout heatmaps of the problem, into its ``long-pokes`` folder;
- ``run_cohort``: the port-category fractions of the cohort's drifting problems, into
  ``drifting/all-drifting-problems/port-finding-summary``;
- ``run_all_cohorts_drifting``: both, pooled over the cohorts, into
  ``all-cohorts/port-finding-summary`` (the excluded-mice set without those mice).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

import src.pipelines.all_cohorts_drifting as all_cohorts
import src.pipelines.cohort_analysis as cohort_analysis
import src.pipelines.open_field as open_field
from src.behavior_import.import_log import ImportLog
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cohort_analysis import run_cohort
from src.pipelines.open_field import AnalysisSelection, run_problem
from tests.all_cohorts_fixtures import synthetic_cohort
from tests.cohort_fixtures import blocked_then_two_drifting
from tests.test_cohort_runner_integration import (
    calls,
    close_figures,
    cohort_extras,
    single_sessions,
)
from tests.test_port_finding_summary import port_finding_only, session

POKES_ONLY = AnalysisSelection(
    run_choice_probability_analysis=False,
    run_first_leave_analysis=False,
    run_number_of_reversals_analysis=False,
    run_rank_proportion_analysis=False,
    run_single_session_analysis=False,
    run_poke_analysis=True,
)


def test_the_problem_heatmaps_go_to_its_long_pokes_folder_with_its_scope(monkeypatch):
    monkeypatch.setattr(open_field, "run_poke_analysis",
                        lambda *a, **k: {"summary": None, "port_finding_check": None})  # fmt: skip
    seen = []
    monkeypatch.setattr(open_field, "write_problem_port_finding_heatmaps",
                        lambda summaries, out_dir, *, scope: seen.append((summaries, out_dir,
                                                                          scope)))  # fmt: skip
    found = {"m1": {"ses-0_date-20251231": port_finding_only(2)}}
    run_problem(
        3,
        {"m1": {"ses-1_date-20260101": session(1, blocks=[1, 1])}},
        task="open-field",
        cohort="cohort-09",
        output_root=Path("out"),
        selection=POKES_ONLY,
        task_type="blocked",
        previous_choice_towers={"m1": frozenset({"A1", "C3"})},
        port_finding_sessions=found,
    )
    ((summaries, out_dir, scope),) = seen
    assert out_dir == Path("out/open-field/cohort-09/blocked/problem-03/long-pokes")
    assert scope == "cohort-09 blocked problem-03"
    # the port-finding-only session is counted and the previous ports outlined
    assert summaries["m1"].n_port_finding_trials == 3
    assert summaries["m1"].previous_ports == frozenset({"A1", "C3"})


def test_no_poke_analysis_means_no_problem_heatmaps(monkeypatch):
    seen = []
    monkeypatch.setattr(open_field, "write_problem_port_finding_heatmaps",
                        lambda *a, **k: seen.append(a))  # fmt: skip
    run_problem(1, {"m1": {}}, task="t", cohort="c", output_root=Path("out"),
                selection=POKES_ONLY.model_copy(update={"run_poke_analysis": False}))  # fmt: skip
    assert seen == []


def test_the_cohort_fractions_cover_its_drifting_problems(
    monkeypatch, tmp_path, calls, cohort_extras, single_sessions
):
    seen = []
    monkeypatch.setattr(cohort_analysis, "write_port_finding_fractions",
                        lambda entries, pooled_dir, scope: seen.append((entries, pooled_dir,
                                                                        scope)))  # fmt: skip
    run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(),
               exclude_mice=["m2"])  # fmt: skip
    (entries, pooled_dir, scope), (without, without_dir, without_scope) = seen
    assert pooled_dir == tmp_path / "figures/open-field/cohort-01/drifting/all-drifting-problems"
    assert scope == "cohort-01 drifting problems"
    assert {labels["task_type"] for labels, _ in entries} == {"drifting"}
    assert {s.mouse for _, s in entries} == {"m1", "m2"}
    assert without_dir.name == "all-drifting-problems-without-m2"
    assert {s.mouse for _, s in without} == {"m1"}


def test_the_cohort_fractions_folder_is_emptied_before_it_is_written(tmp_path):
    out = tmp_path / cohort_analysis.PORT_FINDING_SUMMARY_FOLDER
    out.mkdir()
    (out / "stale.png").write_text("old")
    cohort_analysis.write_port_finding_fractions([], tmp_path, "s")
    assert [p.name for p in out.iterdir()] == ["Port Finding Summary - Left Out.txt"]


@pytest.fixture(scope="module")
def imported():
    return {1: synthetic_cohort(["a1", "a2"], seed=1), 3: synthetic_cohort(["b1"], seed=3)}


def test_all_cohorts_pool_the_drifting_problems_of_every_cohort(monkeypatch, imported, tmp_path):
    seen = []

    def record(entries, out_dir, label):
        seen.append((entries, out_dir))

    monkeypatch.setattr(all_cohorts, "_draw_port_finding_summary", record)
    run_all_cohorts_drifting(
        [1, 3],
        results_root=tmp_path,
        trials_by_cohort=imported,
        import_logs={1: ImportLog(), 3: ImportLog()},
        exclude_mice=["a2"],
    )
    (plain, plain_dir), (without, without_dir) = seen
    assert plain_dir == tmp_path / "figures/open-field/all-cohorts/port-finding-summary"
    assert without_dir.parent.name == "all-cohorts-without-a2"
    assert {key[0] for key in plain} == {"cohort-01", "cohort-03"}
    assert {key[2] for key in plain} == {"a1", "a2", "b1"}
    assert {key[2] for key in without} == {"a1", "b1"}
    # drifting problems only, numbered within the type
    assert {key[1] for key in plain} == {1, 2}
