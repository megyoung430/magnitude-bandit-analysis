"""What ``run_problem`` passes to the analyses and collects from them, with the analyses stubbed."""

from pathlib import Path

import src.pipelines.open_field as pipeline
from src.behavior_analysis.port_finding_sessions import assign_port_finding_sessions
from src.pipelines.open_field import AnalysisSelection, run_drifting_value_analyses, run_problem

NOTHING_BUT = dict.fromkeys(
    (
        "run_choice_probability_analysis",
        "run_first_leave_analysis",
        "run_number_of_reversals_analysis",
        "run_rank_proportion_analysis",
        "run_single_session_analysis",
    ),
    False,
)


def only(**on):
    return AnalysisSelection(**{**NOTHING_BUT, **on})


def run(selection, task_type, **kwargs):
    return run_problem(
        3,
        {"m1": {}},
        task="t",
        cohort="c",
        output_root=Path("out"),
        selection=selection,
        task_type=task_type,
        **kwargs,
    )


def test_the_reversal_figures_are_told_the_task_type_and_problem_for_their_warnings(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        pipeline, "run_choice_probability_analysis", lambda *a, **k: seen.update(choice=k)
    )
    monkeypatch.setattr(pipeline, "run_first_leave_analysis", lambda *a, **k: seen.update(leave=k))
    run(only(run_choice_probability_analysis=True, run_first_leave_analysis=True), "blocked")
    assert seen["choice"] == {"problem_number": 3, "task_type": "blocked"}
    assert seen["leave"] == {"problem_number": 3, "task_type": "blocked"}


def test_the_poke_tables_are_collected_as_diagnostics(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "run_poke_analysis",
        lambda *a, **k: {"summary": "POKE", "port_finding_check": "CHECK"},
    )
    diagnostics = {}
    run(only(run_poke_analysis=True), "blocked", diagnostics=diagnostics)
    assert diagnostics == {"poke_trial_summary": "POKE", "port_finding_check": "CHECK"}


def test_the_port_finding_sessions_of_the_problem_reach_the_poke_analysis(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        pipeline,
        "run_poke_analysis",
        lambda *a, **k: seen.update(k) or {"summary": None, "port_finding_check": None},
    )
    found = {"m1": {"ses-1_date-20260101": {"df": "frames"}}}
    run(only(run_poke_analysis=True), "blocked", port_finding_sessions=found)
    assert seen["port_finding_sessions"] == found


def test_the_open_field_runner_gives_each_problem_its_own_port_finding_sessions(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "run_problem", lambda number, trials, **k: calls.append((number, k)) or {}
    )
    sessions = {
        1: {"m1": {"ses-2_date-20260102": {"choice_towers": [["A1", "B1", "C1"]]}}},
        2: {"m1": {"ses-4_date-20260104": {"choice_towers": [["A3", "B3", "C3"]]}}},
    }
    found = {
        "m1": {
            "ses-3_date-20260103": {"df": "frames", "choice_towers": frozenset(["A3", "B3", "C3"])}
        }
    }
    assignment = assign_port_finding_sessions(sessions, found)
    pipeline.run_open_field_analysis(
        sessions, port_finding_assignment=assignment, task="t", cohort="c", output_root=Path("out")
    )
    assert [(n, k["port_finding_sessions"]) for n, k in calls] == [(1, {}), (2, found)]


def test_the_value_tables_of_a_drifting_problem_are_collected_as_diagnostics(monkeypatch):
    tables = {
        "value_change_events": "E",
        "value_change_bin_counts": "B",
        "value_change_symmetric_exclusions": "X",
    }
    monkeypatch.setattr(pipeline, "run_drifting_value_analyses", lambda *a, **k: tables)
    diagnostics = {}
    run(only(run_value_change_analysis=True), "drifting", diagnostics=diagnostics)
    assert diagnostics == tables


def test_each_value_analysis_returns_its_diagnostic_tables(monkeypatch):
    monkeypatch.setattr(pipeline, "pooled_arm_windows", lambda problems: "WINDOWS")
    monkeypatch.setattr(
        pipeline, "run_reversal_aligned_arm_analysis", lambda *a, **k: {"counts": "ARMS"}
    )
    monkeypatch.setattr(
        pipeline,
        "run_value_change_analysis",
        lambda *a, **k: {
            "contamination_table": "EVENTS",
            "bin_counts_table": "BINS",
            "symmetric_exclusions_table": "EXCLUSIONS",
        },
    )
    selection = only(run_reversal_aligned_arm_analysis=True, run_value_change_analysis=True)
    tables = run_drifting_value_analyses({1: {}}, Path("out"), selection, arm_scope_label="x")
    assert tables == {
        "reversal_aligned_arms_counts": "ARMS",
        "value_change_events": "EVENTS",
        "value_change_bin_counts": "BINS",
        "value_change_symmetric_exclusions": "EXCLUSIONS",
    }


def test_the_stay_switch_psychometrics_run_with_the_value_psychometrics(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_value_psychometric_analysis", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "run_stay_switch_analysis", lambda data, out_dir, **k: calls.append(out_dir)
    )
    selection = only(run_value_psychometric_analysis=True)
    run_drifting_value_analyses({1: {}}, Path("out"), selection, arm_scope_label="x")
    assert calls == [Path("out") / "value-psychometrics-stay-switch"]
    calls.clear()
    run_drifting_value_analyses({1: {}}, Path("out"), only(), arm_scope_label="x")
    assert calls == []
