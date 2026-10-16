"""What ``run_problem`` passes to the analyses and collects from them, with the analyses stubbed."""

from pathlib import Path

import src.pipelines.open_field as pipeline
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


def test_the_poke_summary_is_collected_as_a_diagnostic(monkeypatch):
    monkeypatch.setattr(pipeline, "run_poke_analysis", lambda *a, **k: {"summary": "POKE"})
    diagnostics = {}
    run(only(run_poke_analysis=True), "blocked", diagnostics=diagnostics)
    assert diagnostics == {"poke_trial_summary": "POKE"}


def test_the_value_tables_of_a_drifting_problem_are_collected_as_diagnostics(monkeypatch):
    tables = {"value_change_events": "E", "value_change_bin_counts": "B"}
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
        lambda *a, **k: {"contamination_table": "EVENTS", "bin_counts_table": "BINS"},
    )
    selection = only(run_reversal_aligned_arm_analysis=True, run_value_change_analysis=True)
    tables = run_drifting_value_analyses({1: {}}, Path("out"), selection, arm_scope_label="x")
    assert tables == {
        "reversal_aligned_arms_counts": "ARMS",
        "value_change_events": "EVENTS",
        "value_change_bin_counts": "BINS",
    }
