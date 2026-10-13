from pathlib import Path

import pytest

import src.pipelines.open_field as pipeline
from src.pipelines.open_field import AnalysisSelection, run_open_field_analysis, run_problem

SECTIONS = [
    "run_choice_probability_analysis",
    "run_first_leave_analysis",
    "run_number_of_reversals_analysis",
    "run_rank_proportion_analysis",
    "run_single_session_analysis",
]


@pytest.fixture
def calls(monkeypatch):
    seen = []
    for name in SECTIONS:
        monkeypatch.setattr(pipeline, name, lambda *a, _n=name, **k: seen.append((_n, a[1])))
    return seen


def test_run_problem_runs_selected_sections_in_problem_dir(calls):
    run_problem(2, {}, task="open-field", cohort="cohort-01", output_root=Path("out"))
    assert [name for name, _ in calls] == SECTIONS
    assert {str(d.parent) for _, d in calls} == {"out/open-field/cohort-01/problem-02"}


def test_reversal_window_sections_stop_after_last_problem(calls):
    run_problem(5, {}, task="t", cohort="c", output_root=Path("out"))
    assert [name for name, _ in calls] == SECTIONS[2:]


def test_disabled_sections_are_skipped(calls):
    selection = AnalysisSelection(
        run_choice_probability_analysis=False, run_single_session_analysis=False
    )
    run_problem(1, {}, task="t", cohort="c", output_root=Path("out"), selection=selection)
    assert [name for name, _ in calls] == SECTIONS[1:4]


def test_run_open_field_analysis_goes_through_problems_in_order(calls):
    run_open_field_analysis({3: {}, 1: {}}, task="t", cohort="c", output_root=Path("out"))
    problems = [d.parent.name for name, d in calls if name == SECTIONS[0]]
    assert problems == ["problem-01", "problem-03"]


def test_settings_reject_unknown_options():
    with pytest.raises(ValueError):
        AnalysisSelection(run_choice_probabilty_analysis=False)
