"""run_cohort writes the review tables next to the problem map.

The analyses themselves are replaced by stubs here (the other runner tests cover them): what
is under test is which tables are written, from which inputs, and that the warnings log is
complete while the console still shows every warning.
"""

import warnings

import pandas as pd
import pytest

import src.pipelines.cohort_analysis as cohort_analysis
from src.behavior_import import import_log
from src.behavior_import.import_log import ImportLog, SessionFlag, SkipRecord
from src.behavior_import.load_trials import ImportedTrials
from src.behavior_import.warning_records import WarningRecord
from src.pipelines.cohort_analysis import run_cohort
from tests.cohort_fixtures import blocked_then_two_drifting

TABLE_FILES = {
    "problem_map",
    "suspect_reversals",
    "suspect_sessions",
    "suspect_starting_reward_mags",
    "no_trial_sessions",
    "no_second_best",
    "poke_trial_summary",
    "reversal_aligned_arms_counts",
    "value_change_events",
    "value_change_bin_counts",
    "warnings_log",
}


@pytest.fixture
def stubbed(monkeypatch):
    """Stub the analyses and figures: ``run_problem`` fills the diagnostics and may warn."""
    state = {"warn": []}

    def run_problem(k, subjects_trials, *, diagnostics, task_type, **kwargs):
        for text in state["warn"]:
            warnings.warn(text.format(k=k, task_type=task_type), UserWarning, stacklevel=1)
        mice = sorted(subjects_trials)
        diagnostics["poke_trial_summary"] = pd.DataFrame(
            {"mouse": mice, "session": ["s"] * len(mice), "n_trials": range(len(mice))}
        )
        if task_type == "drifting":
            diagnostics["reversal_aligned_arms_counts"] = pd.DataFrame({"scope": ["pooled"]})
            diagnostics["value_change_events"] = pd.DataFrame({"config": ["best"]})
            diagnostics["value_change_bin_counts"] = pd.DataFrame({"t": [0]})
        return {}

    monkeypatch.setattr(cohort_analysis, "run_problem", run_problem)
    monkeypatch.setattr(cohort_analysis, "run_drifting_value_analyses", lambda *a, **k: {})
    monkeypatch.setattr(cohort_analysis, "pooled_drifting_rank_results", lambda *a, **k: {})
    monkeypatch.setattr(cohort_analysis, "plot_pooled_drifting_rank", lambda *a, **k: None)
    monkeypatch.setattr(cohort_analysis, "plot_trial_block_summaries", lambda *a, **k: [])
    monkeypatch.setattr(cohort_analysis, "subject_numbers_from_sessions", lambda *a, **k: {})
    monkeypatch.setattr(cohort_analysis, "build_trial_block_tables", lambda *a, **k: (None, None))
    return state


def tables_dir(tmp_path):
    return tmp_path / "tables" / "open-field" / "cohort-01"


def read(tmp_path, name):
    return pd.read_csv(tables_dir(tmp_path) / f"{name}.csv")


def test_every_table_is_written_and_returned(tmp_path, stubbed):
    result = run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting())
    assert {p.stem for p in tables_dir(tmp_path).glob("*.csv")} == TABLE_FILES
    assert set(result.review_tables) == TABLE_FILES - {"suspect_reversals"}


def test_the_problem_map_gains_towers_and_session_range(tmp_path, stubbed):
    run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting())
    table = read(tmp_path, "problem_map")
    first = table[(table.mouse == "m1") & (table.overall_problem == 1)].iloc[0]
    assert first["choice_towers"] == "A1;B1;C1"
    assert (first["initiation_tower"], first["n_sessions"], first["session_range"]) == (
        "B2",
        2,
        "1-2",
    )
    second_drifting = table[(table.mouse == "m1") & (table.overall_problem == 3)].iloc[0]
    assert second_drifting["session_range"] == "5-6"


def test_diagnostics_are_stacked_over_problems_with_task_type_and_problem(tmp_path, stubbed):
    run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting())
    poke = read(tmp_path, "poke_trial_summary")
    assert list(poke.columns)[:4] == ["task_type", "problem_in_type", "overall_problem", "mouse"]
    got = {(r.task_type, r.problem_in_type, r.overall_problem, r.mouse) for r in poke.itertuples()}
    assert got == {
        (kind, k, overall, mouse)
        for kind, k, overall in (("blocked", 1, 1), ("drifting", 1, 2), ("drifting", 2, 3))
        for mouse in ("m1", "m2")
    }
    arms = read(tmp_path, "reversal_aligned_arms_counts")
    assert arms[["task_type", "problem_in_type"]].values.tolist() == [
        ["drifting", 1],
        ["drifting", 2],
    ]


def test_tables_with_nothing_to_list_are_written_with_their_columns(tmp_path, stubbed):
    run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting())
    for name in ("suspect_sessions", "no_trial_sessions", "no_second_best"):
        table = read(tmp_path, name)
        assert table.empty and len(table.columns) > 5


def test_the_import_log_feeds_the_skip_flag_and_warning_tables(tmp_path, stubbed, monkeypatch):
    log = ImportLog(
        skips=[
            SkipRecord(
                "m1",
                "ses-9_date-20260109",
                "/raw/m1/behav/m1-x.tsv",
                import_log.SESSION,
                import_log.NO_TRIAL_PRINTS,
                "it prints no trials",
            )
        ],
        flags=[
            SessionFlag(
                "m2",
                "ses-9_date-20260109",
                ("/raw/m2.tsv",),
                "port_finding",
                None,
                False,
                "pf",
                None,
            )
        ],
        warnings=[
            WarningRecord(
                "ProblemNumberingWarning", "m2 ses-9_date-20260109: port finding", "a.py", 3
            )
        ],
    )
    monkeypatch.setattr(
        cohort_analysis,
        "load_trials_with_log",
        lambda *a, **k: ImportedTrials(blocked_then_two_drifting(), log),
    )
    run_cohort(1, data_root=tmp_path, results_root=tmp_path)
    skipped = read(tmp_path, "no_trial_sessions")
    assert skipped[["mouse", "session", "kind"]].values.tolist() == [
        ["m1", "ses-9_date-20260109", "no_trial_prints"]
    ]
    suspect = read(tmp_path, "suspect_sessions")
    assert suspect[["mouse", "session", "flag"]].values.tolist() == [
        ["m2", "ses-9_date-20260109", "port_finding"]
    ]
    logged = read(tmp_path, "warnings_log")
    assert logged["message"].tolist() == ["m2 ses-9_date-20260109: port finding"]
    assert logged["mouse"].tolist() == ["m2"]


def test_warnings_of_the_run_are_logged_after_those_of_the_import_and_still_shown(
    tmp_path, stubbed, monkeypatch
):
    stubbed["warn"] = ["problem {k} of {task_type} looks odd"]
    imported = ImportLog(warnings=[WarningRecord("UserWarning", "from the import", "a.py", 1)])
    monkeypatch.setattr(
        cohort_analysis,
        "load_trials_with_log",
        lambda *a, **k: ImportedTrials(blocked_then_two_drifting(), imported),
    )
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        run_cohort(1, data_root=tmp_path, results_root=tmp_path)
    logged = read(tmp_path, "warnings_log")
    assert logged["message"].tolist() == [
        "from the import",
        "problem 1 of blocked looks odd",
        "problem 1 of drifting looks odd",
        "problem 2 of drifting looks odd",
    ]
    assert [str(w.message) for w in shown] == logged["message"].tolist()[1:]


def test_a_failing_run_still_leaves_the_tables_that_do_not_need_the_analyses(
    tmp_path, stubbed, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(cohort_analysis, "run_problem", fail)
    with pytest.raises(RuntimeError, match="analysis failed"):
        run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting())
    written = {p.stem for p in tables_dir(tmp_path).glob("*.csv")}
    assert {"problem_map", "suspect_sessions", "no_second_best"} <= written
    assert "warnings_log" not in written
