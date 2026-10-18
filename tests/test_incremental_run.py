# ruff: noqa: F811, F401  (fixtures imported from another test module are used as arguments)
"""Partial cohort runs: only the problems of the selected sessions are regenerated.

The fixture cohort (``tests/cohort_fixtures.py``) has, for both mice, blocked problem 1 (ses-1,
ses-2), drifting problem 1 (ses-3, ses-4) and drifting problem 2 (ses-5, ses-6), dated
2026-01-01 to 2026-01-06. The drifting-only entry points and the cohort-level plots are
stubbed as in ``tests/test_cohort_runner_integration.py``, whose fixtures are reused.
"""

import shutil

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import src.pipelines.cohort_analysis as cohort_analysis
from src.behavior_import.task_types import BLOCKED, DRIFTING
from src.pipelines.cohort_analysis import run_cohort, run_cohort_incremental, run_cohorts
from src.pipelines.incremental import NoSessionsSelectedError
from tests.cohort_fixtures import blocked_then_two_drifting
from tests.test_cohort_runner_integration import (
    STUB_RESULTS,
    calls,
    close_figures,
    cohort_dir,
    cohort_extras,
    single_sessions,
)

BLOCKED_1, DRIFTING_1, DRIFTING_2 = (BLOCKED, 1), (DRIFTING, 1), (DRIFTING, 2)
DIAGNOSTICS = (
    "poke_trial_summary",
    "reversal_aligned_arms_counts",
    "value_change_events",
    "value_change_bin_counts",
    "value_change_symmetric_exclusions",
)


def problem_dirs(cohort_dir):
    """Problem folders that exist, as ``"drifting/problem-02"`` and so on."""
    return sorted(f"{p.parent.name}/{p.name}" for p in cohort_dir.glob("*/problem-*") if p.is_dir())


def rerun(tmp_path, problems):
    return run_cohort(
        1,
        results_root=tmp_path,
        trials_by_problem=blocked_then_two_drifting(),
        only_problems=problems,
    )


@pytest.fixture
def tables_dir(tmp_path):
    return tmp_path / "tables" / "open-field" / "cohort-01"


@pytest.fixture
def earlier_full_run(tmp_path, calls, cohort_extras, single_sessions, tables_dir):
    """A full run, then its figures deleted so that only a later run's figures are on disk.

    Its tables stay: they are what a partial run merges into.
    """
    run_cohort(1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting())
    shutil.rmtree(tmp_path / "figures")
    calls.clear()
    return tables_dir


def mark_old_rows(tables_dir):
    """Add an ``old`` column to every diagnostic CSV so rows of the earlier run can be told."""
    for name in DIAGNOSTICS:
        frame = pd.read_csv(tables_dir / f"{name}.csv")
        frame["old"] = "earlier"
        frame.to_csv(tables_dir / f"{name}.csv", index=False)


# --- which figures are written -------------------------------------------------------------


def test_a_partial_run_writes_only_the_figures_of_the_given_problems(
    tmp_path, cohort_dir, earlier_full_run
):
    rerun(tmp_path, {DRIFTING_2})
    assert problem_dirs(cohort_dir) == ["drifting/problem-02"]
    assert (cohort_dir / DRIFTING / "all-drifting-problems" / "value-change" / "stub.txt").exists()
    assert not (cohort_dir / BLOCKED).exists()


def test_the_pooled_drifting_analyses_still_pool_every_drifting_problem(
    tmp_path, earlier_full_run, calls
):
    rerun(tmp_path, {DRIFTING_1})
    # the reversal-aligned entry gets one pre-pooled entry, not the per-problem sessions
    pooled = [
        data
        for name, out_dir, data in calls
        if out_dir.parent.name == "all-drifting-problems"
        and name != "run_reversal_aligned_arm_analysis"
    ]
    assert pooled and all(sorted(data) == [1, 2] for data in pooled)


def test_a_blocked_only_partial_run_skips_the_pooled_drifting_analyses(
    tmp_path, cohort_dir, earlier_full_run, calls
):
    result = rerun(tmp_path, {BLOCKED_1})
    assert problem_dirs(cohort_dir) == ["blocked/problem-01"]
    assert not (cohort_dir / DRIFTING).exists()
    assert calls == []
    assert result.pooled_drifting_ranks == {}
    assert set(result.rank_results) == {BLOCKED_1}


# --- tables ---------------------------------------------------------------------------------


def test_a_partial_run_replaces_the_rows_of_its_problems_and_keeps_the_rest(
    tmp_path, earlier_full_run
):
    before = pd.read_csv(earlier_full_run / "poke_trial_summary.csv")
    mark_old_rows(earlier_full_run)
    rerun(tmp_path, {DRIFTING_2})
    after = pd.read_csv(earlier_full_run / "poke_trial_summary.csv")
    assert len(after) == len(before)
    rerun_rows = (after.task_type == DRIFTING) & (after.problem_in_type == 2)
    assert rerun_rows.any()
    assert after.loc[rerun_rows, "old"].isna().all()
    assert (after.loc[~rerun_rows, "old"] == "earlier").all()
    order = list(
        zip(after.task_type.map({BLOCKED: 0, DRIFTING: 1}), after.problem_in_type, strict=True)
    )
    assert order == sorted(order)


def test_a_partial_run_without_earlier_tables_warns_and_writes_only_its_rows(
    tmp_path, tables_dir, calls, cohort_extras, single_sessions
):
    with pytest.warns(UserWarning, match=r"poke_trial_summary.*partial"):
        rerun(tmp_path, {DRIFTING_2})
    summary = pd.read_csv(tables_dir / "poke_trial_summary.csv")
    assert set(zip(summary.task_type, summary.problem_in_type, strict=True)) == {DRIFTING_2}
    log = pd.read_csv(tables_dir / "warnings_log.csv")
    assert log.message.str.contains("partial").any()


def test_a_partial_run_rewrites_the_cohort_wide_tables_and_summaries(
    tmp_path, earlier_full_run, monkeypatch
):
    summaries = []
    monkeypatch.setattr(
        cohort_analysis, "plot_trial_block_summaries", lambda *a, **k: summaries.append(a)
    )
    (earlier_full_run / "problem_map.csv").unlink()
    rerun(tmp_path, {DRIFTING_2})
    assert len(summaries) == 1
    assert len(pd.read_csv(earlier_full_run / "problem_map.csv")) == 6  # 2 mice x 3 problems


def test_the_warnings_log_of_a_partial_run_covers_that_run_only(tmp_path, earlier_full_run):
    pd.DataFrame({"message": ["from the earlier run"]}).to_csv(
        earlier_full_run / "warnings_log.csv", index=False
    )
    rerun(tmp_path, {DRIFTING_2})
    log = pd.read_csv(earlier_full_run / "warnings_log.csv")
    assert not log.message.eq("from the earlier run").any()


def test_unknown_or_empty_problem_selections_raise_before_anything_is_written(
    tmp_path, calls, cohort_extras
):
    for bad, match in (({(DRIFTING, 9)}, r"drifting.*9"), (set(), "no problems")):
        with pytest.raises(ValueError, match=match):
            rerun(tmp_path, bad)
    assert not list(tmp_path.iterdir())


def test_a_partial_run_reports_what_it_wrote(tmp_path, earlier_full_run, capsys):
    rerun(tmp_path, {DRIFTING_2})
    out = capsys.readouterr().out
    assert "drifting problem 2" in out
    assert "all-drifting-problems" in out and "poke_trial_summary" in out


# --- last N sessions / since a date ------------------------------------------------------------


def test_last_sessions_reruns_the_problems_of_each_mouses_last_sessions(
    tmp_path, cohort_dir, earlier_full_run
):
    run_cohort_incremental(
        1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), last_sessions=1
    )
    assert problem_dirs(cohort_dir) == ["drifting/problem-02"]


def test_since_a_date_reruns_the_problems_with_sessions_on_or_after_it(
    tmp_path, cohort_dir, earlier_full_run
):
    run_cohort_incremental(
        1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), since="2026-01-04"
    )
    assert problem_dirs(cohort_dir) == ["drifting/problem-01", "drifting/problem-02"]


def test_the_plan_is_printed_before_the_run(tmp_path, earlier_full_run, capsys):
    run_cohort_incremental(
        1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), last_sessions=1
    )
    out = capsys.readouterr().out
    assert "last 1 session(s) per mouse" in out
    assert out.index("partial run") < out.index("[INFO] cohort-01 drifting problem 2")
    assert "m1: ses-6_date-20260106" in out


def test_a_selection_matching_no_session_stops_the_run_with_nothing_written(
    tmp_path, calls, cohort_extras
):
    with pytest.raises(NoSessionsSelectedError, match="2027-01-01"):
        run_cohort_incremental(
            1,
            results_root=tmp_path,
            trials_by_problem=blocked_then_two_drifting(),
            since="2027-01-01",
        )
    assert not list(tmp_path.iterdir())


# --- run_cohorts ---------------------------------------------------------------------------


def only_blocked_in_cohort_1(monkeypatch):
    """Cohort 1 has just its first problem (dated January 1-2); cohort 2 has all three."""

    def load(cohort_id, data_root, use_cache):
        cohort = blocked_then_two_drifting()
        return ({1: cohort[1]} if cohort_id == 1 else cohort), None

    monkeypatch.setattr(cohort_analysis, "load_cohort", load)


def test_dry_run_prints_the_plan_and_writes_nothing(
    tmp_path, monkeypatch, capsys, calls, cohort_extras
):
    only_blocked_in_cohort_1(monkeypatch)
    results = run_cohorts([2], results_root=tmp_path, last_sessions=2, dry_run=True)
    assert results == {}
    assert not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "cohort-02" in out and "drifting problem 2" in out and "dry run" in out


def test_dry_run_needs_a_selection(tmp_path):
    with pytest.raises(ValueError, match="dry_run"):
        run_cohorts([1], results_root=tmp_path, dry_run=True)


def test_a_cohort_without_matching_sessions_is_skipped_when_another_cohort_matches(
    tmp_path, monkeypatch, capsys, calls, cohort_extras, single_sessions
):
    only_blocked_in_cohort_1(monkeypatch)
    results = run_cohorts([1, 2], results_root=tmp_path, since="2026-01-05")
    assert list(results) == [2]
    out = capsys.readouterr().out
    assert "[SKIP] cohort-01" in out and not (tmp_path / "tables/open-field/cohort-01").exists()


def test_run_cohorts_raises_when_no_cohort_has_matching_sessions(
    tmp_path, monkeypatch, calls, cohort_extras
):
    only_blocked_in_cohort_1(monkeypatch)
    with pytest.raises(RuntimeError, match="2027-01-01"):
        run_cohorts([1, 2], results_root=tmp_path, since="2027-01-01")


def test_a_full_run_is_unchanged_by_the_incremental_options(
    tmp_path, monkeypatch, cohort_dir, calls, cohort_extras, single_sessions
):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda *a: (blocked_then_two_drifting(), None)
    )
    run_cohorts([1], results_root=tmp_path)
    assert problem_dirs(cohort_dir) == [
        "blocked/problem-01",
        "drifting/problem-01",
        "drifting/problem-02",
    ]


# --- failures and validation in run_cohorts -------------------------------------------------


def test_a_crashed_cohort_is_reported_together_with_the_no_match_error(
    tmp_path, monkeypatch, calls, cohort_extras
):
    def load(cohort_id, data_root, use_cache):
        if cohort_id == 1:
            raise OSError("raw folder unreadable")
        return {1: blocked_then_two_drifting()[1]}, None

    monkeypatch.setattr(cohort_analysis, "load_cohort", load)
    with pytest.raises(RuntimeError) as err:
        run_cohorts([1, 2], results_root=tmp_path, since="2027-01-01", continue_on_error=True)
    assert "cohort-01 failed" in str(err.value) and "raw folder unreadable" in str(err.value)
    assert "cohort-02" in str(err.value) and "2027-01-01" in str(err.value)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"last_sessions": 0}, "last_sessions"),
        ({"since": "16/10/2026"}, "YYYY-MM-DD"),
        ({"last_sessions": 2, "since": "2026-10-16"}, "exactly one"),
    ],
)
def test_bad_selection_options_are_rejected_before_any_cohort_is_loaded(
    options, match, tmp_path, monkeypatch
):
    def load(*args):
        raise AssertionError("a cohort was loaded")

    monkeypatch.setattr(cohort_analysis, "load_cohort", load)
    with pytest.raises(ValueError, match=match):
        run_cohorts([1, 2], results_root=tmp_path, **options)
    with pytest.raises(ValueError, match=match):
        run_cohorts([1, 2], results_root=tmp_path, dry_run=True, **options)


# --- the summary, and partial tables equal the full run's ------------------------------------


@pytest.fixture
def rows_in_every_table(monkeypatch):
    """Give the stubbed value analyses non-empty diagnostics, so their tables have rows."""
    monkeypatch.setitem(
        STUB_RESULTS, "run_reversal_aligned_arm_analysis", {"counts": pd.DataFrame({"n": [1, 2]})}
    )
    monkeypatch.setitem(
        STUB_RESULTS,
        "run_value_change_analysis",
        {
            "contamination_table": pd.DataFrame({"event": [1]}),
            "bin_counts_table": pd.DataFrame({"bin": [3, 4, 5]}),
            "symmetric_exclusions_table": pd.DataFrame({"n_events": [6]}),
        },
    )


def table_bytes(tables_dir):
    return {name: (tables_dir / f"{name}.csv").read_bytes() for name in DIAGNOSTICS}


def test_partial_run_tables_equal_the_full_run_tables_and_a_repeat_changes_nothing(
    tmp_path, rows_in_every_table, earlier_full_run
):
    full = table_bytes(earlier_full_run)
    assert all(len(pd.read_csv(earlier_full_run / f"{n}.csv")) for n in DIAGNOSTICS)
    for _ in range(2):
        rerun(tmp_path, {DRIFTING_2})
        assert table_bytes(earlier_full_run) == full
    rerun(tmp_path, {BLOCKED_1, DRIFTING_1})
    assert table_bytes(earlier_full_run) == full


def test_the_summary_says_the_pooled_analyses_were_regenerated_even_without_rank_results(
    tmp_path, earlier_full_run, capsys
):
    # cohort_extras stubs the pooled rank results to {}; the value analyses still wrote the pool
    rerun(tmp_path, {DRIFTING_2})
    assert "all-drifting-problems regenerated" in capsys.readouterr().out
    rerun(tmp_path, {BLOCKED_1})
    assert "all-drifting-problems not regenerated" in capsys.readouterr().out
