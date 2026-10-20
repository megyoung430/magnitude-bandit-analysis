# ruff: noqa: F811, F401  (fixtures imported from another test module are used as arguments)
"""The cohort runner's extra pooled drifting set without chosen mice, and ``pooled_only`` runs.

The fixture cohort (``tests/cohort_fixtures.py``) has mice m1 and m2, blocked problem 1 and
drifting problems 1 and 2. The drifting-only entry points are stubbed as in
``tests/test_cohort_runner_integration.py`` (they write ``stub.txt`` into their folder), whose
fixtures are reused; the sessions-per-problem figure is drawn for real.
"""

from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import src.pipelines.cohort_analysis as cohort_analysis
import src.pipelines.open_field as pipeline
from src.behavior_analysis.exclude_mice import ExcludedMiceError
from src.behavior_import.task_types import BLOCKED, DRIFTING
from src.behavior_visualization.plot_helpers import current_figure_banner
from src.pipelines.cohort_analysis import run_cohort, run_cohort_incremental, run_cohorts
from tests.cohort_fixtures import blocked_then_two_drifting
from tests.test_cohort_runner_integration import (
    ENTRY_POINTS,
    calls,
    close_figures,
    cohort_dir,
    cohort_extras,
    single_sessions,
)

# partial runs without an earlier full run and cohorts handed in without a log warn by design
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*no earlier table", "ignore:trials were passed in without an import_log"
)

POOLED = "all-drifting-problems"
WITHOUT_M2 = "all-drifting-problems-without-m2"
SESSIONS_PNG = "sessions-per-problem/Sessions per Drifting Problem.png"
RUN_AT = datetime(2026, 10, 20, 9, 30, 5)


@pytest.fixture
def banners(calls, monkeypatch):
    """Record the figure banner active at each drifting-entry-point call: ``(out_dir, banner)``."""
    seen = []

    def spy(name):
        stub = getattr(pipeline, name)

        def wrapped(data, out_dir, **kwargs):
            seen.append((out_dir, current_figure_banner()))
            return stub(data, out_dir, **kwargs)

        return wrapped

    # wrap the recorders that `calls` installed
    for name in ENTRY_POINTS:
        monkeypatch.setattr(pipeline, name, spy(name))
    return seen


@pytest.fixture
def tables_dir(tmp_path):
    return tmp_path / "tables" / "open-field" / "cohort-01"


def run(tmp_path, **settings):
    return run_cohort(
        1,
        results_root=tmp_path,
        trials_by_problem=blocked_then_two_drifting(),
        run_at=RUN_AT,
        **settings,
    )


def mice_of(data):
    """The mice in the data an entry point got (a problem dict, or the one pooled entry)."""
    return sorted({m for subjects in data.values() for m in subjects})


# --- the extra set ---------------------------------------------------------------------------


def test_the_extra_set_has_its_own_folder_next_to_the_normal_one(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    drifting = cohort_dir / DRIFTING
    for folder in (POOLED, WITHOUT_M2):
        for sub in ("value-psychometrics", "reversal-aligned-arms", "value-change"):
            assert (drifting / folder / sub / "stub.txt").exists(), (folder, sub)
        assert (drifting / folder / "port-preference" / "stub.txt").exists()
        assert (drifting / folder / SESSIONS_PNG).exists(), folder


def test_the_extra_set_pools_the_drifting_problems_without_the_mice(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    by_folder = {}
    for name, out_dir, data in calls:
        if name == "run_reversal_aligned_arm_analysis":
            continue  # gets pre-pooled windows, checked through the other entry points
        by_folder.setdefault(out_dir.parent.name, []).append((sorted(data), mice_of(data)))
    assert by_folder[POOLED] and by_folder[WITHOUT_M2]
    assert all(entry == ([1, 2], ["m1", "m2"]) for entry in by_folder[POOLED])
    assert all(entry == ([1, 2], ["m1"]) for entry in by_folder[WITHOUT_M2])


def test_every_figure_of_the_extra_set_is_marked_and_the_normal_set_is_not(
    tmp_path, calls, cohort_extras, single_sessions, banners
):
    run(tmp_path, exclude_mice=["m2"])
    marked = {out_dir.parent.name: banner for out_dir, banner in banners}
    # the problem folders and the normal pool run with no banner; the extra set carries it
    assert marked[WITHOUT_M2] == "without m2"
    assert marked[POOLED] is None
    assert all(b is None for out_dir, b in banners if out_dir.parent.name != WITHOUT_M2)


def test_the_titles_of_the_scoped_figures_name_the_exclusion(
    tmp_path, calls, cohort_extras, single_sessions, monkeypatch
):
    labels = {}
    stub = pipeline.run_reversal_aligned_arm_analysis

    def spy(data, out_dir, **kwargs):
        labels[out_dir.parent.name] = kwargs["scope_label"]
        return stub(data, out_dir, **kwargs)

    monkeypatch.setattr(pipeline, "run_reversal_aligned_arm_analysis", spy)
    run(tmp_path, exclude_mice=["m2"])
    assert labels[POOLED] == "all drifting problems"
    assert labels[WITHOUT_M2] == "all drifting problems without m2"


def test_the_extra_sessions_per_problem_table_leaves_out_the_mice(
    tmp_path, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    extra = pd.read_csv(tables_dir / WITHOUT_M2 / "sessions_per_problem.csv")
    assert set(extra["mouse"]) == {"m1"}
    # the cohort-wide table keeps every mouse
    assert set(pd.read_csv(tables_dir / "sessions_per_problem.csv")["mouse"]) == {"m1", "m2"}


def test_the_extra_set_lists_the_excluded_mice_and_the_run_time(
    tmp_path, cohort_dir, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    for folder in (cohort_dir / DRIFTING / WITHOUT_M2, tables_dir / WITHOUT_M2):
        text = (folder / "excluded_mice.txt").read_text()
        assert "m2" in text and "2026-10-20 09:30:05" in text and "cohort-01" in text


def test_the_extra_tables_are_not_in_the_stacked_diagnostic_tables(
    tmp_path, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    top_level = {p.name for p in tables_dir.glob("*.csv")}
    inside = {p.name for p in (tables_dir / WITHOUT_M2).glob("*.csv")}
    assert "sessions_per_problem.csv" in inside
    assert "reversal_aligned_arms_counts.csv" in inside and "value_change_events.csv" in inside
    # the stacked ones come from the per-problem runs alone: same files and rows as without
    # the extra set
    assert {"reversal_aligned_arms_counts.csv", "value_change_events.csv"} <= top_level


def test_the_normal_set_is_unchanged_by_asking_for_the_extra_one(
    tmp_path, calls, cohort_extras, single_sessions
):
    plain, extra = tmp_path / "plain", tmp_path / "extra"
    run(plain)
    run(extra, exclude_mice=["m2"])

    def tree(root, keep):
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and keep(p.relative_to(root))
        }

    def normal_table(rel):
        return rel.suffix == ".csv" and WITHOUT_M2 not in rel.parts

    def normal_stub(rel):
        return rel.suffix == ".txt" and WITHOUT_M2 not in rel.parts

    assert tree(plain, normal_table) == tree(extra, normal_table)
    assert tree(plain, normal_stub) == tree(extra, normal_stub)

    def figure_files(root):
        return {str(p.relative_to(root)) for p in (root / "figures").rglob("*") if p.is_file()}

    # every figure file of the plain run is in the extra run, and the only new ones are the extra
    # set's
    new = figure_files(extra) - figure_files(plain)
    assert figure_files(plain) <= figure_files(extra)
    assert new and all(WITHOUT_M2 in path for path in new)


def test_without_the_option_no_extra_folder_is_written(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    assert [p.name for p in (cohort_dir / DRIFTING).glob("all-drifting-problems*")] == [POOLED]


# --- checks before anything is written ---------------------------------------------------------


def test_a_mouse_that_is_not_in_the_cohort_raises_before_anything_is_written(
    tmp_path, calls, cohort_extras, single_sessions
):
    with pytest.raises(ValueError, match="MY_9_Z.*m1, m2"):
        run(tmp_path, exclude_mice=["MY_9_Z"])
    assert not list(tmp_path.iterdir())


def test_excluding_every_mouse_raises_before_anything_is_written(
    tmp_path, calls, cohort_extras, single_sessions
):
    with pytest.raises(ValueError, match="no mice left"):
        run(tmp_path, exclude_mice=["m1", "m2"])
    assert not list(tmp_path.iterdir())


def test_an_empty_exclusion_list_raises(tmp_path, calls, cohort_extras, single_sessions):
    with pytest.raises(ValueError, match="no mice"):
        run(tmp_path, exclude_mice=[])
    assert not list(tmp_path.iterdir())


def test_a_cohort_without_drifting_problems_cannot_exclude_mice(
    tmp_path, calls, cohort_extras, single_sessions
):
    only_blocked = {1: blocked_then_two_drifting()[1]}
    with pytest.raises(ValueError, match="m2"):
        run_cohort(1, results_root=tmp_path, trials_by_problem=only_blocked, exclude_mice=["m2"])
    assert not list(tmp_path.iterdir())


# --- pooled_only -------------------------------------------------------------------------------


def test_a_pooled_only_run_writes_no_problem_folders(
    tmp_path, cohort_dir, tables_dir, calls, cohort_extras, single_sessions
):
    result = run(tmp_path, pooled_only=True, exclude_mice=["m2"])
    assert not list(cohort_dir.glob("*/problem-*"))
    assert single_sessions == []
    assert (cohort_dir / DRIFTING / POOLED / "value-change" / "stub.txt").exists()
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "stub.txt").exists()
    assert (cohort_dir / DRIFTING / POOLED / SESSIONS_PNG).exists()
    assert not (cohort_dir / "trial-block-summaries").exists()
    assert result.rank_results == {}


def test_a_pooled_only_run_writes_the_review_tables_but_leaves_the_stacked_diagnostics(
    tmp_path, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)  # an earlier full run leaves the diagnostic tables
    marker = pd.DataFrame({"kept": ["from the full run"]})
    for name in ("poke_trial_summary", "value_change_events"):
        marker.to_csv(tables_dir / f"{name}.csv", index=False)
    (tables_dir / "problem_map.csv").unlink()
    run(tmp_path, pooled_only=True)
    assert len(pd.read_csv(tables_dir / "problem_map.csv")) == 6
    for name in ("poke_trial_summary", "value_change_events"):
        assert pd.read_csv(tables_dir / f"{name}.csv").equals(marker)


def test_a_pooled_only_run_without_the_option_does_the_normal_pool_only(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, pooled_only=True)
    assert [p.name for p in (cohort_dir / DRIFTING).iterdir()] == [POOLED]


def test_pooled_only_cannot_be_combined_with_a_problem_selection(
    tmp_path, calls, cohort_extras, single_sessions
):
    with pytest.raises(ValueError, match="pooled_only"):
        run(tmp_path, pooled_only=True, only_problems={(DRIFTING, 1)})
    assert not list(tmp_path.iterdir())


def test_pooled_only_needs_drifting_problems(tmp_path, calls, cohort_extras, single_sessions):
    only_blocked = {1: blocked_then_two_drifting()[1]}
    with pytest.raises(ValueError, match="drifting"):
        run_cohort(1, results_root=tmp_path, trials_by_problem=only_blocked, pooled_only=True)
    assert not list(tmp_path.iterdir())


# --- partial runs ------------------------------------------------------------------------------


def incremental(tmp_path, **kwargs):
    return run_cohort_incremental(
        1,
        results_root=tmp_path,
        trials_by_problem=blocked_then_two_drifting(),
        run_at=RUN_AT,
        **kwargs,
    )


def test_a_partial_run_regenerates_the_extra_set_when_the_pooled_set_is_regenerated(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    incremental(tmp_path, last_sessions=1, exclude_mice=["m2"])  # the last session is drifting
    assert (cohort_dir / DRIFTING / POOLED / "value-change" / "stub.txt").exists()
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "stub.txt").exists()


def test_a_blocked_only_partial_run_redraws_the_extra_sessions_per_problem_but_not_its_analyses(
    tmp_path, cohort_dir, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    extra_table = tables_dir / WITHOUT_M2 / "sessions_per_problem.csv"
    extra_table.write_text("mouse\nstale\n")  # a table an earlier run left behind
    (cohort_dir / DRIFTING / WITHOUT_M2 / SESSIONS_PNG).unlink()
    calls.clear()
    run(tmp_path, only_problems={(BLOCKED, 1)}, exclude_mice=["m2"])
    assert calls == []  # the analyses are not rerun ...
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "stub.txt").exists()  # or cleared
    # ... but the sessions-per-problem figure and table are redrawn, as the plain ones are
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / SESSIONS_PNG).exists()
    assert set(pd.read_csv(extra_table)["mouse"]) == {"m1"}


def test_a_first_blocked_only_partial_run_writes_the_extra_sessions_per_problem_alone(
    tmp_path, cohort_dir, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, only_problems={(BLOCKED, 1)}, exclude_mice=["m2"])
    folder = cohort_dir / DRIFTING / WITHOUT_M2
    assert {p.name for p in folder.iterdir()} == {"sessions-per-problem", "excluded_mice.txt"}
    assert (tables_dir / WITHOUT_M2 / "sessions_per_problem.csv").exists()


def test_a_rerun_clears_the_extra_figure_and_table_folders_first(
    tmp_path, cohort_dir, tables_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    stale_figure = cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "left over.png"
    stale_table = tables_dir / WITHOUT_M2 / "left_over.csv"
    stale_figure.write_text("x")
    stale_table.write_text("x")
    run(tmp_path, exclude_mice=["m2"])
    assert not stale_figure.exists() and not stale_table.exists()
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "stub.txt").exists()


def test_the_plain_pool_folders_are_not_cleared(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=["m2"])
    kept = cohort_dir / DRIFTING / POOLED / "value-change" / "kept.txt"
    kept.write_text("x")
    run(tmp_path, exclude_mice=["m2"])
    assert kept.exists()


def test_the_mice_may_come_as_a_generator(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path, exclude_mice=(m for m in ["m2"]))
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "stub.txt").exists()


def test_a_generator_of_mice_is_read_once_by_a_partial_run(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    incremental(tmp_path, last_sessions=1, exclude_mice=(m for m in ["m2"]))
    assert (cohort_dir / DRIFTING / WITHOUT_M2 / "value-change" / "stub.txt").exists()


def test_a_blocked_only_partial_run_still_checks_the_names(
    tmp_path, calls, cohort_extras, single_sessions
):
    with pytest.raises(ValueError, match="MY_9_Z"):
        run(tmp_path, only_problems={(BLOCKED, 1)}, exclude_mice=["MY_9_Z"])
    assert not list(tmp_path.iterdir())


def test_the_plan_of_a_partial_run_names_the_extra_set(
    tmp_path, calls, cohort_extras, single_sessions, capsys
):
    incremental(tmp_path, last_sessions=1, exclude_mice=["m2"])
    out = capsys.readouterr().out
    assert f"{DRIFTING}/{WITHOUT_M2}: will be regenerated" in out


# --- run_cohorts -------------------------------------------------------------------------------


def test_run_cohorts_passes_the_options_to_every_cohort(
    tmp_path, monkeypatch, calls, cohort_extras, single_sessions
):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda *a: (blocked_then_two_drifting(), None)
    )
    with pytest.warns(UserWarning):  # no import log
        run_cohorts([1], results_root=tmp_path, pooled_only=True, exclude_mice=["m2"])
    assert (tmp_path / "figures/open-field/cohort-01" / DRIFTING / WITHOUT_M2).is_dir()
    assert not list((tmp_path / "figures").glob("*/cohort-01/*/problem-*"))


def test_a_pooled_only_dry_run_prints_the_plan_and_writes_nothing(
    tmp_path, monkeypatch, capsys, calls, cohort_extras
):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda *a: (blocked_then_two_drifting(), None)
    )
    results = run_cohorts(
        [1], results_root=tmp_path, pooled_only=True, exclude_mice=["m2"], dry_run=True
    )
    assert results == {}
    assert not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "pooled only" in out and "dry run" in out
    assert f"{DRIFTING}/{POOLED}" in out and f"{DRIFTING}/{WITHOUT_M2}" in out
    assert "m1" in out  # the mice that stay


def test_a_dry_run_names_the_extra_set_of_a_partial_run(
    tmp_path, monkeypatch, capsys, calls, cohort_extras
):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda *a: (blocked_then_two_drifting(), None)
    )
    run_cohorts([1], results_root=tmp_path, last_sessions=1, exclude_mice=["m2"], dry_run=True)
    assert f"{DRIFTING}/{WITHOUT_M2}: will be regenerated" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [{"pooled_only": True}, {"last_sessions": 1}])
def test_a_dry_run_checks_the_names_too(tmp_path, monkeypatch, calls, cohort_extras, options):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda *a: (blocked_then_two_drifting(), None)
    )
    with pytest.raises(ExcludedMiceError, match="MY_9_Z"):
        run_cohorts([1], results_root=tmp_path, exclude_mice=["MY_9_Z"], dry_run=True, **options)
    assert not list(tmp_path.iterdir())


def test_a_dry_run_that_leaves_no_mouse_says_so(tmp_path, monkeypatch, calls, cohort_extras):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda *a: (blocked_then_two_drifting(), None)
    )
    with pytest.raises(ExcludedMiceError, match="no mice left"):
        run_cohorts(
            [1], results_root=tmp_path, pooled_only=True, exclude_mice=["m1", "m2"], dry_run=True
        )


def test_pooled_only_with_a_selection_raises_before_any_cohort_is_loaded(tmp_path):
    with pytest.raises(ValueError, match="pooled_only"):
        run_cohorts([1], results_root=tmp_path, pooled_only=True, last_sessions=2)
    with pytest.raises(ValueError, match="pooled_only"):
        run_cohorts([1], results_root=tmp_path, pooled_only=True, since="2026-01-01")


def test_an_invalid_mouse_id_raises_before_any_cohort_is_loaded(tmp_path):
    with pytest.raises(ValueError, match="not a mouse id"):
        run_cohorts([1], results_root=tmp_path, exclude_mice=["../x"])


# --- the names are checked against every cohort first ----------------------------------------


def test_every_listed_cohort_is_checked_before_the_first_one_runs(tmp_path, monkeypatch):
    cohorts = {1: blocked_then_two_drifting(), 2: blocked_then_two_drifting()}
    for problem in cohorts[2].values():
        problem.pop("m2")  # cohort 2 has no m2
    monkeypatch.setattr(cohort_analysis, "load_cohort", lambda c, *a: (cohorts[c], None))
    ran = []
    monkeypatch.setattr(cohort_analysis, "run_cohort", lambda i, **k: ran.append(i))
    with pytest.raises(ExcludedMiceError, match="cohort-02.*m2"):
        run_cohorts([1, 2], results_root=tmp_path, exclude_mice=["m2"])
    assert ran == [] and not list(tmp_path.iterdir())


def test_a_generator_of_mice_reaches_every_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cohort_analysis, "load_cohort", lambda c, *a: (blocked_then_two_drifting(), None)
    )
    seen = []
    monkeypatch.setattr(
        cohort_analysis, "run_cohort", lambda i, **k: seen.append(k["exclude_mice"])
    )
    run_cohorts([1, 2], results_root=tmp_path, exclude_mice=(m for m in ["m2"]))
    assert seen == [("m2",), ("m2",)]
