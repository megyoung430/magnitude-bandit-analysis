"""The cohort runner: analyses chosen by task type, pooled drifting ranks, outputs, errors."""

import warnings
from pathlib import Path

import pandas as pd
import pytest

import src.behavior_import.task_types as task_types
import src.pipelines.cohort_analysis as cohort_analysis
import src.pipelines.open_field as pipeline
from src.behavior_analysis.get_good_reversal_info import ReversalWarning
from src.behavior_analysis.suspect_reversal_table import COLUMNS
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import ImportedTrials
from src.behavior_import.task_types import BLOCKED, DRIFTING
from src.pipelines.cohort_analysis import (
    cohort_name,
    cohort_raw_dir,
    pooled_drifting_rank_results,
    run_cohort,
    run_cohorts,
    selection_for_task_type,
    subject_numbers_from_sessions,
    suspect_table_with_task_types,
)

BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"
SECTIONS = [
    "run_choice_probability_analysis",
    "run_first_leave_analysis",
    "run_number_of_reversals_analysis",
    "run_rank_proportion_analysis",
    "run_single_session_analysis",
]


def session(task_name, subject_number, problem):
    frame = pd.DataFrame({"type": [], "subtype": [], "content": []})
    frame.attrs["source_path"] = f"/raw/sub-{subject_number:02d}_id-M{subject_number}/x.tsv"
    return {
        "task_name": task_name,
        "df": frame,
        "problem": problem,
        "choice_towers": ["A1", "B1", "C1"],
    }


def mouse_problems(subject_number, task_names):
    """``{problem: {subject: {session_key: session}}}`` for one mouse."""
    subject = f"M{subject_number}"
    return {
        p: {subject: {f"ses-{p}_date-2026010{p}": session(name, subject_number, p)}}
        for p, name in enumerate(task_names, start=1)
    }


def two_mice():
    """A: blocked, blocked, drifting.  B: blocked x3, drifting."""
    a = mouse_problems(1, [BLOCKED_NAME, BLOCKED_NAME, DRIFTING_NAME])
    b = mouse_problems(2, [BLOCKED_NAME] * 3 + [DRIFTING_NAME])
    merged = {}
    for one in (a, b):
        for p, subjects in one.items():
            merged.setdefault(p, {}).update(subjects)
    return merged


@pytest.fixture
def sections(monkeypatch):
    """Replace every analysis section by a recorder of ``(name, out_dir, selection)``.

    The poke analysis and the drifting-only value analyses are replaced by no-ops: the sessions
    here have no raw events (``tests/test_cohort_runner_integration.py`` runs them).
    """
    seen = []
    monkeypatch.setattr(
        pipeline,
        "run_poke_analysis",
        lambda *a, **k: {"summary": pd.DataFrame(), "port_finding_check": pd.DataFrame()},
    )
    monkeypatch.setattr(pipeline, "mouse_poke_counts", lambda *a, **k: {})
    for name in ("run_drifting_value_analyses",):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: {})
        monkeypatch.setattr(cohort_analysis, name, lambda *a, **k: {})
    for name in SECTIONS:
        monkeypatch.setattr(
            pipeline, name, lambda *a, _n=name, **k: seen.append((_n, a[1], a[2:])) or {}
        )
    return seen


@pytest.fixture
def no_extras(monkeypatch):
    """Replace the cohort-level pooled ranks, suspect table and summaries by stubs."""
    monkeypatch.setattr(cohort_analysis, "plot_pooled_drifting_rank", lambda *a, **k: None)
    monkeypatch.setattr(
        cohort_analysis,
        "suspect_reversal_table_by_problem",
        lambda by_k, **k: pd.DataFrame(
            [
                {"problem": p, **dict.fromkeys(COLUMNS), "subject": s}
                for p, subs in by_k.items()
                for s in subs
            ],
            columns=["problem", *COLUMNS],
        ),
    )
    monkeypatch.setattr(cohort_analysis, "pooled_drifting_rank_results", lambda *a, **k: {})
    monkeypatch.setattr(cohort_analysis, "plot_trial_block_summaries", lambda *a, **k: [])
    monkeypatch.setattr(cohort_analysis, "build_trial_block_tables", lambda *a, **k: (None, None))


# --- paths and selections ---------------------------------------------------


def test_cohort_name_and_raw_dir():
    assert cohort_name(3) == "cohort-03"
    root = Path("/data")
    assert cohort_raw_dir(root, 4) == root / "3x3_field_magnitude_bandit/rawdata/cohort-04"


def test_cohort_name_rejects_non_positive_ids():
    with pytest.raises(ValueError, match="cohort"):
        cohort_name(0)


def test_selection_for_task_type_follows_the_notebook_settings():
    blocked = selection_for_task_type(BLOCKED)
    assert blocked.run_choice_probability_analysis and blocked.run_first_leave_analysis
    assert blocked.rank_value_bases == ("true",)
    drifting = selection_for_task_type(DRIFTING)
    assert not drifting.run_choice_probability_analysis and not drifting.run_first_leave_analysis
    assert drifting.rank_value_bases == ("true", "believed")
    for selection in (blocked, drifting):
        assert selection.rank_pooling_modes == (True, False)
        assert selection.run_number_of_reversals_analysis
        assert selection.run_single_session_analysis


def test_selection_for_unknown_task_type_raises():
    with pytest.raises(ValueError, match="task type"):
        selection_for_task_type("pretraining")


# --- analyses per task type ---------------------------------------------------


def test_blocked_runs_choice_probabilities_and_drifting_does_not(tmp_path, sections, no_extras):
    run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    by_dir = {}
    for name, out_dir, _ in sections:
        by_dir.setdefault(out_dir.parent.parent.name + "/" + out_dir.parent.name, []).append(name)
    assert by_dir["blocked/problem-01"] == SECTIONS
    assert by_dir["blocked/problem-02"] == SECTIONS
    assert by_dir["blocked/problem-03"] == SECTIONS
    assert by_dir["drifting/problem-01"] == SECTIONS[2:]


def test_rank_analysis_gets_the_selection_of_its_type(tmp_path, sections, no_extras):
    run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    bases = {
        out_dir.parent.parent.name: rest[0].rank_value_bases
        for name, out_dir, rest in sections
        if name == "run_rank_proportion_analysis"
    }
    assert bases == {BLOCKED: ("true",), DRIFTING: ("true", "believed")}


def test_figures_go_under_cohort_type_and_within_type_problem(tmp_path, sections, no_extras):
    run_cohort(2, results_root=tmp_path, trials_by_problem=two_mice())
    roots = {out_dir.parent.parent.parent for _, out_dir, _ in sections}
    assert roots == {tmp_path / "figures" / "open-field" / "cohort-02"}
    drifting_dirs = {
        out_dir.parent for _, out_dir, _ in sections if out_dir.parent.parent.name == DRIFTING
    }
    assert drifting_dirs == {tmp_path / "figures/open-field/cohort-02/drifting/problem-01"}


# --- tables ---------------------------------------------------------------------


def test_problem_map_csv(tmp_path, sections, no_extras):
    run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    table = pd.read_csv(tmp_path / "tables/open-field/cohort-01/problem_map.csv")
    assert list(table.columns) == [
        "mouse",
        "task_type",
        "problem_in_type",
        "overall_problem",
        "sessions",
        "initiation_tower",
        "choice_towers",
        "n_sessions",
        "session_range",
    ]
    drifting = table[table.task_type == DRIFTING].set_index("mouse")
    assert drifting.loc["M1", "choice_towers"] == "A1;B1;C1"
    assert drifting.loc["M1", "initiation_tower"] == "B2"
    assert drifting.loc["M1", "n_sessions"] == 1
    assert drifting.loc["M1", "overall_problem"] == 3
    assert drifting.loc["M2", "overall_problem"] == 4
    assert drifting.problem_in_type.tolist() == [1, 1]
    assert len(table) == 3 + 4  # mouse A has 3 problems, mouse B has 4


def test_suspect_table_has_task_type_and_within_type_problem(tmp_path, sections, no_extras):
    run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    table = pd.read_csv(tmp_path / "tables/open-field/cohort-01/suspect_reversals.csv")
    assert list(table.columns) == [
        "task_type",
        "problem_in_type",
        "overall_problem",
        *COLUMNS,
    ]
    assert len(table) == 7  # one stub row per mouse and problem
    row = table[(table.task_type == DRIFTING) & (table.subject == "M2")].iloc[0]
    assert (row.problem_in_type, row.overall_problem) == (1, 4)


def test_suspect_table_with_no_problems_keeps_the_columns():
    grouped = {}
    empty = suspect_table_with_task_types(grouped, cohort_analysis.problem_map_table({}))
    assert list(empty.columns) == [
        "task_type",
        "problem_in_type",
        "overall_problem",
        *COLUMNS,
    ]
    assert len(empty) == 0


# --- trial and block summaries ----------------------------------------------------


def test_subject_numbers_come_from_the_session_paths():
    assert subject_numbers_from_sessions(two_mice()) == {"M1": 1, "M2": 2}


def test_subject_numbers_raise_when_a_mouse_has_no_path():
    by_problem = {1: {"A": {"s": {"task_name": DRIFTING_NAME}}}}
    with pytest.raises(ValueError, match="A"):
        subject_numbers_from_sessions(by_problem)


def test_summaries_cover_all_problems_and_go_to_the_all_problems_folder(
    tmp_path, sections, no_extras, monkeypatch
):
    seen = {}
    monkeypatch.setattr(
        cohort_analysis,
        "build_trial_block_tables",
        lambda problems, numbers: seen.update(problems=problems, numbers=numbers) or (1, 2),
    )
    monkeypatch.setattr(
        cohort_analysis,
        "plot_trial_block_summaries",
        lambda s, b, save_dir, show: seen.update(save_dir=save_dir, show=show) or [],
    )
    run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    assert sorted(seen["problems"]) == [1, 2, 3, 4]
    assert seen["numbers"] == {"M1": 1, "M2": 2}
    expected = tmp_path / "figures/open-field/cohort-01/trial-block-summaries/all-problems"
    assert seen["save_dir"] == expected
    assert seen["show"] is False


# --- pooled drifting ranks ----------------------------------------------------------


class RankRecorder:
    """Stand-ins for the rank computations that remember what they were given."""

    def __init__(self, monkeypatch):
        self.windows_calls, self.count_calls, self.p_calls = [], [], []
        self.proportion_calls, self.significance_calls = [], []
        monkeypatch.setattr(cohort_analysis, "get_good_reversal_info", self.windows)
        monkeypatch.setattr(cohort_analysis, "get_rank_counts_by_value", self.counts)
        monkeypatch.setattr(cohort_analysis, "pvalue_paired_t_best_vs_second_vs_third", self.p)
        monkeypatch.setattr(cohort_analysis, "proportions_by_mouse", self.proportions)
        monkeypatch.setattr(cohort_analysis, "rank_comparison_tests", self.significance)

    def windows(self, trials, include_first_block):
        assert include_first_block is True
        self.windows_calls.append(trials)
        return {"windows-of": trials}

    def counts(self, trials, windows, value_basis):
        assert windows == {"windows-of": trials}
        self.count_calls.append((trials, value_basis))
        # one entry per mouse, labelled with the problem's own marker
        return {s: [f"{trials['marker']}-{value_basis}"] for s in trials["mice"]}

    def p(self, counts, pool_trials_within_subject):
        self.p_calls.append((counts, pool_trials_within_subject))
        return {"best_vs_second": 0.5}

    def proportions(self, counts, pool_trials_within_subject):
        self.proportion_calls.append((counts, pool_trials_within_subject))
        return {"proportions-of": (pool_trials_within_subject,)}

    def significance(self, per_mouse):
        self.significance_calls.append(per_mouse)
        return {"tests-of": per_mouse}


def test_pooled_drifting_ranks_are_computed_within_each_problem_then_combined(monkeypatch):
    rec = RankRecorder(monkeypatch)
    p1 = {"marker": "p1", "mice": ["A", "B"]}
    p2 = {"marker": "p2", "mice": ["A"]}
    results = pooled_drifting_rank_results({1: p1, 2: p2}, ("true",), (True, False))
    # ranks come from each problem on its own, never from a merged problem
    assert rec.windows_calls == [p1, p2]
    assert rec.count_calls == [(p1, "true"), (p2, "true")]
    counts = results["true"][True]["counts"]
    assert counts == {"A": ["p1-true", "p2-true"], "B": ["p1-true"]}
    assert results["true"][False]["counts"] == counts
    assert results["true"][True]["problems"] == [1, 2]


def test_pooled_drifting_ranks_cover_both_bases_and_modes_with_p_values(monkeypatch):
    rec = RankRecorder(monkeypatch)
    problems = {1: {"marker": "p1", "mice": ["A"]}}
    results = pooled_drifting_rank_results(problems, ("true", "believed"), (True, False))
    assert set(results) == {"true", "believed"}
    for basis in results:
        assert set(results[basis]) == {True, False}
        assert results[basis][True]["p_values"] == {"best_vs_second": 0.5}
        # the paper-ready Wilcoxon tests use the per-mouse proportions of the same pooling mode
        for mode in (True, False):
            assert results[basis][mode]["significance"] == {"tests-of": {"proportions-of": (mode,)}}
    assert [mode for _, mode in rec.p_calls] == [True, False, True, False]
    assert [mode for _, mode in rec.proportion_calls] == [True, False, True, False]
    assert [counts for counts, _ in rec.proportion_calls] == [c for c, _ in rec.p_calls]


def test_pooled_drifting_ranks_skip_a_basis_without_windows(monkeypatch):
    RankRecorder(monkeypatch)
    results = pooled_drifting_rank_results({1: {"marker": "p", "mice": []}}, ("true",), (True,))
    assert results == {}


def test_run_cohort_pools_every_drifting_problem_of_every_mouse(tmp_path, sections, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cohort_analysis,
        "pooled_drifting_rank_results",
        lambda problems, bases, modes: (
            seen.update(problems=problems, bases=bases, modes=modes) or {}
        ),
    )
    monkeypatch.setattr(
        cohort_analysis,
        "plot_pooled_drifting_rank",
        lambda results, save_dir: seen.update(save_dir=save_dir),
    )
    monkeypatch.setattr(
        cohort_analysis, "suspect_table_with_task_types", lambda g, m: pd.DataFrame(columns=COLUMNS)
    )
    monkeypatch.setattr(cohort_analysis, "plot_trial_block_summaries", lambda *a, **k: [])
    monkeypatch.setattr(cohort_analysis, "build_trial_block_tables", lambda *a, **k: (None, None))
    by_problem = two_mice()
    by_problem[5] = {"M1": {"ses-5_date-20260105": session(DRIFTING_NAME, 1, 5)}}
    run_cohort(1, results_root=tmp_path, trials_by_problem=by_problem)
    assert list(seen["problems"]) == [1, 2]  # M1 has two drifting problems
    assert set(seen["problems"][1]) == {"M1", "M2"}
    assert seen["bases"] == ("true", "believed") and seen["modes"] == (True, False)
    assert seen["save_dir"] == (
        tmp_path / "figures/open-field/cohort-01/drifting/all-drifting-problems/choice-stats"
    )


def test_run_cohort_loads_with_task_types_classified(tmp_path, monkeypatch):
    seen = {}

    def fake_load(root, **kwargs):
        seen.update(root=root, **kwargs)
        return ImportedTrials({}, ImportLog())

    monkeypatch.setattr(cohort_analysis, "load_trials_with_log", fake_load)
    with pytest.raises(ValueError, match="No problems"):
        run_cohort(2, data_root=tmp_path, results_root=tmp_path)
    assert seen["classify_task_types"] is True
    assert seen["root"] == cohort_raw_dir(tmp_path, 2)


# --- numbering and warnings happen once -----------------------------------------------


def test_run_cohort_numbers_the_problems_once(tmp_path, sections, no_extras, monkeypatch):
    calls = []
    real = task_types.number_within_task_type
    monkeypatch.setattr(
        task_types, "number_within_task_type", lambda data: calls.append(1) or real(data)
    )
    monkeypatch.setattr(
        cohort_analysis, "number_within_task_type", task_types.number_within_task_type
    )
    run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    assert len(calls) == 1


def test_a_given_numbering_is_used_instead_of_recomputing_it():
    by_problem = two_mice()
    numbering = task_types.number_within_task_type(by_problem)
    assert list(task_types.regroup_by_task_type(by_problem, numbering)) == list(
        task_types.regroup_by_task_type(by_problem)
    )
    assert task_types.problem_map_table(by_problem, numbering).equals(
        task_types.problem_map_table(by_problem)
    )


def test_each_reversal_warning_is_shown_once_per_cohort(tmp_path, sections, no_extras, monkeypatch):
    def noisy(*args, **kwargs):
        warnings.warn("dropped 1 good reversal(s): M1 ses-1", ReversalWarning, stacklevel=2)
        warnings.warn("dropped 2 good reversal(s): M2 ses-1", ReversalWarning, stacklevel=2)
        warnings.warn("something else", UserWarning, stacklevel=2)

    for name in SECTIONS:
        monkeypatch.setattr(pipeline, name, noisy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
    reversal = [str(w.message) for w in caught if issubclass(w.category, ReversalWarning)]
    assert sorted(reversal) == [
        "dropped 1 good reversal(s): M1 ses-1",
        "dropped 2 good reversal(s): M2 ses-1",
    ]
    other = [w for w in caught if w.category is UserWarning and "something else" in str(w.message)]
    assert len(other) > 2  # other warnings are left alone


def test_a_reversal_warning_is_shown_again_in_the_next_cohort(
    tmp_path, sections, no_extras, monkeypatch
):
    def noisy(*args, **kwargs):
        warnings.warn("dropped 1 good reversal(s): M1 ses-1", ReversalWarning, stacklevel=2)

    for name in SECTIONS:
        monkeypatch.setattr(pipeline, name, noisy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cohort(1, results_root=tmp_path, trials_by_problem=two_mice())
        run_cohort(2, results_root=tmp_path, trials_by_problem=two_mice())
    assert len([w for w in caught if issubclass(w.category, ReversalWarning)]) == 2


# --- errors -----------------------------------------------------------------------------


def test_no_problems_raises(tmp_path, no_extras):
    with pytest.raises(ValueError, match="No problems"):
        run_cohort(1, results_root=tmp_path, trials_by_problem={})


def test_unknown_task_name_raises_before_any_output(tmp_path, sections, no_extras):
    by_problem = mouse_problems(1, ["Meg\\mystery_task"])
    with pytest.raises(ValueError, match="mystery_task"):
        run_cohort(1, results_root=tmp_path, trials_by_problem=by_problem)
    assert not list(tmp_path.iterdir())
    assert sections == []


def test_run_cohorts_stops_at_the_first_failure_and_names_the_cohort(monkeypatch):
    ran = []

    def fake(cohort_id, **kwargs):
        ran.append(cohort_id)
        if cohort_id == 2:
            raise ValueError("bad task name")
        return f"done {cohort_id}"

    monkeypatch.setattr(cohort_analysis, "run_cohort", fake)
    with pytest.raises(RuntimeError, match=r"cohort-02.*bad task name") as err:
        run_cohorts([1, 2, 3])
    assert ran == [1, 2]
    assert isinstance(err.value.__cause__, ValueError)


def test_run_cohorts_returns_each_result(monkeypatch):
    monkeypatch.setattr(cohort_analysis, "run_cohort", lambda i, **k: f"done {i}")
    assert run_cohorts([1, 3]) == {1: "done 1", 3: "done 3"}


def test_run_cohorts_can_continue_and_still_reports_the_failures(monkeypatch):
    ran = []

    def fake(cohort_id, **kwargs):
        ran.append(cohort_id)
        if cohort_id in (1, 3):
            raise ValueError(f"broken {cohort_id}")
        return "ok"

    monkeypatch.setattr(cohort_analysis, "run_cohort", fake)
    with pytest.raises(RuntimeError) as err:
        run_cohorts([1, 2, 3], continue_on_error=True)
    assert ran == [1, 2, 3]
    assert "cohort-01" in str(err.value) and "cohort-03" in str(err.value)
    assert "broken 1" in str(err.value) and "broken 3" in str(err.value)


def test_run_cohorts_passes_settings_to_each_cohort(monkeypatch):
    seen = []
    monkeypatch.setattr(cohort_analysis, "run_cohort", lambda i, **k: seen.append((i, k)))
    run_cohorts([1, 2], results_root=Path("r"), reversal_trial_window=50)
    assert seen == [
        (1, {"results_root": Path("r"), "reversal_trial_window": 50}),
        (2, {"results_root": Path("r"), "reversal_trial_window": 50}),
    ]
