"""run_cohort calls the value, reversal-aligned and poke analyses for the right problems.

The cohort has a blocked problem followed by two drifting ones (``tests/cohort_fixtures.py``).
The poke analysis and the single-session figures run for real on the synthetic recordings; the
three drifting-only entry points are replaced by recorders that write a marker file (they have
their own tests), so these tests are about which folders get which data.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import src.behavior_analysis.poke_counts as poke_counts_module
import src.pipelines.cohort_analysis as cohort_analysis
import src.pipelines.open_field as pipeline
from src.behavior_analysis.poke_events import PokeAlignmentError, PokeParseError
from src.behavior_import.task_types import BLOCKED, DRIFTING
from src.behavior_visualization import plot_maze_heatmaps
from src.pipelines.cohort_analysis import (
    BLOCKED_SELECTION,
    DRIFTING_SELECTION,
    run_cohort,
    selection_for_task_type,
)
from src.pipelines.open_field import AnalysisSelection, run_drifting_value_analyses
from tests.cohort_fixtures import MICE, PROBLEMS, blocked_then_two_drifting
from tests.plot_capture import capture_figures
from tests.plot_fixtures import make_mouse

ENTRY_POINTS = (
    "run_value_psychometric_analysis",
    "run_reversal_aligned_arm_analysis",
    "run_value_change_analysis",
)
FOLDERS = ("value-psychometrics", "reversal-aligned-arms", "value-change")
LONG_POKE_FILES = (
    "Long Pokes by Relative Session.png",
    "Maze Heatmaps - Pokes After Choice.png",
    "Maze Heatmaps - Pokes After Initiation.png",
    "Poke Trial Summary.csv",
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def calls(monkeypatch):
    """Replace the drifting-only entry points by recorders of ``(name, out_dir, data)``."""
    seen = []

    def recorder(name):
        def record(data, out_dir, **kwargs):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "stub.txt").write_text(name)
            seen.append((name, Path(out_dir), data))

        return record

    for name in ENTRY_POINTS:
        monkeypatch.setattr(pipeline, name, recorder(name))
    # The real builder needs real block structure. This one gives each mouse a single "window",
    # the overall problem number of its sessions, to show which problems a pool was built from.
    monkeypatch.setattr(
        pipeline,
        "build_windows_with_magnitudes",
        lambda trials, *, pre, post: {
            m: [next(iter(sessions.values()))["problem"]] for m, sessions in trials.items()
        },
    )
    return seen


@pytest.fixture
def cohort_extras(monkeypatch):
    """Stub the cohort-level pieces that are not under test (they need real reversal data)."""
    monkeypatch.setattr(cohort_analysis, "plot_pooled_drifting_rank", lambda *a, **k: None)
    monkeypatch.setattr(cohort_analysis, "pooled_drifting_rank_results", lambda *a, **k: {})
    monkeypatch.setattr(
        cohort_analysis, "suspect_table_with_task_types", lambda g, m: pd.DataFrame()
    )
    monkeypatch.setattr(cohort_analysis, "write_suspect_reversal_table", lambda *a, **k: None)
    monkeypatch.setattr(cohort_analysis, "plot_trial_block_summaries", lambda *a, **k: [])
    monkeypatch.setattr(cohort_analysis, "subject_numbers_from_sessions", lambda *a, **k: {})
    monkeypatch.setattr(cohort_analysis, "build_trial_block_tables", lambda *a, **k: (None, None))
    for name in (
        "run_choice_probability_analysis",
        "run_first_leave_analysis",
        "run_number_of_reversals_analysis",
        "run_rank_proportion_analysis",
    ):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: {})


@pytest.fixture
def single_sessions(monkeypatch):
    """Record the single-session calls instead of drawing: ``(out_dir, kwargs)``."""
    seen = []
    monkeypatch.setattr(
        pipeline,
        "run_single_session_analysis",
        lambda subjects_trials, out_dir, **kwargs: seen.append((Path(out_dir), kwargs)),
    )
    return seen


@pytest.fixture
def cohort_dir(tmp_path):
    return tmp_path / "figures" / "open-field" / "cohort-01"


def run(tmp_path, **settings):
    return run_cohort(
        1, results_root=tmp_path, trials_by_problem=blocked_then_two_drifting(), **settings
    )


# --- selections ------------------------------------------------------------------------


def test_a_bare_selection_runs_none_of_the_new_analyses():
    selection = AnalysisSelection()
    assert not selection.run_poke_analysis
    assert not any(getattr(selection, name) for name in ENTRY_POINTS)


def test_every_problem_gets_the_poke_analysis_and_only_drifting_gets_the_value_analyses():
    assert BLOCKED_SELECTION.run_poke_analysis and DRIFTING_SELECTION.run_poke_analysis
    assert not any(getattr(BLOCKED_SELECTION, name) for name in ENTRY_POINTS)
    assert all(getattr(DRIFTING_SELECTION, name) for name in ENTRY_POINTS)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_a_drifting_only_analysis_on_a_blocked_problem_raises(name, tmp_path, calls, cohort_extras):
    bad = BLOCKED_SELECTION.model_copy(update={name: True})
    with pytest.raises(ValueError, match="blocked"):
        selection_for_task_type(BLOCKED, blocked=bad)
    with pytest.raises(ValueError, match=name):
        pipeline.run_problem(
            1, {}, task="t", cohort="c", output_root=tmp_path, selection=bad, task_type=BLOCKED
        )
    with pytest.raises(ValueError, match="blocked"):
        run(tmp_path, blocked_selection=bad)
    assert not list(tmp_path.iterdir()), "nothing is written before the selections are checked"


# --- folders ---------------------------------------------------------------------------


def test_drifting_problems_get_the_value_folders_and_blocked_problems_do_not(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    for k in (1, 2):
        problem = cohort_dir / DRIFTING / f"problem-{k:02d}"
        for folder in FOLDERS:
            assert (problem / folder / "stub.txt").exists(), (k, folder)
    blocked = cohort_dir / BLOCKED / "problem-01"
    assert not any((blocked / folder).exists() for folder in FOLDERS)
    pooled = cohort_dir / DRIFTING / "all-drifting-problems"
    for folder in FOLDERS:
        assert (pooled / folder / "stub.txt").exists(), folder
    assert not (cohort_dir / BLOCKED / "all-blocked-problems").exists()


def test_long_pokes_are_written_for_every_problem_and_not_for_the_pool(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    for kind, k in ((BLOCKED, 1), (DRIFTING, 1), (DRIFTING, 2)):
        folder = cohort_dir / kind / f"problem-{k:02d}" / "long-pokes"
        for name in LONG_POKE_FILES:
            assert (folder / name).exists(), (kind, k, name)
    assert not (cohort_dir / DRIFTING / "all-drifting-problems" / "long-pokes").exists()


def test_the_value_analyses_run_once_per_drifting_problem_and_once_pooled(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    drifting = cohort_dir / DRIFTING
    keys_by_dir = {out_dir: sorted(data) for _, out_dir, data in calls}
    for folder in (FOLDERS[0], FOLDERS[2]):
        assert keys_by_dir[drifting / "problem-01" / folder] == [1]
        assert keys_by_dir[drifting / "problem-02" / folder] == [2]
        assert keys_by_dir[drifting / "all-drifting-problems" / folder] == [1, 2]
    assert len(calls) == 3 * 3  # three analyses x (two problems + the pool)


def test_the_arm_windows_of_each_folder_come_from_its_own_problems(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    windows = {
        out_dir: data["pooled"]["m1"] for name, out_dir, data in calls if name == ENTRY_POINTS[1]
    }
    drifting = cohort_dir / DRIFTING
    # windows hold the overall problem number of the sessions they were built from
    assert windows[drifting / "problem-01" / "reversal-aligned-arms"] == [2]
    assert windows[drifting / "problem-02" / "reversal-aligned-arms"] == [3]
    assert windows[drifting / "all-drifting-problems" / "reversal-aligned-arms"] == [2, 3]


def test_problems_are_numbered_within_the_task_type_and_stay_separate_in_the_pool(
    tmp_path, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    pooled = {
        name: data
        for name, out_dir, data in calls
        if out_dir.parent.name == "all-drifting-problems" and name != ENTRY_POINTS[1]
    }
    for data in pooled.values():
        assert sorted(data) == [1, 2]
        # within-type problem 1 holds the sessions of overall problem 2, and so on
        assert {ses["problem"] for ses in data[1]["m1"].values()} == {2}
        assert {ses["problem"] for ses in data[2]["m1"].values()} == {3}


# --- reversal-aligned arms: one pooled entry, windows per problem ----------------------------


def arm_data():
    return {
        4: {"m": make_mouse(1, pick=0), "n": make_mouse(1, pick=1)},
        5: {"m": make_mouse(1, pick=0, first_day=11)},
    }


def test_reversal_aligned_arms_are_pooled_without_per_problem_duplicates(tmp_path, monkeypatch):
    capture = capture_figures(monkeypatch)
    selection = AnalysisSelection(run_reversal_aligned_arm_analysis=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty best_to_third selections
        run_drifting_value_analyses(arm_data(), tmp_path, selection)
    names = capture.saved_names()
    assert names and all("_pooled_" in name for name in names), names
    assert (tmp_path / "reversal-aligned-arms" / "reversal_aligned_arms_counts.csv").exists()


def test_pooled_arm_windows_are_the_per_problem_windows_concatenated_per_mouse(
    tmp_path, monkeypatch
):
    capture_figures(monkeypatch)
    seen = []
    real = pipeline.run_reversal_aligned_arm_analysis
    monkeypatch.setattr(
        pipeline,
        "run_reversal_aligned_arm_analysis",
        lambda windows, out_dir, **kw: seen.append(windows) or real(windows, out_dir, **kw),
    )
    selection = AnalysisSelection(run_reversal_aligned_arm_analysis=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_drifting_value_analyses(arm_data(), tmp_path, selection)
    ((pooled_name, pooled),) = [(k, v) for entry in seen for k, v in entry.items()]
    assert pooled_name == "pooled"
    one = pipeline.build_windows_with_magnitudes(arm_data()[4], pre=10, post=30)
    two = pipeline.build_windows_with_magnitudes(arm_data()[5], pre=10, post=30)
    assert len(pooled["m"]) == len(one["m"]) + len(two["m"])
    assert len(pooled["n"]) == len(one["n"])


# --- red outlines ------------------------------------------------------------------------


@pytest.fixture
def maze_figures(monkeypatch):
    """Figures of the problem-level maze heatmaps, in drawing order."""
    figures = []
    monkeypatch.setattr(plot_maze_heatmaps, "save_or_show", lambda fig, path: figures.append(fig))
    return figures


def ring_ports(figure, which):
    gids = [p.get_gid() for ax in figure.axes for p in ax.patches if p.get_gid()]
    return sorted({g.split(":")[1] for g in gids if g.endswith(f":{which}")})


def test_red_outlines_come_from_the_problem_before_even_when_it_is_another_task_type(
    tmp_path, calls, cohort_extras, single_sessions, maze_figures
):
    run(tmp_path)
    # order of drawing: blocked 1, drifting 1, drifting 2; two figures (after choice, after
    # initiation) each
    blocked, drifting_1, drifting_2 = (maze_figures[i : i + 2] for i in (0, 2, 4))
    assert len(maze_figures) == 6
    assert all(ring_ports(f, "previous") == [] for f in blocked)
    # drifting problem 1 is overall problem 2: its previous problem is the blocked one
    assert all(ring_ports(f, "previous") == ["A1", "B1", "C1"] for f in drifting_1)
    assert all(ring_ports(f, "previous") == ["A3", "B3", "C3"] for f in drifting_2)
    assert all(ring_ports(f, "current") == ["A3", "B3", "C3"] for f in drifting_1)


def test_single_sessions_get_the_previous_problems_ports_per_mouse(
    tmp_path, cohort_dir, calls, cohort_extras, single_sessions
):
    run(tmp_path)
    by_dir = {out_dir: kwargs["previous_choice_towers"] for out_dir, kwargs in single_sessions}
    assert by_dir[cohort_dir / BLOCKED / "problem-01" / "single-sessions"] == dict.fromkeys(MICE)
    expected = dict.fromkeys(MICE, frozenset(PROBLEMS[1][1]))
    assert by_dir[cohort_dir / DRIFTING / "problem-01" / "single-sessions"] == expected
    expected = dict.fromkeys(MICE, frozenset(PROBLEMS[2][1]))
    assert by_dir[cohort_dir / DRIFTING / "problem-02" / "single-sessions"] == expected


def test_single_session_plots_receive_the_mouses_ports_and_the_parsed_counts(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr(pipeline, "plot_single_session", lambda session, **kw: drawn.append(kw))
    sessions = {"m1": {"ses-1": {"trial_info": [1]}, "ses-2": {"trial_info": []}}, "m2": {}}
    previous = {"m1": frozenset({"A1"}), "m2": None}
    parsed = {"m1": {"ses-1": "counts-1", "ses-2": "counts-2"}, "m2": {}}
    pipeline.run_single_session_analysis(
        sessions, tmp_path, previous_choice_towers=previous, poke_counts=parsed
    )
    (kwargs,) = drawn  # the session without trials is skipped
    assert kwargs["previous_choice_towers"] == frozenset({"A1"})
    assert kwargs["poke_counts"] == "counts-1"


def test_single_session_defaults_draw_no_outlines_and_parse_by_themselves(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr(pipeline, "plot_single_session", lambda session, **kw: drawn.append(kw))
    pipeline.run_single_session_analysis({"m1": {"ses-1": {"trial_info": [1]}}}, tmp_path)
    assert drawn[0]["previous_choice_towers"] is None and drawn[0]["poke_counts"] is None


# --- parsing once, errors ----------------------------------------------------------------


@pytest.fixture
def parse_calls(monkeypatch):
    seen = []
    real = poke_counts_module.parse_session

    def counting(session, *args, **kwargs):
        seen.append(kwargs.get("label"))
        return real(session, *args, **kwargs)

    monkeypatch.setattr(poke_counts_module, "parse_session", counting)
    return seen


def test_every_session_is_parsed_once_for_both_the_poke_analysis_and_the_single_session_figure(
    tmp_path, calls, cohort_extras, parse_calls
):
    run(tmp_path)
    assert len(parse_calls) == len(PROBLEMS) * len(MICE) * 2
    assert len(set(parse_calls)) == len(parse_calls)


def test_the_parsed_counts_given_to_the_single_sessions_are_the_ones_the_poke_analysis_used(
    tmp_path, calls, cohort_extras, single_sessions, monkeypatch
):
    used = []
    real = pipeline.run_poke_analysis
    monkeypatch.setattr(
        pipeline,
        "run_poke_analysis",
        lambda *a, **kw: used.append(kw["poke_counts"]) or real(*a, **kw),
    )
    run(tmp_path)
    assert len(used) == len(single_sessions) == 3
    for given_to_poke, (_, kwargs) in zip(used, single_sessions, strict=True):
        assert kwargs["poke_counts"] == given_to_poke
        for mouse, by_session in given_to_poke.items():
            assert all(kwargs["poke_counts"][mouse][k] is v for k, v in by_session.items())


def test_a_session_with_unreadable_events_warns_and_the_run_goes_on(
    tmp_path, cohort_dir, calls, cohort_extras, monkeypatch
):
    real = poke_counts_module.parse_session

    def failing(session, initiation, *, label):
        if label == "m1 ses-3_date-20260103":
            raise PokeParseError("impossible log")
        return real(session, initiation, label=label)

    monkeypatch.setattr(poke_counts_module, "parse_session", failing)
    with pytest.warns(UserWarning, match="impossible log"):
        run(tmp_path)
    table = pd.read_csv(
        cohort_dir / DRIFTING / "problem-01" / "long-pokes" / "Poke Trial Summary.csv"
    )
    assert table.loc[table["session"] == "ses-3_date-20260103", "parse_error"].notna().any()
    assert (cohort_dir / DRIFTING / "problem-02" / "long-pokes" / "Poke Trial Summary.csv").exists()


def test_misaligned_trials_stop_the_run(tmp_path, calls, cohort_extras, monkeypatch):
    def misaligned(session, initiation, *, label):
        raise PokeAlignmentError(f"{label}: parsed trials differ from the imported ones")

    monkeypatch.setattr(poke_counts_module, "parse_session", misaligned)
    with pytest.raises(PokeAlignmentError):
        run(tmp_path)


def test_arm_figure_titles_name_the_problem_or_the_pool_never_pooled(tmp_path, monkeypatch):
    capture = capture_figures(monkeypatch)
    selection = AnalysisSelection(run_reversal_aligned_arm_analysis=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_drifting_value_analyses({4: arm_data()[4]}, tmp_path / "one", selection)
        one = [s.figure.axes[0].get_title() for s in capture.saved]
        capture.saved.clear()
        run_drifting_value_analyses(arm_data(), tmp_path / "pool", selection)
        pool = [s.figure.axes[0].get_title() for s in capture.saved]
    assert one and all("[problem-04]" in t and "pooled" not in t for t in one)
    assert pool and all("[all drifting problems]" in t for t in pool)
