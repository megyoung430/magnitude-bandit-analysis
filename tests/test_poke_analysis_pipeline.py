"""run_poke_analysis on tiny synthetic problems."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_import.extract_trials import extract_session_trials
from src.behavior_visualization import plot_maze_heatmaps
from src.pipelines.poke_analysis import run_poke_analysis
from tests.poke_fixtures import Recording, full_trial, session_record


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def recording(long_before=(), choices=("A1", "B1", "B1"), after=("C3", "A2")):
    rec = Recording()
    for i, choice in enumerate(choices):
        full_trial(
            rec,
            long_before=long_before if i == 0 else (),
            choice=choice,
            after_choice=after if i == 0 else (),
        )
    return rec.to_frame()


def imported_session(frames, number=1, day=1):
    """A session record as the importer gives it: raw frames plus extracted trial variables."""
    session = session_record(frames, problem=1)
    extract_session_trials(session, "m", f"ses-{number}_date-202601{day:02d}")
    return session


def problem_data():
    return {
        "m1": {
            "ses-2_date-20260102": imported_session([recording(long_before=["A3"])], 2, 2),
            "ses-1_date-20260101": imported_session([recording()], 1, 1),
        },
        "m2": {"ses-7_date-20260107": imported_session([recording(), recording()], 7, 7)},
    }


@pytest.fixture
def captured(monkeypatch):
    """Figures handed to ``save_or_show`` by the maze module, by order of drawing."""
    figures = []
    monkeypatch.setattr(plot_maze_heatmaps, "save_or_show", lambda fig, path: figures.append(fig))
    return figures


def test_writes_all_figures_and_tables(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path / "long-pokes")
    names = {p.name for p in (tmp_path / "long-pokes").iterdir()}
    for stem in (
        "Long Pokes by Relative Session",
        "Maze Heatmaps - Pokes After Choice",
        "Maze Heatmaps - Pokes After Initiation",
    ):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names
    for table in (
        "Long Pokes by Session.csv",
        "Poke Counts by Port.csv",
        "Poke Counts by Session.csv",
        "Long Poke Cross-check.csv",
    ):
        assert table in names
    assert set(result) >= {"long_pokes", "counts", "per_session", "cross_check", "files"}


def test_problem_counts_are_sums_over_sessions_and_recordings(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path)
    after_initiation = result["counts"]["after_initiation"]["m1"]
    # m1: session 1 chooses A1, B1, B1; session 2 adds a long A3 poke before its first choice.
    assert {p: n for p, n in after_initiation.items() if n} == {"A1": 2, "B1": 4, "A3": 1}
    # m2's one session has two recordings of 3 trials each: both are counted.
    assert sum(result["counts"]["after_initiation"]["m2"].values()) == 6


def test_after_choice_counts_leave_out_trials_whose_window_never_closed(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path)
    # Per recording only the first trial has pokes (C3, A2) and the last trial's window is open.
    assert {p: n for p, n in result["counts"]["after_choice"]["m1"].items() if n} == {
        "C3": 2,
        "A2": 2,
    }
    assert sum(c.n_open_windows for c in result["per_session"]["m1"].values()) == 2


def test_long_pokes_are_aligned_by_relative_session(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path)
    relative = result["long_pokes"]["relative"]
    assert list(relative["m1"]) == ["rel_ses-01", "rel_ses-02"]
    assert list(relative["m2"]) == ["rel_ses-01"]
    # m1 session 2 has one non-choice long poke in its first of three trials.
    assert relative["m1"]["rel_ses-02"] == pytest.approx(1 / 3)
    assert relative["m1"]["rel_ses-01"] == 0.0


def test_long_poke_means_come_from_the_merged_multi_recording_session(tmp_path):
    data = problem_data()
    assert len(data["m2"]["ses-7_date-20260107"]["num_long_pokes"]) == 6
    result = run_poke_analysis(data, tmp_path)
    assert result["long_pokes"]["problem_mean_by_mouse"]["m2"] == 0.0


def test_green_and_red_outlines_follow_the_given_ports(tmp_path, captured):
    previous = {"m1": frozenset({"A1", "C3"}), "m2": None}
    run_poke_analysis(problem_data(), tmp_path, previous_choice_towers=previous)
    after_choice_fig = captured[0]
    gids = [p.get_gid() for ax in after_choice_fig.axes for p in ax.patches if p.get_gid()]
    assert sorted(g for g in gids if g.endswith(":previous")) == [
        "ring:A1:previous",
        "ring:C3:previous",
    ]
    # Both mice choose from A1, B1, C1 in this fixture.
    assert sum(g.endswith(":current") for g in gids) == 6


def test_without_previous_ports_no_red_ring_is_drawn(tmp_path, captured):
    run_poke_analysis(problem_data(), tmp_path)
    gids = [p.get_gid() for fig in captured for ax in fig.axes for p in ax.patches if p.get_gid()]
    assert not any(g.endswith(":previous") for g in gids)


def test_initiation_plot_marks_b2_in_blue_and_the_choice_plot_does_not(tmp_path, captured):
    run_poke_analysis(problem_data(), tmp_path)
    after_choice_fig, after_initiation_fig = captured

    def gids(fig):
        return {p.get_gid() for ax in fig.axes for p in ax.patches if p.get_gid()}

    assert "ring:B2:initiation" in gids(after_initiation_fig)
    assert not any(g.endswith(":initiation") for g in gids(after_choice_fig))


def test_previous_ports_must_cover_every_mouse(tmp_path):
    with pytest.raises(ValueError, match="m2"):
        run_poke_analysis(problem_data(), tmp_path, previous_choice_towers={"m1": None})


def test_input_without_raw_recordings_raises(tmp_path):
    data = problem_data()
    del data["m1"]["ses-1_date-20260101"]["df"]
    with pytest.raises(ValueError, match="m1.*ses-1_date-20260101.*df"):
        run_poke_analysis(data, tmp_path)


def test_empty_input_raises(tmp_path):
    with pytest.raises(ValueError, match="no mice"):
        run_poke_analysis({}, tmp_path)


def test_input_keyed_by_problem_instead_of_mouse_raises(tmp_path):
    with pytest.raises(ValueError, match="one problem"):
        run_poke_analysis({1: problem_data()}, tmp_path)


def test_disagreeing_printed_long_pokes_warn_and_are_reported(tmp_path):
    data = problem_data()
    frame = data["m1"]["ses-1_date-20260101"]["df"]
    printed = frame["subtype"] == "print"
    frame.loc[printed, "content"] = frame.loc[printed, "content"].str.replace(
        '"long_pokes": []', '"long_pokes": ["A2"]'
    )
    with pytest.warns(UserWarning, match="printed long_pokes"):
        result = run_poke_analysis(data, tmp_path)
    check = result["cross_check"]["m1"]["ses-1_date-20260101"]
    assert check.n_agree == 0


def test_no_warning_when_raw_and_printed_long_pokes_agree(tmp_path):
    data = problem_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_poke_analysis(data, tmp_path)


def test_csv_tables_hold_the_returned_numbers(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path)
    by_port = pd.read_csv(tmp_path / "Poke Counts by Port.csv")
    row = by_port[(by_port.mouse == "m1") & (by_port.kind == "after_initiation")]
    assert (
        dict(zip(row.port, row["count"], strict=True)) == result["counts"]["after_initiation"]["m1"]
    )
    sessions = pd.read_csv(tmp_path / "Long Pokes by Session.csv")
    assert list(sessions.columns) == [
        "mouse",
        "session",
        "relative_session",
        "n_trials",
        "mean_long_pokes",
    ]
    assert np.isclose(sessions.mean_long_pokes.sum(), 1 / 3)
