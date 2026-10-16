"""run_poke_analysis on tiny synthetic problems."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.poke_events import PokeAlignmentError
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


def test_writes_the_figures_and_no_tables(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path / "long-pokes")
    names = {p.name for p in (tmp_path / "long-pokes").iterdir()}
    assert names == {
        f"{stem}.{ext}"
        for stem in (
            "Long Pokes by Relative Session",
            "Maze Heatmaps - Pokes After Choice",
            "Maze Heatmaps - Pokes After Initiation",
        )
        for ext in ("png", "pdf")
    }
    assert set(result) >= {"long_pokes", "counts", "per_session", "cross_check", "summary", "files"}


def test_problem_counts_are_sums_over_sessions_and_recordings(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path)
    after_initiation = result["counts"]["after_initiation"]["m1"]
    # m1: session 1 chooses A1, B1, B1; session 2 adds a long A3 poke before its first choice.
    assert {p: n for p, n in after_initiation.items() if n} == {"A1": 2, "B1": 4, "A3": 1}
    # m2's one session has two recordings of 3 trials each: both are counted.
    assert sum(result["counts"]["after_initiation"]["m2"].values()) == 6


def test_after_choice_counts_leave_out_trials_whose_window_never_closed(tmp_path):
    result = run_poke_analysis(problem_data(), tmp_path)
    # Per recording the first trial has pokes (C3, A2), every closed window ends with the
    # initiating B2 poke (two closed windows per session), and the last window is open.
    assert {p: n for p, n in result["counts"]["after_choice"]["m1"].items() if n} == {
        "C3": 2,
        "A2": 2,
        "B2": 4,
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


def test_both_poke_plots_mark_the_initiation_port_b2_in_blue(tmp_path, captured):
    run_poke_analysis(problem_data(), tmp_path)
    after_choice_fig, after_initiation_fig = captured

    def gids(fig):
        return {p.get_gid() for ax in fig.axes for p in ax.patches if p.get_gid()}

    assert "ring:B2:initiation" in gids(after_initiation_fig)
    assert "ring:B2:initiation" in gids(after_choice_fig)


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


def test_each_mouse_gets_its_own_initiation_outline(tmp_path, captured):
    data = problem_data()
    rec = Recording(init="B3")
    for choice in ("A1", "B1", "B1"):
        full_trial(rec, choice=choice)
    session = session_record([rec.to_frame()], problem=1, initiation_tower="B3")
    extract_session_trials(session, "m2", "ses-7_date-20260107")
    data["m2"] = {"ses-7_date-20260107": session}
    run_poke_analysis(data, tmp_path)
    after_initiation_fig = captured[1]
    gids = [p.get_gid() for ax in after_initiation_fig.axes for p in ax.patches if p.get_gid()]
    assert sorted(g for g in gids if g.endswith(":initiation")) == [
        "ring:B2:initiation",
        "ring:B3:initiation",
    ]


def test_mouse_with_two_initiation_towers_in_one_problem_raises(tmp_path):
    data = problem_data()
    data["m1"]["ses-1_date-20260101"]["initiation_tower"] = "B3"
    with pytest.raises(ValueError, match="m1.*several initiation towers"):
        run_poke_analysis(data, tmp_path)


def test_files_lists_exactly_what_this_call_wrote(tmp_path):
    (tmp_path / "unrelated.csv").write_text("a\n1\n")
    result = run_poke_analysis(problem_data(), tmp_path)
    assert tmp_path / "unrelated.csv" not in result["files"]
    assert all(p.exists() for p in result["files"])
    assert len(result["files"]) == 6


def test_trial_summary_reports_dropped_trials(tmp_path):
    rec = Recording()
    full_trial(rec, choice="A1")
    rec.initiate()
    rec.choose("B1", held="B3")  # held port differs from the printed choice
    rec.iti().state("wait_init")
    rec.initiate()  # aborted: ITI with no print
    rec.iti().state("wait_init")
    full_trial(rec, choice="C1")
    data = {"m1": {"ses-1_date-20260101": imported_session([rec.to_frame()], 1, 1)}}
    with pytest.warns(UserWarning, match="dropped 2 trials") as caught:
        result = run_poke_analysis(data, tmp_path)
    assert len([w for w in caught if "dropped" in str(w.message)]) == 1
    summary = result["summary"]
    assert list(summary.columns) == [
        "mouse",
        "session",
        "n_trials",
        "n_port_finding_trials",
        "n_port_finding_counted",
        "n_port_finding_dropped",
        "n_port_finding_open_after_choice_windows",
        "n_trials_counted",
        "n_trials_dropped_no_print",
        "n_trials_dropped_inconsistent",
        "n_open_after_choice_windows",
        "n_long_poke_checked",
        "n_long_poke_agree",
        "parse_error",
    ]
    row = summary.iloc[0]
    assert (row.n_trials, row.n_trials_counted) == (3, 2)
    assert (row.n_trials_dropped_no_print, row.n_trials_dropped_inconsistent) == (1, 1)
    assert row.n_port_finding_trials == 0
    assert result["counts"]["after_initiation"]["m1"]["B3"] == 0


def test_summary_reports_port_finding_trials_apart_from_drops(tmp_path):
    rec = Recording()
    rec.port_finding_trial("A1")
    rec.port_finding_trial("C1")
    full_trial(rec, choice="A1")
    full_trial(rec, choice="B1")
    data = {"m1": {"ses-1_date-20260101": imported_session([rec.to_frame()], 1, 1)}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # port finding is not a drop and must not warn
        result = run_poke_analysis(data, tmp_path)
    (row,) = result["summary"].itertuples()
    assert (row.n_trials, row.n_port_finding_trials, row.n_trials_counted) == (2, 2, 2)
    assert (row.n_trials_dropped_no_print, row.n_trials_dropped_inconsistent) == (0, 0)
    assert result["per_session"]["m1"]["ses-1_date-20260101"].n_port_finding_trials == 2


# ---------------------------------------------- port-finding maps

PORT_FINDING_STEMS = (
    "Maze Heatmaps - Pokes After Choice (Port Finding)",
    "Maze Heatmaps - Pokes After Initiation (Port Finding)",
)


def port_finding_recording(n_port_finding=2, choices=("A1", "B1")):
    """Port finding first (as in the raw files), then task trials."""
    rec = Recording()
    for i in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"], after_choice=["C2"] if i == 0 else ())
    for choice in choices:
        full_trial(rec, choice=choice)
    return rec.to_frame()


def port_finding_problem():
    """m1 has port finding in two of its three sessions, m2 in none."""
    return {
        "m1": {
            "ses-1_date-20260101": imported_session([port_finding_recording(2)], 1, 1),
            "ses-2_date-20260102": imported_session([recording()], 2, 2),
            "ses-3_date-20260103": imported_session([port_finding_recording(3)], 3, 3),
        },
        "m2": {"ses-7_date-20260107": imported_session([recording()], 7, 7)},
    }


def test_port_finding_maps_are_drawn_only_for_mice_with_port_finding(tmp_path, captured):
    result = run_poke_analysis(port_finding_problem(), tmp_path)
    after_choice, after_initiation = captured[2:]
    assert [ax.get_title() for ax in after_choice.axes if ax.get_title()] == [
        "m1 (n=5 port-finding trials)"
    ]
    assert [ax.get_title() for ax in after_initiation.axes if ax.get_title()] == [
        "m1 (n=5 port-finding trials)"
    ]
    assert {p.name for p in result["files"]} >= {
        f"{stem}.{ext}" for stem in PORT_FINDING_STEMS for ext in ("png", "pdf")
    }


def test_port_finding_maps_are_written_next_to_the_others(tmp_path):
    result = run_poke_analysis(port_finding_problem(), tmp_path)
    assert len(result["files"]) == 10
    assert all(p.exists() for p in result["files"])


def test_port_finding_counts_sum_over_the_sessions_of_the_problem(tmp_path):
    result = run_poke_analysis(port_finding_problem(), tmp_path)
    counts = result["port_finding_counts"]
    assert list(counts["after_initiation"]) == ["m1"] == list(counts["after_choice"])
    # five trials, each a long poke at C3 then the choice A1
    assert {p: n for p, n in counts["after_initiation"]["m1"].items() if n} == {"C3": 5, "A1": 5}
    # the first trial of each session: C2, then every window ends with the initiating B2
    assert {p: n for p, n in counts["after_choice"]["m1"].items() if n} == {"C2": 2, "B2": 5}


def test_task_counts_do_not_change_when_port_finding_is_added(tmp_path):
    def problem(n_port_finding):
        frame = port_finding_recording(n_port_finding, choices=("A1", "B1", "C1"))
        return {"m1": {"ses-1_date-20260101": imported_session([frame])}}

    with_port_finding = run_poke_analysis(problem(4), tmp_path / "a")
    plain = run_poke_analysis(problem(0), tmp_path / "b")
    assert with_port_finding["counts"] == plain["counts"]
    task_columns = ["n_trials", "n_trials_counted", "n_open_after_choice_windows"]
    assert (with_port_finding["summary"][task_columns] == plain["summary"][task_columns]).all(None)


def test_no_port_finding_map_when_no_mouse_has_port_finding(tmp_path, captured):
    result = run_poke_analysis(problem_data(), tmp_path)
    assert len(captured) == 2
    assert result["port_finding_counts"] == {"after_initiation": {}, "after_choice": {}}
    assert not [p for p in tmp_path.iterdir() if "Port Finding" in p.name]
    assert len(result["files"]) == 6


def test_port_finding_maps_carry_the_same_outlines(tmp_path, captured):
    previous = {"m1": frozenset({"A1", "C3"}), "m2": None}
    run_poke_analysis(port_finding_problem(), tmp_path, previous_choice_towers=previous)
    for fig in captured[2:]:
        gids = [p.get_gid() for ax in fig.axes for p in ax.patches if p.get_gid()]
        assert sorted(g for g in gids if g.endswith(":previous")) == [
            "ring:A1:previous",
            "ring:C3:previous",
        ]
        assert "ring:B2:initiation" in gids
        assert sum(g.endswith(":current") for g in gids) == 3


def test_summary_adds_the_port_finding_counts(tmp_path):
    result = run_poke_analysis(port_finding_problem(), tmp_path)
    summary = result["summary"].set_index("session")
    assert summary.loc["ses-1_date-20260101", "n_port_finding_trials"] == 2
    assert summary.loc["ses-3_date-20260103", "n_port_finding_counted"] == 3
    assert summary.loc["ses-2_date-20260102", "n_port_finding_counted"] == 0
    # the last port-finding trial's window is closed by the first task trial's initiation
    assert summary["n_port_finding_open_after_choice_windows"].sum() == 0
    assert summary["n_port_finding_dropped"].sum() == 0


# ---------------------------------------------- two problems end to end: previous-problem outlines

PREVIOUS_RGB = (209, 57, 61)


def problem_sessions(towers, problem, mouse="m1"):
    rec = Recording(towers=towers)
    for i, choice in enumerate(towers):
        full_trial(rec, choice=choice, after_choice=["B2"] if i == 0 else ())
    session = session_record([rec.to_frame()], problem=problem)
    extract_session_trials(session, mouse, "ses-1_date-20260101")
    return {mouse: {f"ses-{problem}_date-2026010{problem}": session}}


def count_pixels(fig, rgb):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return int((image == np.array(rgb)).all(axis=-1).sum())


def test_red_outlines_of_the_previous_problem_appear_in_problem_two_only(tmp_path, captured):
    from src.behavior_analysis.poke_counts import (
        previous_choice_towers_for_problem,
        previous_problem_choice_towers,
    )

    all_problems = {
        2: problem_sessions(["A3", "B3", "C3"], 2),
        1: problem_sessions(["A1", "B1", "C1"], 1),
    }
    previous = previous_problem_choice_towers(all_problems)
    for problem in (1, 2):
        subjects = all_problems[problem]
        run_poke_analysis(
            subjects,
            tmp_path / f"problem-{problem}",
            previous_choice_towers=previous_choice_towers_for_problem(subjects, previous),
        )
    problem_one_fig, problem_two_fig = captured[0], captured[2]  # after-choice figures

    def red_ports(fig):
        return sorted(
            p.get_gid().split(":")[1]
            for ax in fig.axes
            for p in ax.patches
            if (p.get_gid() or "").endswith(":previous")
        )

    assert red_ports(problem_one_fig) == []
    assert red_ports(problem_two_fig) == ["A1", "B1", "C1"]
    assert count_pixels(problem_one_fig, PREVIOUS_RGB) == 0
    assert count_pixels(problem_two_fig, PREVIOUS_RGB) > 0


def test_single_session_of_problem_two_draws_red_outlines_from_the_same_helper(monkeypatch):
    from src.behavior_analysis.poke_counts import (
        previous_choice_towers_for_problem,
        previous_problem_choice_towers,
    )
    from src.behavior_visualization import plot_single_session as single_session_module

    figures = []
    monkeypatch.setattr(
        single_session_module, "save_or_show", lambda fig, path: figures.append(fig)
    )
    all_problems = {
        1: problem_sessions(["A1", "B1", "C1"], 1),
        2: problem_sessions(["A3", "B3", "C3"], 2),
    }
    previous = previous_problem_choice_towers(all_problems)
    prev_ports = previous_choice_towers_for_problem(all_problems[2], previous)["m1"]
    (session,) = all_problems[2]["m1"].values()
    single_session_module.plot_single_session(session, previous_choice_towers=prev_ports)
    single_session_module.plot_single_session(
        session,
        previous_choice_towers=previous_choice_towers_for_problem(all_problems[1], previous)["m1"],
    )
    with_red, without_red = figures
    assert count_pixels(with_red, PREVIOUS_RGB) > 0
    assert count_pixels(without_red, PREVIOUS_RGB) == 0


def unreadable_session():
    rec = Recording()
    full_trial(rec, choice="A1")
    rec.initiate()
    rec.poke_in("A1").state("poked_in")  # poked_in without hold_poke
    return imported_session([rec.to_frame()], 3, 3)


def test_session_with_unreadable_events_is_skipped_with_a_warning_and_reported(tmp_path):
    data = problem_data()
    data["m1"]["ses-3_date-20260103"] = unreadable_session()
    with pytest.warns(UserWarning, match="m1 ses-3_date-20260103: skipped.*poked_in"):
        result = run_poke_analysis(data, tmp_path)
    summary = result["summary"].fillna("")
    bad = summary[(summary.mouse == "m1") & (summary.session == "ses-3_date-20260103")].iloc[0]
    assert "poked_in" in bad.parse_error
    assert (bad.n_trials, bad.n_trials_counted) == (0, 0)
    assert (summary.parse_error == "").sum() == len(summary) - 1
    # The other sessions' counts are untouched.
    assert result["counts"]["after_initiation"]["m1"]["B1"] == 4
    assert sum(result["per_session"]["m1"]["ses-3_date-20260103"].after_choice.values()) == 0


def test_misaligned_session_stops_run_poke_analysis(tmp_path):
    data = problem_data()
    data["m1"]["ses-1_date-20260101"]["choice"] = ["A1", "C1", "B1"]
    with pytest.raises(PokeAlignmentError, match="trial 2"):
        run_poke_analysis(data, tmp_path)


def test_panel_titles_count_the_trials_each_map_uses(tmp_path, captured):
    run_poke_analysis(problem_data(), tmp_path)
    after_choice_fig, after_initiation_fig = captured

    def titles(fig):
        return sorted(ax.get_title() for ax in fig.axes if ax.get_title())

    # m1: 2 sessions x 3 trials, one open after-choice window per session; m2: 2 recordings.
    assert titles(after_initiation_fig) == ["m1 (n=6 trials)", "m2 (n=6 trials)"]
    assert titles(after_choice_fig) == ["m1 (n=4 trials)", "m2 (n=4 trials)"]


def test_pre_parsed_counts_are_reused_instead_of_parsing_again(tmp_path, monkeypatch):
    from src.pipelines import poke_analysis as module

    data = problem_data()
    first = run_poke_analysis(data, tmp_path / "first")

    def must_not_parse(*args, **kwargs):
        raise AssertionError("parsed again")

    monkeypatch.setattr(module, "mouse_poke_counts", must_not_parse)
    second = run_poke_analysis(data, tmp_path / "second", poke_counts=first["per_session"])
    assert second["counts"] == first["counts"]


def test_pre_parsed_counts_must_cover_every_session(tmp_path):
    data = problem_data()
    first = run_poke_analysis(data, tmp_path / "first")
    partial = {m: dict(s) for m, s in first["per_session"].items()}
    del partial["m1"]["ses-1_date-20260101"]
    with pytest.raises(ValueError, match="m1 sessions .*ses-1_date-20260101"):
        run_poke_analysis(data, tmp_path / "second", poke_counts=partial)
