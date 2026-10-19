"""The port-finding poke maps count the trials of port-finding-only sessions.

A mouse that does not finish port finding in one session starts the next session in port finding
again. The first of those sessions has no bandit trials, so it is not in the problem, but its
port-finding trials belong to the problem's port-finding maps.
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.behavior_analysis.port_finding_sessions import (
    assign_port_finding_sessions,
    port_finding_sessions_for_problem,
)
from src.behavior_import.extract_trials import extract_session_trials
from src.behavior_import.import_log import ImportLog
from src.behavior_visualization import plot_maze_heatmaps
from src.pipelines.poke_analysis import run_poke_analysis
from tests.poke_fixtures import Recording, full_trial, session_record

TASK_SESSION = "ses-2_date-20260102"
FIRST_SESSION = "ses-1_date-20260101"
EARLIER_SESSION = "ses-0_date-20251231"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def captured(monkeypatch):
    """Figures handed to ``save_or_show`` by the maze module, by order of drawing."""
    figures = []
    monkeypatch.setattr(plot_maze_heatmaps, "save_or_show", lambda fig, path: figures.append(fig))
    return figures


def with_run_start(rec, **settings):
    settings = {"found_ports": False, "found_t_thresh": 4, **settings}
    rec.rows.insert(1, (0.001, "variable", "run_start", json.dumps(settings)))
    return rec


def task_session(n_port_finding=0, choices=("A1", "B1")):
    """A bandit session that starts with *n_port_finding* port-finding trials."""
    rec = Recording()
    for _ in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"])
    for choice in choices:
        full_trial(rec, choice=choice)
    session = session_record([with_run_start(rec).to_frame()], problem=1)
    extract_session_trials(session, "m1", TASK_SESSION)
    return session


def port_finding_only(n_port_finding, **run_start):
    """A session record as the import log keeps it: raw recordings, no trial variables."""
    rec = Recording()
    for _ in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"])
    frame = with_run_start(rec, **run_start).to_frame()
    return {"df": frame, "choice_towers": frozenset(["A1", "B1", "C1"]), "initiation_tower": "B2"}


def problem(n_port_finding=2):
    return {"m1": {TASK_SESSION: task_session(n_port_finding)}}


def run(tmp_path, data, found=None):
    return run_poke_analysis(data, tmp_path, port_finding_sessions=found)


def titles(figure):
    return [ax.get_title() for ax in figure.axes if ax.get_title()]


def test_the_trials_of_a_port_finding_only_session_are_in_the_port_finding_counts(tmp_path):
    found = {"m1": {FIRST_SESSION: port_finding_only(3)}}
    result = run(tmp_path, problem(2), found)
    counts = result["port_finding_counts"]["after_initiation"]["m1"]
    # five trials in all (2 in the task session, 3 in the other), each a long C3 poke then A1
    assert {port: n for port, n in counts.items() if n} == {"C3": 5, "A1": 5}


def test_the_panel_titles_count_the_trials_of_both_kinds_of_session(tmp_path, captured):
    run(tmp_path, problem(2), {"m1": {FIRST_SESSION: port_finding_only(3)}})
    after_choice, after_initiation = captured[2:]
    # the other session ends after its last trial, whose after-choice window never closes
    assert titles(after_choice) == ["m1 (n=4 port-finding trials)"]
    assert titles(after_initiation) == ["m1 (n=5 port-finding trials)"]


def test_a_mouse_whose_port_finding_is_only_in_the_other_session_still_gets_a_panel(
    tmp_path, captured
):
    result = run(tmp_path, problem(0), {"m1": {FIRST_SESSION: port_finding_only(4)}})
    assert titles(captured[2]) == ["m1 (n=3 port-finding trials)"]
    assert titles(captured[3]) == ["m1 (n=4 port-finding trials)"]
    assert any("Port Finding" in p.name for p in result["files"])


def test_two_port_finding_only_sessions_in_a_row_are_both_counted(tmp_path, captured):
    found = {"m1": {EARLIER_SESSION: port_finding_only(6), FIRST_SESSION: port_finding_only(3)}}
    run(tmp_path, problem(2), found)
    assert titles(captured[2]) == ["m1 (n=9 port-finding trials)"]  # two open windows
    assert titles(captured[3]) == ["m1 (n=11 port-finding trials)"]


def test_the_task_maps_and_long_pokes_do_not_change(tmp_path):
    plain = run(tmp_path / "plain", problem(2))
    extra = run(tmp_path / "extra", problem(2), {"m1": {FIRST_SESSION: port_finding_only(3)}})
    assert extra["counts"] == plain["counts"]
    assert extra["long_pokes"]["by_session"] == plain["long_pokes"]["by_session"]
    assert extra["long_pokes"]["relative"] == plain["long_pokes"]["relative"]
    task_columns = ["n_trials", "n_trials_counted", "n_open_after_choice_windows"]
    task_row = extra["summary"].set_index("session").loc[[TASK_SESSION], task_columns]
    assert (task_row.to_numpy() == plain["summary"][task_columns].to_numpy()).all()


def test_the_task_titles_do_not_change(tmp_path, captured):
    run(tmp_path, problem(2), {"m1": {FIRST_SESSION: port_finding_only(3)}})
    after_choice, after_initiation = captured[:2]
    assert titles(after_choice) == ["m1 (n=1 trials)"]  # the last window never closes
    assert titles(after_initiation) == ["m1 (n=2 trials)"]


def test_the_summary_has_a_marked_row_per_port_finding_only_session(tmp_path):
    found = {"m1": {EARLIER_SESSION: port_finding_only(6), FIRST_SESSION: port_finding_only(3)}}
    summary = run(tmp_path, problem(2), found)["summary"]
    assert summary["session"].tolist() == [EARLIER_SESSION, FIRST_SESSION, TASK_SESSION]
    assert summary["port_finding_only"].tolist() == [True, True, False]
    row = summary.set_index("session").loc[FIRST_SESSION]
    assert (row["n_trials"], row["n_port_finding_trials"], row["n_port_finding_counted"]) == (
        0,
        3,
        3,
    )
    assert summary["n_port_finding_trials"].sum() == 11  # every counted trial is traceable


def test_without_port_finding_only_sessions_the_column_is_false_everywhere(tmp_path):
    summary = run(tmp_path, problem(2))["summary"]
    assert not summary["port_finding_only"].any()


def test_passing_none_or_an_empty_dict_is_the_same(tmp_path):
    none = run(tmp_path / "a", problem(2), None)
    empty = run(tmp_path / "b", problem(2), {})
    assert none["port_finding_counts"] == empty["port_finding_counts"]
    assert none["summary"].equals(empty["summary"])
    assert none["port_finding_check"].equals(empty["port_finding_check"])


def test_the_check_table_counts_the_other_session_and_flags_a_short_problem(tmp_path):
    result = run(tmp_path, problem(2), {"m1": {FIRST_SESSION: port_finding_only(3)}})
    (row,) = result["port_finding_check"].to_dict("records")
    assert row["n_port_finding_trials"] == 5
    assert row["n_port_finding_only_sessions"] == 1
    assert row["expected_min_port_finding_trials"] == 12
    assert row["below_expected_min"]


def test_a_flagged_problem_warns_and_does_not_fail(tmp_path):
    with pytest.warns(UserWarning, match=r"m1.*5 port-finding trials.*12"):
        run(tmp_path, problem(2), {"m1": {FIRST_SESSION: port_finding_only(3)}})


def test_a_problem_that_reaches_the_minimum_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = run(tmp_path, problem(5), {"m1": {FIRST_SESSION: port_finding_only(7)}})
    assert not result["port_finding_check"]["below_expected_min"].any()


def test_a_hand_set_count_is_named_when_the_problem_is_short(tmp_path):
    counts = {"A1": 4, "B1": 3, "C1": 4}
    rec = Recording()
    for _ in range(5):
        rec.port_finding_trial("A1", long_before=["C3"])
    full_trial(rec, choice="A1")
    rec._add("variable", "user_set", json.dumps({"found_port_counts": counts}))
    session = session_record([with_run_start(rec, found_port_counts=counts).to_frame()], problem=1)
    extract_session_trials(session, "m1", TASK_SESSION)
    with pytest.warns(UserWarning, match="user_set"):
        result = run(tmp_path, {"m1": {TASK_SESSION: session}})
    assert "user_set" in result["port_finding_check"].iloc[0]["flag_reason"]


def test_the_check_has_no_row_for_a_mouse_without_port_finding(tmp_path):
    assert run(tmp_path, problem(0))["port_finding_check"].empty


# ---------------------------------------------- inputs that cannot be right


def test_a_mouse_that_is_not_in_the_problem_raises(tmp_path):
    with pytest.raises(ValueError, match="m9"):
        run(tmp_path, problem(2), {"m9": {FIRST_SESSION: port_finding_only(3)}})


def test_a_session_key_that_is_already_a_task_session_raises(tmp_path):
    with pytest.raises(ValueError, match=TASK_SESSION):
        run(tmp_path, problem(2), {"m1": {TASK_SESSION: port_finding_only(3)}})


def test_a_record_without_raw_recordings_raises(tmp_path):
    with pytest.raises(ValueError, match="df"):
        run(tmp_path, problem(2), {"m1": {FIRST_SESSION: {"choice_towers": frozenset("A1")}}})


# ---------------------------------------------- assignment feeds the pipeline


def test_the_assignment_of_a_log_feeds_the_poke_analysis(tmp_path, captured):
    log = ImportLog(port_finding_sessions={"m1": {FIRST_SESSION: port_finding_only(3)}})
    data = problem(2)
    assignment = assign_port_finding_sessions({1: data}, log.port_finding_sessions)
    found = port_finding_sessions_for_problem(assignment, data)
    run(tmp_path, data, found)
    assert titles(captured[3]) == ["m1 (n=5 port-finding trials)"]
