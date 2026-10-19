"""The port-finding sanity check: a problem cannot be finished in fewer than n_ports x threshold."""

import json
import math

from src.behavior_analysis.port_finding_check import (
    port_finding_check_table,
    session_port_finding_settings,
)
from tests.poke_fixtures import Recording, full_trial, session_record

TOWERS = ["A1", "B1", "C1"]


def recording(n_port_finding, *, thresh=4, run_start=True, user_set=None, **run_start_extra):
    """*n_port_finding* port-finding trials, then one task trial, with the task's variables."""
    rec = Recording(towers=TOWERS)
    for _ in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"])
    full_trial(rec, choice="A1")
    if user_set is not None:
        rec._add("variable", "user_set", json.dumps(user_set))
    if run_start:
        settings = {"found_ports": False, "found_t_thresh": thresh, **run_start_extra}
        rec.rows.insert(1, (0.001, "variable", "run_start", json.dumps(settings)))
    return rec.to_frame()


def session(*frames):
    return session_record(frames)


def table(sessions_by_mouse):
    return port_finding_check_table(sessions_by_mouse)


def only_row(sessions_by_mouse):
    result = table(sessions_by_mouse)
    assert len(result) == 1
    return result.iloc[0]


KEY = "ses-1_date-20260101"


def test_settings_are_read_from_the_run_start_and_user_set_rows():
    counts = {"A1": 4, "B1": 3, "C1": 4}
    frame = recording(1, found_port_counts=counts, user_set={"found_port_counts": counts})
    settings = session_port_finding_settings(session(frame))
    assert settings.found_t_thresh == 4
    assert settings.start_counts == counts
    assert settings.user_set_counts


def test_a_session_without_run_start_or_threshold_has_unknown_settings():
    for frame in (recording(1, run_start=False), recording(1, thresh=None)):
        settings = session_port_finding_settings(session(frame))
        assert settings.found_t_thresh is None
        assert settings.start_counts is None
        assert not settings.user_set_counts


def test_user_set_of_other_variables_is_not_a_hand_set_count():
    settings = session_port_finding_settings(
        session(recording(1, user_set={"with_init_rew": False}))
    )
    assert not settings.user_set_counts


def test_a_problem_that_reaches_the_minimum_is_not_flagged():
    row = only_row({"m1": {KEY: session(recording(12))}})
    assert row["n_port_finding_trials"] == 12
    assert row["n_ports"] == 3
    assert row["found_t_thresh"] == 4
    assert row["expected_min_port_finding_trials"] == 12
    assert not row["below_expected_min"]
    assert row["flag_reason"] == ""


def test_the_threshold_of_two_gives_a_minimum_of_six():
    row = only_row({"m1": {KEY: session(recording(5, thresh=2))}})
    assert row["expected_min_port_finding_trials"] == 6
    assert row["below_expected_min"]


def test_below_the_minimum_is_flagged_not_raised():
    row = only_row({"m1": {KEY: session(recording(5))}})
    assert row["below_expected_min"]
    assert "each of 3 ports was chosen 4 times" in row["flag_reason"]


def test_a_hand_set_found_port_counts_is_named_in_the_reason():
    counts = {"A1": 4, "B1": 3, "C1": 4}
    frame = recording(5, found_port_counts=counts, user_set={"found_port_counts": counts})
    row = only_row({"m1": {KEY: session(frame)}})
    assert row["below_expected_min"]
    assert "user_set" in row["flag_reason"]
    assert KEY in row["flag_reason"]


def test_counts_carried_in_from_a_missing_session_are_named_in_the_reason():
    carried = {"A1": 5, "B1": 29, "C1": 1}
    row = only_row({"m1": {KEY: session(recording(5, found_port_counts=carried))}})
    assert row["below_expected_min"]
    assert "earlier session" in row["flag_reason"]
    assert "not in the data" in row["flag_reason"]


def test_zero_counts_at_the_start_do_not_look_carried_in():
    zeros = dict.fromkeys(TOWERS, 0)
    row = only_row({"m1": {KEY: session(recording(5, found_port_counts=zeros))}})
    assert "earlier session" not in row["flag_reason"]


def test_trials_are_summed_over_all_sessions_of_the_problem():
    sessions = {
        "ses-1_date-20260101": session(recording(7)),
        "ses-2_date-20260102": session(recording(6)),
        "ses-3_date-20260103": session(recording(0)),
    }
    row = only_row({"m1": sessions})
    assert row["n_port_finding_trials"] == 13
    assert not row["below_expected_min"]


def test_the_settings_come_from_the_first_session_that_has_port_finding():
    sessions = {
        "ses-1_date-20260101": session(recording(0, thresh=2)),
        "ses-2_date-20260102": session(recording(6, thresh=2)),
        "ses-3_date-20260103": session(recording(1, thresh=4)),
    }
    row = only_row({"m1": sessions})
    assert row["found_t_thresh"] == 2
    assert row["expected_min_port_finding_trials"] == 6
    assert row["n_port_finding_trials"] == 7


def test_unlogged_threshold_gives_an_unknown_minimum_and_no_flag():
    row = only_row({"m1": {KEY: session(recording(1, thresh=None))}})
    assert math.isnan(row["expected_min_port_finding_trials"])
    assert not row["below_expected_min"]
    assert "found_t_thresh" in row["flag_reason"]


def test_a_mouse_without_port_finding_has_no_row_and_mice_are_separate():
    result = table(
        {
            "m1": {KEY: session(recording(0))},
            "m2": {KEY: session(recording(12))},
            "m3": {KEY: session(recording(3))},
        }
    )
    assert result["mouse"].tolist() == ["m2", "m3"]
    assert result["below_expected_min"].tolist() == [False, True]


def test_the_table_has_stable_columns_even_when_empty():
    result = port_finding_check_table({})
    assert result.empty
    assert list(result.columns) == [
        "mouse",
        "n_port_finding_trials",
        "n_port_finding_only_sessions",
        "n_ports",
        "found_t_thresh",
        "expected_min_port_finding_trials",
        "below_expected_min",
        "flag_reason",
    ]


def port_finding_only_record(n_port_finding):
    """A session record without any task trial, as the importer keeps port-finding-only sessions."""
    rec = Recording(towers=TOWERS)
    for _ in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"])
    rec.rows.insert(1, (0.001, "variable", "run_start", json.dumps({"found_t_thresh": 4})))
    return {"df": rec.to_frame(), "choice_towers": frozenset(TOWERS), "initiation_tower": "B2"}


def test_port_finding_only_sessions_are_counted_in_their_own_column():
    sessions = {"ses-0_date-20251231": port_finding_only_record(9), KEY: session(recording(5))}
    row = only_row({"m1": sessions})
    assert row["n_port_finding_trials"] == 14
    assert row["n_port_finding_only_sessions"] == 1
    assert not row["below_expected_min"]
