"""The port-finding sanity check: a problem cannot be finished in fewer than n_ports x threshold."""

import json
import math

import pytest

from src.behavior_analysis.port_finding_check import (
    port_finding_check_table,
    session_port_finding_settings,
)
from tests.poke_fixtures import Recording, full_trial, session_record

TOWERS = ["A1", "B1", "C1"]


def recording(n_port_finding, *, thresh=4, run_start=True, user_set=None, **run_start_extra):
    """*n_port_finding* port-finding trials, then one task trial, with the task's variables.

    *user_set* is a dict, or a list of dicts, each logged as a ``user_set`` row in the middle of
    port finding (after the first port-finding trial).
    """
    rec = Recording(towers=TOWERS)
    sets = [user_set] if isinstance(user_set, dict) else list(user_set or [])
    for i in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"])
        if i == 0:
            for changed in sets:
                rec._add("variable", "user_set", json.dumps(changed))
    full_trial(rec, choice="A1")
    rec.rows = [row for row in rec.rows if row[2] != "run_start"]  # the fixture's default
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
    assert settings.hand_set == ("found_port_counts",)


def test_a_session_without_a_logged_threshold_has_unknown_settings():
    settings = session_port_finding_settings(session(recording(1, thresh=None)))
    assert settings.found_t_thresh is None
    assert settings.start_counts is None
    assert settings.hand_set == ()


def test_a_session_without_run_start_raises_naming_the_session():
    with pytest.raises(ValueError, match="m1 ses-1.*run_start"):
        port_finding_check_table({"m1": {KEY: session(recording(1, run_start=False))}})


def test_user_set_of_other_variables_is_not_a_hand_set_count():
    settings = session_port_finding_settings(
        session(recording(1, user_set={"with_init_rew": False}))
    )
    assert settings.hand_set == ()


def test_hand_set_threshold_and_found_ports_are_detected_with_the_latest_threshold():
    changes = [{"found_t_thresh": 2}, {"found_ports": True}, {"found_t_thresh": 3}]
    settings = session_port_finding_settings(session(recording(2, user_set=changes)))
    assert set(settings.hand_set) == {"found_t_thresh", "found_ports"}
    assert settings.hand_set_thresh == 3


def test_a_problem_that_reaches_the_minimum_is_not_flagged():
    row = only_row({"m1": {KEY: session(recording(12))}})
    assert row["n_port_finding_trials"] == 12
    assert row["n_ports"] == 3
    assert row["found_t_thresh"] == 4
    assert row["expected_min_port_finding_trials"] == 12
    assert not row["below_expected_min"]
    assert row["flag_reason"] == ""
    assert not row["flagged"]
    assert row["carried_in_counts"] == ""


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


def test_a_hand_set_found_port_counts_is_flagged_whatever_the_total():
    frame = recording(14, user_set={"found_port_counts": dict.fromkeys(TOWERS, 4)})
    row = only_row({"m1": {KEY: session(frame)}})
    assert not row["below_expected_min"]
    assert row["flagged"]
    assert "found_port_counts were set by hand" in row["flag_reason"]


def test_a_hand_set_threshold_is_named_and_gives_the_minimum():
    row = only_row({"m1": {KEY: session(recording(5, user_set={"found_t_thresh": 1}))}})
    assert row["flagged"] and not row["below_expected_min"]
    assert row["found_t_thresh"] == 1 and row["expected_min_port_finding_trials"] == 3
    assert "found_t_thresh was set by hand" in row["flag_reason"]


def test_a_hand_set_found_ports_is_flagged():
    row = only_row({"m1": {KEY: session(recording(14, user_set={"found_ports": True}))}})
    assert row["flagged"]
    assert "found_ports was set by hand" in row["flag_reason"]


def test_counts_carried_in_from_a_missing_session_are_flagged_and_listed():
    carried = {"A1": 5, "B1": 29, "C1": 1}
    row = only_row({"m1": {KEY: session(recording(5, found_port_counts=carried))}})
    assert row["below_expected_min"] and row["flagged"]
    assert row["carried_in_counts"] == "A1=5;B1=29;C1=1"
    assert "earlier session" in row["flag_reason"]
    assert "missing from the data or not assigned (see the warnings)" in row["flag_reason"]


def test_counts_carried_in_are_flagged_even_when_the_total_reaches_the_minimum():
    # ses-63 missing: ses-64's 13 trials alone reach 12 but the problem began earlier
    carried = {"A1": 5, "B1": 29, "C1": 1}
    row = only_row({"m1": {KEY: session(recording(13, found_port_counts=carried))}})
    assert not row["below_expected_min"]
    assert row["flagged"]
    assert "earlier session" in row["flag_reason"]


def test_carried_in_counts_of_a_later_session_are_not_a_missing_start():
    carried = {"A1": 5, "B1": 29, "C1": 1}
    sessions = {
        "ses-0_date-20251231": port_finding_only_record(35),
        KEY: session(recording(13, found_port_counts=carried)),
    }
    row = only_row({"m1": sessions})
    assert not row["flagged"]
    assert row["carried_in_counts"] == ""


def test_zero_counts_at_the_start_do_not_look_carried_in():
    zeros = dict.fromkeys(TOWERS, 0)
    row = only_row({"m1": {KEY: session(recording(14, found_port_counts=zeros))}})
    assert not row["flagged"] and row["carried_in_counts"] == ""


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
        "carried_in_counts",
        "below_expected_min",
        "flagged",
        "flag_reason",
    ]


def port_finding_only_record(n_port_finding):
    """A session record without any task trial, as the importer keeps port-finding-only sessions."""
    rec = Recording(towers=TOWERS)
    for _ in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["C3"])
    run_start = {"found_ports": False, "found_t_thresh": 4}
    rec.rows.insert(1, (0.001, "variable", "run_start", json.dumps(run_start)))
    return {"df": rec.to_frame(), "choice_towers": frozenset(TOWERS), "initiation_tower": "B2"}


def test_port_finding_only_sessions_are_counted_in_their_own_column():
    sessions = {"ses-0_date-20251231": port_finding_only_record(9), KEY: session(recording(5))}
    row = only_row({"m1": sessions})
    assert row["n_port_finding_trials"] == 14
    assert row["n_port_finding_only_sessions"] == 1
    assert not row["below_expected_min"]
