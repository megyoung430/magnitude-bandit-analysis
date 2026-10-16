"""Sessions that may have run on incorrect settings: numbering flags and changed settings."""

import pandas as pd
import pytest

from src.behavior_analysis import suspect_sessions as ss
from src.behavior_analysis.suspect_sessions import (
    COLUMNS,
    SETTINGS_KEYS,
    read_snapshots,
    setting_changes,
    suspect_sessions_table,
)
from src.behavior_import.import_log import SessionFlag
from src.behavior_import.problem_numbering import PROVISIONAL_REASON
from tests.review_fixtures import SETTINGS, cohort, recording, session

S1, S2, S3 = "ses-1_date-20260101", "ses-2_date-20260102", "ses-3_date-20260103"


def flag(session_key, flag_name, *, problem=None, provisional=False, reason="because"):
    return SessionFlag(
        "M1",
        session_key,
        (f"/raw/M1/{session_key}/behav/M1-2026-01-03-100000.tsv",),
        flag_name,
        problem,
        provisional,
        reason,
        "Meg\\3x3_field_drifting_value_bandit",
    )


def settings_table(sessions, flags=()):
    _, index, grouped, _ = cohort({1: {"M1": sessions}})
    return suspect_sessions_table(list(flags), grouped, index)


def changes_of(table):
    return [
        (r["comparison"], r["key"], r["value_before"], r["value_after"])
        for r in table.replace({pd.NA: None}).to_dict("records")
    ]


# --- numbering flags ---


@pytest.mark.parametrize(
    "name",
    [
        "suspected_settings_error",
        "review_settings",
        "port_finding",
        "pretraining",
        "no_choice_towers",
    ],
)
def test_each_numbering_flag_gets_a_row_with_its_reason(name):
    _, index, grouped, _ = cohort({1: {"M1": {S1: session(), S2: session()}}})
    table = suspect_sessions_table([flag(S3, name, reason="why")], grouped, index)
    (row,) = table.to_dict("records")
    assert (row["mouse"], row["session"], row["comparison"]) == ("M1", S3, "problem numbering")
    assert (row["flag"], row["reason"]) == (name, "why")
    assert row["file"] == "M1-2026-01-03-100000.tsv"
    # placed after the problem the mouse was running
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("drifting", 1, 1)
    assert pd.isna(row["key"]) and pd.isna(row["value_before"])


def test_a_provisional_number_gets_its_own_row_next_to_the_flag():
    both = flag(S2, "review_settings", problem=1, provisional=True, reason="starts wrong")
    alone = flag(S3, None, problem=1, provisional=True, reason="")
    table = settings_table({S1: session(), S2: session()}, [both, alone])
    rows = table.to_dict("records")
    assert [(r["session"], r["flag"]) for r in rows] == [
        (S2, "review_settings"),
        (S2, "provisional"),
        (S3, "provisional"),
    ]
    assert rows[1]["reason"] == PROVISIONAL_REASON
    assert rows[0]["reason"] == "starts wrong"


def test_a_flagged_session_that_is_in_a_problem_uses_that_problem():
    review = flag(S2, "review_settings", problem=1)
    (row,) = settings_table({S1: session(), S2: session()}, [review]).to_dict("records")
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("drifting", 1, 1)


# --- settings that change within a session ---


def with_settings(**changes):
    return {**SETTINGS, **changes}


def test_identical_start_and_end_settings_give_no_rows():
    assert settings_table({S1: session()}).empty


def test_a_setting_that_changed_inside_a_recording_is_listed_with_both_values():
    changed = session([recording(with_settings(), with_settings(ITI_dur=800.0))])
    table = settings_table({S1: changed})
    assert changes_of(table) == [("run_start vs run_end", "ITI_dur", "1000.0", "800.0")]
    (row,) = table.to_dict("records")
    assert (row["mouse"], row["session"], row["file"]) == ("M1", S1, "M1-2026-01-01-100000.tsv")
    assert pd.isna(row["file_after"])
    assert pd.isna(row["flag"])
    assert "ITI_dur" in row["reason"]


def test_dynamic_state_and_counters_are_ignored():
    start = with_settings(
        num_t=0,
        num_blocks=1,
        tot_rew_amt=0,
        curr_poke=None,
        found_port_counts=None,
        curr_rew_mag={"A1": 2},
        tot_num_t=5,
        rank_counts={"best": 0},
        long_pokes=[],
    )
    end = with_settings(
        num_t=25,
        num_blocks=2,
        tot_rew_amt=152,
        curr_poke="B1",
        found_port_counts={"B1": 5},
        curr_rew_mag={"A1": 4},
        tot_num_t=30,
        rank_counts={"best": 14},
        long_pokes=[1],
    )
    assert settings_table({S1: session([recording(start, end)])}).empty


def test_the_key_list_has_no_counter_or_state_names():
    state = {"num_t", "num_blocks", "tot_rew_amt", "curr_poke", "curr_rew_mag", "found_port_counts"}
    assert not state & set(SETTINGS_KEYS)
    assert "found_ports" in SETTINGS_KEYS


def test_list_and_dict_values_are_written_as_json():
    start = with_settings(rew_mags=[1, 2, 3, 4])
    end = with_settings(rew_mags=[1, 2, 3])
    table = settings_table({S1: session([recording(start, end)])})
    assert changes_of(table) == [("run_start vs run_end", "rew_mags", "[1, 2, 3, 4]", "[1, 2, 3]")]


def test_a_key_logged_on_one_side_only_is_not_a_change():
    start = {k: v for k, v in SETTINGS.items() if k != "p_ud"}
    assert settings_table({S1: session([recording(start, with_settings())])}).empty


def test_a_recording_without_run_end_gets_a_not_checked_row():
    crashed = recording(with_settings(), None, path="M1-crashed.tsv")
    table = settings_table({S1: session([crashed])})
    (row,) = table.to_dict("records")
    assert (row["session"], row["file"], row["comparison"]) == (
        S1,
        "M1-crashed.tsv",
        ss.NOT_CHECKED,
    )
    assert row["reason"] == "not checked: no run_end"
    assert pd.isna(row["key"]) and pd.isna(row["flag"])


def test_a_recording_without_run_start_gets_a_not_checked_row():
    bare = recording(None, with_settings(ITI_dur=1.0))
    (row,) = settings_table({S1: session([bare])}).to_dict("records")
    assert row["comparison"] == ss.NOT_CHECKED
    assert row["reason"] == "not checked: no run_start"


def test_a_recording_with_neither_snapshot_says_so():
    (row,) = settings_table({S1: session([recording(None, None)])}).to_dict("records")
    assert row["reason"] == "not checked: no run_start, no run_end"


def test_recordings_with_both_snapshots_get_no_not_checked_row():
    assert settings_table({S1: session()}).empty


# --- settings that change between recordings of a session ---


def test_a_setting_that_changed_between_two_recordings_names_both_files():
    first = recording(with_settings(), with_settings(), path="M1-2026-01-01-100000.tsv")
    second = recording(
        with_settings(poke_hold_dur=200), with_settings(poke_hold_dur=200), path="M1-b.tsv"
    )
    table = settings_table({S1: session([first, second])})
    assert changes_of(table) == [("run_end vs next run_start", "poke_hold_dur", "0", "200")]
    (row,) = table.to_dict("records")
    assert (row["file"], row["file_after"]) == ("M1-2026-01-01-100000.tsv", "M1-b.tsv")


def test_three_recordings_are_compared_pairwise_in_order():
    a = recording(with_settings(), with_settings(ses_length=90), path="a.tsv")
    b = recording(with_settings(ses_length=90), with_settings(ses_length=90), path="b.tsv")
    c = recording(with_settings(ses_length=45), with_settings(ses_length=45), path="c.tsv")
    table = settings_table({S1: session([a, b, c])})
    assert changes_of(table) == [
        ("run_start vs run_end", "ses_length", "60", "90"),
        ("run_end vs next run_start", "ses_length", "90", "45"),
    ]


def test_a_crashed_recording_does_not_hide_the_comparison_of_its_neighbours():
    first = recording(with_settings(), with_settings())
    crashed = recording(with_settings(), None)
    third = recording(with_settings(ITI_dur=5.0), with_settings(ITI_dur=5.0))
    table = settings_table({S1: session([first, crashed, third])})
    # no run_end on the crashed one to compare with the third; only the crashed one is listed
    assert changes_of(table) == [(ss.NOT_CHECKED, None, None, None)]


# --- the expected change at the start of a problem ---

PORT_FINDING_START = with_settings(found_ports=False)


def test_port_finding_at_the_start_of_a_problem_is_not_flagged():
    first = session([recording(PORT_FINDING_START, with_settings(found_ports=True))])
    assert settings_table({S1: first, S2: session()}).empty


def test_port_finding_in_a_later_session_is_flagged():
    found = session([recording(PORT_FINDING_START, with_settings(found_ports=True))])
    table = settings_table({S1: session(), S2: found})
    assert changes_of(table) == [("run_start vs run_end", "found_ports", "False", "True")]
    assert table["session"].tolist() == [S2]


def test_found_ports_turning_false_at_the_start_of_a_problem_is_still_flagged():
    odd = session([recording(with_settings(found_ports=True), with_settings(found_ports=False))])
    assert changes_of(settings_table({S1: odd})) == [
        ("run_start vs run_end", "found_ports", "True", "False")
    ]


def test_another_setting_changing_during_port_finding_is_flagged():
    first = session([recording(PORT_FINDING_START, with_settings(found_ports=True, ses_length=90))])
    assert changes_of(settings_table({S1: first})) == [
        ("run_start vs run_end", "ses_length", "60", "90")
    ]


def test_the_exemption_belongs_to_each_problem_of_a_mouse():
    port_finding = session([recording(PORT_FINDING_START, with_settings(found_ports=True))])
    _, index, grouped, _ = cohort(
        {1: {"M1": {S1: port_finding}}, 2: {"M1": {S2: port_finding, S3: port_finding}}}
    )
    table = suspect_sessions_table([], grouped, index)
    assert table["session"].tolist() == [S3]


# --- table shape ---


def test_columns_and_order_are_fixed_and_rows_follow_session_order():
    changed = session([recording(with_settings(), with_settings(ITI_dur=2.0))])
    table = settings_table({S2: changed, S1: changed}, [flag(S3, "port_finding", reason="pf")])
    assert list(table.columns) == list(COLUMNS)
    assert table["session"].tolist() == [S1, S2, S3]


def test_empty_input_gives_the_columns():
    table = settings_table({S1: session()})
    assert table.empty and list(table.columns) == list(COLUMNS)


# --- pure helpers ---


def test_setting_changes_compares_only_the_given_keys_present_on_both_sides():
    before = {"ITI_dur": 1, "ses_length": 60, "num_t": 0, "p_ud": 0.1}
    after = {"ITI_dur": 2, "ses_length": 60, "num_t": 9}
    assert setting_changes(before, after, ("ITI_dur", "ses_length", "p_ud", "num_t")) == [
        ("ITI_dur", 1, 2),
        ("num_t", 0, 9),
    ]


def test_read_snapshots_takes_the_first_run_start_and_the_last_run_end():
    frame = pd.DataFrame(
        [
            (0, "variable", "run_start", '{"a": 1}'),
            (1, "variable", "run_start", '{"a": 2}'),
            (2, "variable", "run_end", '{"a": 3}'),
            (3, "variable", "run_end", '{"a": 4}'),
        ],
        columns=["time", "type", "subtype", "content"],
    )
    assert read_snapshots(frame) == ({"a": 1}, {"a": 4})


def test_read_snapshots_treats_a_truncated_run_end_as_missing_but_not_a_bad_run_start():
    truncated = pd.DataFrame(
        [(0, "variable", "run_start", '{"a": 1}'), (1, "variable", "run_end", '{"a": ')],
        columns=["time", "type", "subtype", "content"],
    )
    assert read_snapshots(truncated) == ({"a": 1}, None)
    bad_start = truncated.assign(content=['{"a": ', '{"a": 1}'])
    with pytest.raises(ValueError, match="run_start"):
        read_snapshots(bad_start)


def test_the_documented_key_list_matches_the_module_constant_groups():
    assert set(SETTINGS_KEYS) == {k for keys in ss.SETTINGS_KEY_GROUPS.values() for k in keys}


# --- pinned behaviour ---


def test_port_finding_between_two_recordings_of_a_first_session_is_still_listed():
    first = recording(PORT_FINDING_START, with_settings(found_ports=False), path="a.tsv")
    second = recording(
        with_settings(found_ports=True), with_settings(found_ports=True), path="b.tsv"
    )
    table = settings_table({S1: session([first, second])})
    assert changes_of(table) == [("run_end vs next run_start", "found_ports", "False", "True")]


def test_numbering_rows_come_before_the_settings_rows_of_the_same_session():
    changed = session([recording(with_settings(), with_settings(ITI_dur=2.0))])
    table = settings_table({S1: changed}, [flag(S1, "review_settings", problem=1)])
    assert table["comparison"].tolist() == ["problem numbering", "run_start vs run_end"]


# --- which logged keys are settings ---

# Every key of a run_start / run_end in the staged drifting and blocked recordings.
LOGGED_KEYS = (
    "ITI_dur arm_to_stream base_choice_vol best_val can_switch change_chosen curr_block_len "
    "curr_poke curr_rew_mag found_port_counts found_ports found_t_thresh hw_solenoid_calibration "
    "i inactive_arm init_cue_dur init_rew_vol led long_pokes max_block_len min_block_len min_hold "
    "no_rew_cue_dur num_back_b num_back_s num_bad_rev num_blocks num_good_rev num_init_rew "
    "num_long_pokes num_problem num_rew_pulses num_t num_t_found num_t_in_block num_t_rev "
    "num_t_switch p_bi p_ud perf_thresh poke_hold_dur problems pulse_isi pulses_deliv rank_counts "
    "rew_mag rew_mags s second_perf_thresh ses_length stream_hold stream_state switch_best "
    "t_ses_start t_thresh tot_num_bad_rev tot_num_blocks tot_num_good_rev tot_num_t "
    "tot_num_t_switch tot_rew_amt with_LED with_init_rew"
).split()


def test_every_logged_key_is_either_compared_or_documented_as_not_a_setting():
    documented = {key for keys in ss.NOT_SETTINGS.values() for key in keys}
    assert not documented & set(SETTINGS_KEYS)
    assert set(LOGGED_KEYS) == documented | set(SETTINGS_KEYS)


def test_best_val_and_switch_best_are_configuration():
    assert {"best_val", "switch_best"} <= set(SETTINGS_KEYS)
    start = with_settings(best_val=4, switch_best=False)
    end = with_settings(best_val=3, switch_best=True)
    table = settings_table({S1: session([recording(start, end)])})
    assert changes_of(table) == [
        ("run_start vs run_end", "best_val", "4", "3"),
        ("run_start vs run_end", "switch_best", "False", "True"),
    ]


def test_can_switch_and_the_solenoid_constants_are_not_settings():
    start = with_settings(can_switch=False, i=10, s=7)
    end = with_settings(can_switch=True, i=11, s=8)
    assert settings_table({S1: session([recording(start, end)])}).empty
    for key in ("can_switch", "i", "s", "hw_solenoid_calibration"):
        assert key not in SETTINGS_KEYS
    assert {"i", "s", "hw_solenoid_calibration"} <= set(ss.NOT_SETTINGS["solenoid calibration"])
    assert "can_switch" in ss.NOT_SETTINGS["changes as the task runs"]
