"""Tests for trial outcome times read from a pyControl event log."""

import json

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.outcome_events import (
    OUTCOME_COLUMNS,
    OutcomeParseError,
    load_outcome_events,
    outcome_events,
    outcome_settings,
    session_intervals,
)

PULSE_ISI = 300.0
NO_REW_CUE_DUR = 500.0


def _row(time, type_, subtype, content):
    return (time, type_, subtype, content)


def _print(time, num_rew_pulses, *, phase="task", n=1, tow="B1", rank="best", **extra):
    payload = {"chosen_tow": tow, "num_rew_pulses": num_rew_pulses, "num_long_pokes": 0}
    if phase == "task":
        payload.update(num_t=n)
        if rank is not None:
            payload.update(chosen_rank=rank)
    else:
        payload.update(num_t_found=n)
    payload.update(extra)
    return _row(time, "variable", "print", json.dumps(payload))


def _rewarded(time, n=1, *, state_lag=0.006, **kwargs):
    """Rows of a rewarded trial whose print is at *time*: choice, two pulses, then the ITI."""
    choice = time + state_lag
    return [
        _print(time, 2, n=n, **kwargs),
        _row(choice, "state", None, "choice"),
        _row(choice + 0.302, "event", "timer", "pulse_off_timer"),
        _row(choice + 0.330, "event", "timer", "pulse_on_timer"),
        _row(choice + 0.634, "event", "timer", "pulse_off_timer"),
        _row(choice + 0.640, "state", None, "ITI"),
        _row(choice + 1.640, "state", None, "wait_init"),
    ]


def _unrewarded(time, n=1, *, state_lag=0.003, **kwargs):
    """Rows of an unrewarded trial: the ITI starts the no-reward cue, which ends 0.5 s later."""
    iti = time + state_lag
    return [
        _print(time, 0, n=n, rank="third", **kwargs),
        _row(iti, "state", None, "ITI"),
        _row(iti + 0.5, "event", "timer", "no_rew_cue_timer"),
        _row(iti + 1.0, "state", None, "wait_init"),
    ]


def _run_start(**overrides):
    payload = {"pulse_isi": PULSE_ISI, "no_rew_cue_dur": NO_REW_CUE_DUR, "ITI_dur": 1000.0}
    payload.update(overrides)
    return _row(0.0, "variable", "run_start", json.dumps(payload))


def _session(*row_groups, run_start=True, path="/data/RT_1.tsv"):
    rows = [_row(0.0, "info", "subject_id", "RT_1")]
    if run_start:
        rows.append(_run_start())
    for group in row_groups:
        rows += group
    df = pd.DataFrame(rows, columns=["time", "type", "subtype", "content"])
    df.attrs["source_path"] = path
    return df


# --- outcome_events: the two outcomes ---------------------------------------------------


def test_rewarded_trial_outcome_is_the_choice_state_entry():
    df = _session(_rewarded(10.0, n=1))
    (row,) = outcome_events(df).to_dict("records")

    assert row["trial"] == 1
    assert row["phase"] == "task"
    assert row["num_t"] == 1
    assert pd.isna(row["num_t_found"])
    assert row["chosen_tow"] == "B1"
    assert row["chosen_rank"] == "best"
    assert row["num_rew_pulses"] == 2
    assert row["rewarded"] is True
    assert row["print_time"] == 10.0
    assert row["outcome_time"] == pytest.approx(10.006)
    assert row["outcome_kind"] == "reward_onset"
    # first pulse_off_timer after choice entry, not the second one
    assert row["check_time"] == pytest.approx(10.006 + 0.302)


def test_unrewarded_trial_outcome_is_the_iti_entry_and_the_cue_timer_marks_its_offset():
    df = _session(_unrewarded(20.0, n=2))
    (row,) = outcome_events(df).to_dict("records")

    assert row["rewarded"] is False
    assert row["num_rew_pulses"] == 0
    assert row["chosen_rank"] == "third"
    assert row["outcome_time"] == pytest.approx(20.003)
    assert row["outcome_kind"] == "no_reward_cue_onset"
    assert row["check_time"] == pytest.approx(20.003 + 0.5)


def test_check_time_minus_outcome_time_matches_the_logged_settings():
    df = _session(_rewarded(10.0), _unrewarded(20.0))
    events = outcome_events(df)
    settings = outcome_settings(df)
    gap_ms = (events["check_time"] - events["outcome_time"]) * 1000
    rewarded = events["rewarded"].to_numpy()
    assert gap_ms[rewarded].iloc[0] == pytest.approx(settings["pulse_isi"], abs=5)
    assert gap_ms[~rewarded].iloc[0] == pytest.approx(settings["no_rew_cue_dur"], abs=5)


def test_trials_are_numbered_in_file_order_with_port_finding_marked():
    df = _session(
        _rewarded(10.0, n=1, phase="port_finding", rank=None),
        _unrewarded(20.0, n=2, phase="port_finding"),
        _rewarded(30.0, n=1),
        _unrewarded(40.0, n=2),
    )
    events = outcome_events(df)

    assert list(events.columns) == OUTCOME_COLUMNS
    assert events["trial"].tolist() == [1, 2, 3, 4]
    assert events["phase"].tolist() == ["port_finding", "port_finding", "task", "task"]
    assert events["num_t_found"].tolist()[:2] == [1, 2]
    assert events["num_t"].tolist()[2:] == [1, 2]
    assert events["num_t"].isna().tolist()[:2] == [True, True]
    assert events["num_t_found"].isna().tolist()[2:] == [True, True]
    assert events["rewarded"].tolist() == [True, False, True, False]
    assert events["chosen_rank"].tolist() == [None, None, "best", "third"]


def test_only_rows_between_a_print_and_the_next_print_are_used():
    # Trial 1 is unrewarded but its no-reward cue timer is missing; trial 2's timer must not
    # be borrowed for it.
    broken = [row for row in _unrewarded(20.0) if row[3] != "no_rew_cue_timer"]
    df = _session(broken, _unrewarded(30.0))
    with pytest.raises(OutcomeParseError, match="trial 1.*no_rew_cue_timer"):
        outcome_events(df)


def test_rows_before_the_first_print_are_ignored():
    stray = [_row(1.0, "state", None, "choice"), _row(1.5, "event", "timer", "pulse_off_timer")]
    df = _session(stray, _unrewarded(20.0))
    assert outcome_events(df)["trial"].tolist() == [1]


def test_the_last_trial_reads_to_the_end_of_the_file():
    df = _session(_rewarded(10.0), _rewarded(30.0))
    assert len(outcome_events(df)) == 2


def test_a_session_without_prints_gives_an_empty_table_with_the_columns():
    events = outcome_events(_session([_row(5.0, "state", None, "wait_init")]))
    assert events.empty
    assert list(events.columns) == OUTCOME_COLUMNS


def test_extra_keys_in_the_print_are_ignored():
    df = _session(_rewarded(10.0, tot_rew_amt=99, choice_counts={"B1": 3}))
    assert len(outcome_events(df)) == 1


def test_traceback_rows_are_dropped_like_everywhere_else_in_the_repo():
    traceback = [_row("AttributeError: ", "variable", None, np.nan)]
    df = _session(_rewarded(10.0), _unrewarded(20.0), traceback)
    with pytest.warns(UserWarning, match="non-numeric"):
        events = outcome_events(df)
    assert events["trial"].tolist() == [1, 2]


# --- outcome_events: errors -------------------------------------------------------------


def _drop(rows, content):
    return [r for r in rows if r[3] != content]


def test_rewarded_trial_without_a_choice_state_raises():
    df = _session(_drop(_rewarded(10.0), "choice"))
    with pytest.raises(OutcomeParseError, match=r"RT_1\.tsv.*trial 1.*choice"):
        outcome_events(df)


def test_a_second_choice_state_raises():
    rows = _rewarded(10.0)
    rows.insert(2, _row(10.02, "state", None, "choice"))
    with pytest.raises(OutcomeParseError, match="trial 1.*twice"):
        outcome_events(_session(rows))


def test_a_choice_state_more_than_100_ms_after_the_print_raises():
    with pytest.raises(OutcomeParseError, match="trial 1.*100 ms"):
        outcome_events(_session(_rewarded(10.0, state_lag=0.15)))


def test_a_choice_state_just_inside_100_ms_is_accepted():
    assert len(outcome_events(_session(_rewarded(10.0, state_lag=0.099)))) == 1


def test_unrewarded_trial_with_a_choice_state_raises():
    rows = _unrewarded(10.0)
    rows.insert(1, _row(10.002, "state", None, "choice"))
    with pytest.raises(OutcomeParseError, match="trial 1.*unrewarded.*choice"):
        outcome_events(_session(rows))


def test_unrewarded_trial_without_an_iti_raises():
    with pytest.raises(OutcomeParseError, match="trial 1.*ITI"):
        outcome_events(_session(_drop(_unrewarded(10.0), "ITI")))


def test_a_second_iti_state_raises():
    rows = _unrewarded(10.0)
    rows.insert(2, _row(10.2, "state", None, "ITI"))
    with pytest.raises(OutcomeParseError, match="trial 1.*twice"):
        outcome_events(_session(rows))


def test_an_iti_more_than_100_ms_after_the_print_raises():
    with pytest.raises(OutcomeParseError, match="trial 1.*100 ms"):
        outcome_events(_session(_unrewarded(10.0, state_lag=0.5)))


def test_rewarded_trial_without_a_pulse_off_timer_raises():
    with pytest.raises(OutcomeParseError, match="trial 1.*pulse_off_timer"):
        outcome_events(_session(_drop(_rewarded(10.0), "pulse_off_timer")))


def test_error_names_the_trial_and_the_file():
    df = _session(_rewarded(10.0), _drop(_rewarded(20.0), "choice"), path="/x/RT_9_session.tsv")
    with pytest.raises(OutcomeParseError, match=r"RT_9_session\.tsv.*trial 2"):
        outcome_events(df)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"chosen_tow": "B1", "num_rew_pulses": 2}, "num_t"),
        ({"num_t": 1, "num_t_found": 1, "chosen_tow": "B1", "num_rew_pulses": 2}, "both"),
        ({"num_t": 1, "num_rew_pulses": 2}, "chosen_tow"),
        ({"num_t": 1, "chosen_tow": "B1"}, "num_rew_pulses"),
        ({"num_t": 1, "chosen_tow": "B1", "num_rew_pulses": -1}, "num_rew_pulses"),
    ],
)
def test_print_with_unusable_fields_raises(payload, message):
    df = _session([_row(10.0, "variable", "print", json.dumps(payload))])
    with pytest.raises(OutcomeParseError, match=f"trial 1.*{message}"):
        outcome_events(df)


def test_print_that_is_not_json_raises():
    df = _session([_row(10.0, "variable", "print", "plain text")])
    with pytest.raises(OutcomeParseError, match="trial 1"):
        outcome_events(df)


# --- outcome_settings / session_intervals -----------------------------------------------


def test_settings_are_read_from_run_start_in_ms():
    assert outcome_settings(_session(_rewarded(10.0))) == {
        "pulse_isi": PULSE_ISI,
        "no_rew_cue_dur": NO_REW_CUE_DUR,
    }


def test_settings_without_run_start_raise():
    with pytest.raises(OutcomeParseError, match="run_start"):
        outcome_settings(_session(_rewarded(10.0), run_start=False))


def test_settings_with_two_run_starts_raise():
    with pytest.raises(OutcomeParseError, match="more than one run_start"):
        outcome_settings(_session([_run_start()], _rewarded(10.0)))


def test_settings_missing_a_field_raise():
    rows = [_row(0.0, "variable", "run_start", json.dumps({"pulse_isi": 300.0}))]
    with pytest.raises(OutcomeParseError, match="no_rew_cue_dur"):
        outcome_settings(_session(rows, run_start=False))


def test_session_intervals_are_the_first_and_last_row_times():
    df = _session(_rewarded(10.0), _unrewarded(20.0))
    start, end = session_intervals(df)
    assert start == 0.0
    assert end == pytest.approx(21.003)


def test_session_intervals_of_an_empty_table_raise():
    empty = pd.DataFrame(columns=["time", "type", "subtype", "content"])
    with pytest.raises(OutcomeParseError, match="no rows"):
        session_intervals(empty)


# --- loader -----------------------------------------------------------------------------


def test_loader_reads_the_tsv_and_names_it_in_errors(tmp_path):
    good = _session(_rewarded(10.0), _unrewarded(20.0))
    path = tmp_path / "RT_1-2026-10-16.tsv"
    good.to_csv(path, sep="\t", index=False)
    events = load_outcome_events(path)
    assert events["rewarded"].tolist() == [True, False]

    bad = _session(_drop(_rewarded(10.0), "choice"))
    bad_path = tmp_path / "RT_1-bad.tsv"
    bad.to_csv(bad_path, sep="\t", index=False)
    with pytest.raises(OutcomeParseError, match=r"RT_1-bad\.tsv.*trial 1"):
        load_outcome_events(bad_path)
