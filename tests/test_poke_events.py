"""Parsing of raw pyControl recordings into per-trial pokes, and the two counting rules."""

import pandas as pd
import pytest

from src.behavior_analysis.poke_events import (
    cross_check_long_pokes,
    parse_recording,
    parse_session,
    pokes_after_choice,
    pokes_after_initiation,
)
from tests.poke_fixtures import Recording, full_trial, session_record

# ------------------------------------------------------------- after-choice window rules


def test_initiating_poke_of_the_next_trial_is_not_counted():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    full_trial(rec, choice="B1")  # initiates with B2_in, the poke that must not count
    first = parse_recording(rec.to_frame())[0]
    assert first.window_pokes == ("C3",)


def test_other_b2_pokes_in_the_window_are_counted():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["B2", "C3"])
    full_trial(rec, choice="B1", pre_init=["B2"])  # a B2 poke that does not initiate
    first = parse_recording(rec.to_frame())[0]
    assert first.window_pokes == ("B2", "C3", "B2")


def test_pokes_are_not_collapsed_when_repeated():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3", "C3", "C3", "A2", "A2"])
    full_trial(rec, choice="A1")
    first = parse_recording(rec.to_frame())[0]
    assert pokes_after_choice(first.window_pokes, first.chosen) == ["C3", "C3", "C3", "A2", "A2"]


def test_pokes_during_the_reward_state_precede_the_window():
    rec = Recording()
    rec.initiate()
    rec.choose("A1", rewarded=True, choice_pokes=["A1", "A1"])
    rec.iti()
    rec.poke_in("C2")
    rec.state("wait_init")
    full_trial(rec, choice="A1")
    first = parse_recording(rec.to_frame())[0]
    assert first.window_pokes == ("C2",)


def test_leading_choice_tower_pokes_are_dropped_until_another_port_is_poked():
    # The example from the analysis spec: after choosing A1, A1 A1 C3 C2 A1 A1 -> C3 C2 A1 A1.
    window = ("A1", "A1", "C3", "C2", "A1", "A1")
    assert pokes_after_choice(window, "A1") == ["C3", "C2", "A1", "A1"]


def test_choice_tower_pokes_count_when_another_port_came_first():
    assert pokes_after_choice(("B3", "A1", "A1"), "A1") == ["B3", "A1", "A1"]


def test_window_of_only_choice_tower_pokes_counts_nothing():
    assert pokes_after_choice(("A1", "A1", "A1"), "A1") == []


def test_empty_window_counts_nothing():
    assert pokes_after_choice((), "A1") == []


def test_last_trial_without_a_following_initiation_has_an_open_window():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    full_trial(rec, choice="B1", after_choice=["A3"])  # recording ends in wait_init
    trials = parse_recording(rec.to_frame())
    assert [t.window_pokes for t in trials] == [("C3",), None]


def test_recording_that_ends_before_the_iti_has_an_open_window():
    rec = Recording()
    rec.initiate()
    rec.choose("A1", rewarded=True)
    (only,) = parse_recording(rec.to_frame())
    assert only.window_pokes is None


# ------------------------------------------------------ after-initiation (long poke) rules


def test_long_pokes_include_the_choice_poke_exactly_once():
    rec = Recording()
    full_trial(rec, long_before=["A3"], shorts=["C2"], choice="A1")
    (trial,) = parse_recording(rec.to_frame())
    # The short poke at C2 is not a long poke; the choice poke is the last long poke.
    assert trial.long_pokes == ("A3", "A1")
    assert pokes_after_initiation(trial.long_pokes, "B2") == ["A3", "A1"]


def test_trial_without_non_choice_long_pokes_counts_only_the_choice():
    rec = Recording()
    full_trial(rec, choice="C1")
    (trial,) = parse_recording(rec.to_frame())
    assert pokes_after_initiation(trial.long_pokes, "B2") == ["C1"]


@pytest.mark.parametrize(
    ("long_pokes", "expected"),
    [
        (("B2", "B1"), ["B1"]),
        (("B2", "B2", "A3", "B2", "A1"), ["A3", "B2", "A1"]),
        (("B1", "B2", "A1"), ["B1", "B2", "A1"]),
        (("A1",), ["A1"]),
    ],
)
def test_initiation_tower_long_pokes_count_only_after_a_long_poke_elsewhere(long_pokes, expected):
    assert pokes_after_initiation(long_pokes, "B2") == expected


def test_initiating_poke_itself_is_not_a_long_poke_of_the_trial():
    rec = Recording()
    rec.initiate()  # B2_in -> wait_choice; its hold is not a long poke
    rec.choose("A1")
    (trial,) = parse_recording(rec.to_frame())
    assert trial.long_pokes == ("A1",)


# --------------------------------------------------------------- alignment and structure


def two_recordings_with_a_crash():
    """Recording 1: two trials, then a third that is initiated but never chosen (crash).
    Recording 2: two trials. The third trial of recording 1 has no print and no ITI."""
    first = Recording()
    full_trial(first, choice="A1", after_choice=["C3"])
    full_trial(first, choice="B1", after_choice=["A3"])
    first.initiate()  # trial never completed: the crash
    second = Recording()
    full_trial(second, choice="C1", after_choice=["B3"])
    full_trial(second, choice="A1", after_choice=["A2"])
    full_trial(second, choice="B1")
    return first, second


def test_windows_belong_to_the_trial_whose_print_precedes_them_across_recordings():
    first, second = two_recordings_with_a_crash()
    session = session_record([first.to_frame(), second.to_frame()])
    trials = parse_session(session)
    assert [t.chosen for t in trials] == ["A1", "B1", "C1", "A1", "B1"]
    # Trial 2 (B1) is followed by the crashed initiation in the same recording: closed window.
    assert [t.window_pokes for t in trials] == [
        ("C3",),
        ("A3",),
        ("B3",),
        ("A2",),
        None,
    ]


def test_alignment_with_the_sessions_merged_choice_list_is_checked():
    first, second = two_recordings_with_a_crash()
    session = session_record(
        [first.to_frame(), second.to_frame()], choice=["A1", "B1", "C1", "A1", "B1"]
    )
    assert len(parse_session(session)) == 5
    session["choice"] = ["A1", "B1", "C1", "B1", "A1"]
    with pytest.raises(ValueError, match="trial 4"):
        parse_session(session)
    session["choice"] = ["A1", "B1"]
    with pytest.raises(ValueError, match="5 trials"):
        parse_session(session)


def test_recording_without_bandit_trials_is_skipped():
    pretraining = pd.DataFrame(
        [(0.0, "state", None, "wait_init"), (1.0, "variable", "print", '{"num_t_found": 1}')],
        columns=["time", "type", "subtype", "content"],
    )
    rec = Recording()
    full_trial(rec, choice="A1")
    assert parse_recording(pretraining) == []
    assert len(parse_session(session_record([pretraining, rec.to_frame()]))) == 1


def test_initiation_event_at_another_port_raises():
    rec = Recording(init="B2")
    full_trial(rec, choice="A1")
    frame = rec.to_frame()
    with pytest.raises(ValueError, match="initiat"):
        parse_recording(frame, initiation_tower="B3")


def test_poked_in_without_hold_poke_raises():
    rec = Recording()
    full_trial(rec, choice="B1")
    rec.initiate()
    rec.poke_in("A1").state("poked_in")
    with pytest.raises(ValueError, match="poked_in"):
        parse_recording(rec.to_frame())


def test_a_second_iti_before_the_next_initiation_raises():
    rec = Recording()
    rec.initiate()
    rec.choose("A1")
    rec.iti().iti()
    with pytest.raises(ValueError, match="ITI"):
        parse_recording(rec.to_frame())


def test_time_going_backwards_raises():
    rec = Recording()
    full_trial(rec, choice="A1")
    frame = rec.to_frame()
    frame.loc[5, "time"] = -1.0
    with pytest.raises(ValueError, match="time"):
        parse_recording(frame)


# ------------------------------------------------ raw events against the printed long pokes


def test_cross_check_agrees_when_raw_and_printed_long_pokes_match():
    rec = Recording()
    full_trial(rec, long_before=["B2", "A3"], choice="A1")
    full_trial(rec, choice="B1")
    check = cross_check_long_pokes(parse_recording(rec.to_frame()))
    assert (check.n_trials, check.n_agree, check.mismatches) == (2, 2, [])


def test_cross_check_reports_a_mismatch_with_its_trial():
    rec = Recording()
    full_trial(rec, long_before=["A3"], choice="A1")
    frame = rec.to_frame()
    printed = frame["subtype"] == "print"
    frame.loc[printed, "content"] = frame.loc[printed, "content"].str.replace(
        '"long_pokes": ["A3"]', '"long_pokes": ["A2"]'
    )
    check = cross_check_long_pokes(parse_recording(frame))
    assert check.n_agree == 0
    assert check.mismatches == [(1, ("A3", "A1"), ("A2", "A1"))]
