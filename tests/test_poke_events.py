"""Parsing of raw pyControl recordings into per-trial pokes, and the two counting rules."""

import warnings

import pandas as pd
import pytest

from src.behavior_analysis.poke_events import (
    NO_PRINT,
    PokeAlignmentError,
    PokeParseError,
    count_port_finding,
    cross_check_long_pokes,
    is_port_finding,
    parse_recording,
    parse_session,
    pokes_after_choice,
    pokes_after_initiation,
)
from tests.poke_fixtures import Recording, full_trial, session_record

# ------------------------------------------------------------- after-choice window rules


def test_initiating_poke_of_the_next_trial_is_the_last_poke_of_the_window():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    full_trial(rec, choice="B1")  # initiates with B2_in, the last poke of the previous window
    first = parse_recording(rec.to_frame())[0]
    assert first.window_pokes == ("C3", "B2")


def test_initiating_poke_is_counted_after_the_choice_tower_pokes_are_dropped():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["A1", "A1", "C3"])
    full_trial(rec, choice="B1")
    first = parse_recording(rec.to_frame())[0]
    assert pokes_after_choice(first.window_pokes, first.chosen) == ["C3", "B2"]


def test_a_window_with_only_choice_tower_pokes_still_counts_the_initiating_poke():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["A1", "A1"])
    full_trial(rec, choice="B1")
    first = parse_recording(rec.to_frame())[0]
    assert pokes_after_choice(first.window_pokes, first.chosen) == ["B2"]


def test_other_b2_pokes_in_the_window_are_counted():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["B2", "C3"])
    full_trial(rec, choice="B1", pre_init=["B2"])  # a B2 poke that does not initiate
    first = parse_recording(rec.to_frame())[0]
    assert first.window_pokes == ("B2", "C3", "B2", "B2")


def test_repeated_pokes_at_one_port_count_as_one_visit():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3", "C3", "C3", "A2", "A2"])
    full_trial(rec, choice="A1")
    first = parse_recording(rec.to_frame())[0]
    assert pokes_after_choice(first.window_pokes, first.chosen) == ["C3", "A2", "B2"]


def test_pokes_during_the_reward_state_precede_the_window():
    rec = Recording()
    rec.initiate()
    rec.choose("A1", rewarded=True, choice_pokes=["A1", "A1"])
    rec.iti()
    rec.poke_in("C2")
    rec.state("wait_init")
    full_trial(rec, choice="A1")
    first = parse_recording(rec.to_frame())[0]
    assert first.window_pokes == ("C2", "B2")


def test_leading_choice_tower_pokes_are_dropped_until_another_port_is_poked():
    # The example from the analysis spec: after choosing A1, A1 A1 C3 C2 A1 A1 gives the
    # visits A1 C3 C2 A1; the first A1 visit is dropped.
    window = ("A1", "A1", "C3", "C2", "A1", "A1")
    assert pokes_after_choice(window, "A1") == ["C3", "C2", "A1"]


def test_beam_flicker_after_one_poke_elsewhere_counts_once():
    # MY_71_R ses-229 trial 36: B1 x4, A1, then B1 x59 while drinking -> A1 and one B1 visit.
    window = ("B1",) * 4 + ("A1",) + ("B1",) * 59 + ("B2",)
    assert pokes_after_choice(window, "B1") == ["A1", "B1", "B2"]


def test_a_port_visited_twice_with_another_port_between_counts_twice():
    assert pokes_after_choice(("C3", "A2", "C3", "C3"), "A1") == ["C3", "A2", "C3"]


def test_choice_tower_pokes_count_when_another_port_came_first():
    assert pokes_after_choice(("B3", "A1", "A1"), "A1") == ["B3", "A1"]


def test_window_of_only_choice_tower_pokes_counts_nothing():
    assert pokes_after_choice(("A1", "A1", "A1"), "A1") == []


def test_empty_window_counts_nothing():
    assert pokes_after_choice((), "A1") == []


def test_last_trial_without_a_following_initiation_has_an_open_window():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    full_trial(rec, choice="B1", after_choice=["A3"])  # recording ends in wait_init
    trials = parse_recording(rec.to_frame())
    assert [t.window_pokes for t in trials] == [("C3", "B2"), None]


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
        ("C3", "B2"),
        ("A3", "B2"),
        ("B3", "B2"),
        ("A2", "B2"),
        None,
    ]


def test_alignment_with_the_sessions_merged_choice_list_is_checked():
    first, second = two_recordings_with_a_crash()
    session = session_record(
        [first.to_frame(), second.to_frame()], choice=["A1", "B1", "C1", "A1", "B1"]
    )
    assert len(parse_session(session)) == 5
    session["choice"] = ["A1", "B1", "C1", "B1", "A1"]
    with pytest.raises(PokeAlignmentError, match="trial 4"):
        parse_session(session)
    session["choice"] = ["A1", "B1"]
    with pytest.raises(PokeAlignmentError, match="5 trials"):
        parse_session(session)


# ------------------------------------------------------------------------ port finding


def test_port_finding_trials_are_counted_separately_and_are_not_drops():
    rec = Recording()
    rec.port_finding_trial("A1")
    rec.port_finding_trial("C1")
    full_trial(rec, choice="B1", after_choice=["C3"])
    full_trial(rec, choice="A1")
    session = session_record([rec.to_frame()], choice=["B1", "A1"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # port finding must not raise the "dropped" warning
        trials = parse_session(session)
    assert count_port_finding(trials) == 2
    task = [t for t in trials if not is_port_finding(t)]
    assert [(t.chosen, t.drop_reason, t.window_pokes) for t in task] == [
        ("B1", None, ("C3", "B2")),
        ("A1", None, None),
    ]


def test_port_finding_between_task_trials_does_not_disturb_the_open_window():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    rec.port_finding_trial("B3")
    full_trial(rec, choice="B1")
    trials = parse_recording(rec.to_frame())
    assert count_port_finding(trials) == 1
    task = [t for t in trials if not is_port_finding(t)]
    assert [t.drop_reason for t in task] == [None, None]
    assert task[0].window_pokes == ("C3", "B2")


def test_a_recording_of_only_port_finding_counts_its_trials():
    rec = Recording()
    rec.port_finding_trial("A1")
    rec.port_finding_trial("C1")
    trials = parse_recording(rec.to_frame())
    assert count_port_finding(trials) == 2
    assert not any(t.has_print for t in trials)


def test_recording_without_bandit_trials_is_skipped():
    pretraining = pd.DataFrame(
        [
            (0.0, "state", None, "wait_init"),
            (1.0, "variable", "print", '{"num_t_found": 1, "found_ports": false}'),
        ],
        columns=["time", "type", "subtype", "content"],
    )
    rec = Recording()
    full_trial(rec, choice="A1")
    (only,) = parse_recording(pretraining)
    assert is_port_finding(only)
    trials = parse_session(session_record([pretraining, rec.to_frame()]))
    assert [t.chosen for t in trials if not is_port_finding(t)] == ["A1"]


def test_port_finding_timeout_is_port_finding_not_a_dropped_task_trial():
    rec = Recording()
    rec.port_finding_timeout()
    rec.port_finding_trial("A1")
    full_trial(rec, choice="C1")
    session = session_record([rec.to_frame()], choice=["C1"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trials = parse_session(session)
    assert count_port_finding(trials) == 2
    assert [t.chosen for t in trials if not is_port_finding(t)] == ["C1"]


def test_timeout_after_the_last_port_finding_print_is_a_task_drop():
    rec = Recording()
    rec.port_finding_trial("A1")
    rec.port_finding_timeout()  # ports are found by now: the next print is a task trial
    full_trial(rec, choice="C1")
    trials = parse_recording(rec.to_frame())
    assert count_port_finding(trials) == 1
    assert [t.drop_reason for t in trials if not is_port_finding(t)] == [NO_PRINT, None]


@pytest.mark.parametrize(
    "printed",
    [
        {"num_t_found": 1, "found_ports": True},
        {"num_t_found": 1},
        {"num_t_found": 1, "found_ports": False, "num_t": 3, "chosen_tow": "A1"},
    ],
    ids=["ports-found", "no-found-ports-field", "also-has-num_t"],
)
def test_print_that_only_partly_matches_the_port_finding_rule_raises(printed):
    rec = Recording()
    full_trial(rec, choice="A1")
    rec.initiate()
    rec.raw_print(printed)
    with pytest.raises(PokeParseError, match="port-finding"):
        parse_recording(rec.to_frame())


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


def test_state_rows_going_backwards_in_time_raise():
    rec = Recording()
    full_trial(rec, choice="A1")
    frame = rec.to_frame()
    last_state = frame.index[frame["type"] == "state"][-1]
    frame.loc[last_state, "time"] = -1.0
    with pytest.raises(PokeParseError, match="state rows"):
        parse_recording(frame)


def test_timer_events_logged_before_the_preceding_state_row_are_ignored():
    # pyControl may log a timer a few ms before the state row that precedes it.
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    full_trial(rec, choice="B1")
    clean = parse_recording(rec.to_frame())
    frame = rec.to_frame()
    wait_choice = frame.index[frame["content"] == "wait_choice"][0]
    early = frame.loc[[wait_choice]].assign(
        time=frame.loc[wait_choice, "time"] - 0.005,
        type="event",
        subtype="timer",
        content="init_rew_timer",
    )
    frame = pd.concat([frame.iloc[: wait_choice + 1], early, frame.iloc[wait_choice + 1 :]])
    frame = frame.reset_index(drop=True)
    assert frame["time"].diff().min() < 0  # the log really is out of time order
    assert parse_recording(frame) == clean


def test_input_events_logged_out_of_time_order_are_read_in_file_order():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3", "A2"])
    full_trial(rec, choice="B1")
    frame = rec.to_frame()
    clean = parse_recording(frame)
    poke = frame.index[frame["content"] == "C3_in"][0]
    frame.loc[poke, "time"] = frame.loc[poke - 1, "time"] - 0.003
    assert parse_recording(frame) == clean


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


# ------------------------------------------------- non-initiating B2 pokes count as elsewhere


def test_a_non_initiating_b2_poke_counts_as_elsewhere_for_the_chosen_tower_rule():
    assert pokes_after_choice(("A1", "B2", "A1", "A1"), "A1") == ["B2", "A1"]


def test_b2_poke_in_the_iti_lets_later_chosen_tower_pokes_count_in_a_parsed_trial():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["A1", "B2", "A1"])
    full_trial(rec, choice="B1")
    first = parse_recording(rec.to_frame())[0]
    assert pokes_after_choice(first.window_pokes, first.chosen) == ["B2", "A1", "B2"]


# ------------------------------------------------------------------ dropped trials


def test_parser_errors_are_poke_parse_errors_and_value_errors():
    assert issubclass(PokeParseError, ValueError)
    rec = Recording()
    full_trial(rec, choice="A1")
    rec.initiate()
    rec.poke_in("A1").state("poked_in")
    with pytest.raises(PokeParseError):
        parse_recording(rec.to_frame())


def test_aborted_trial_without_a_print_is_dropped_and_not_a_session_trial():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])
    rec.initiate()  # initiated, then the trial times out: ITI with no print
    rec.iti().state("wait_init")
    full_trial(rec, choice="B1")
    trials = parse_recording(rec.to_frame())
    assert [(t.chosen, t.drop_reason is None, t.has_print) for t in trials] == [
        ("A1", True, True),
        (None, False, False),
        ("B1", True, True),
    ]
    # The first trial's window is closed by the aborted trial's initiation.
    assert trials[0].window_pokes == ("C3", "B2")
    session = session_record([rec.to_frame()], choice=["A1", "B1"])
    with pytest.warns(UserWarning, match="dropped 1 trial"):
        assert len(parse_session(session)) == 3


def test_plain_text_print_is_a_dropped_trial_that_keeps_the_alignment():
    rec = Recording()
    full_trial(rec, choice="A1")
    rec.initiate()
    rec.long_poke("A3")
    rec._long("B1")
    rec.text_print()
    rec.iti().state("wait_init")
    full_trial(rec, choice="C1")
    session = session_record([rec.to_frame()], choice=["A1", None, "C1"])
    with pytest.warns(UserWarning, match="no summary print"):
        trials = parse_session(session)
    assert [t.drop_reason is None for t in trials] == [True, False, True]


def test_trial_that_returns_to_wait_init_without_an_iti_is_dropped_not_open():
    rec = Recording()
    rec.initiate()
    rec.choose("A1")
    rec.state("wait_init")  # no ITI
    full_trial(rec, choice="B1", after_choice=["C3"])
    full_trial(rec, choice="C1")
    first, second, third = parse_recording(rec.to_frame())
    assert first.drop_reason is not None and "ITI" in first.drop_reason
    assert second.window_pokes == ("C3", "B2") and third.window_pokes is None


def test_choice_poke_port_different_from_the_printed_choice_drops_the_trial():
    rec = Recording()
    rec.initiate()
    rec.choose("A1", held="A2")
    rec.iti().state("wait_init")
    full_trial(rec, choice="B1")
    first, _ = parse_recording(rec.to_frame())
    assert first.drop_reason is not None and "A2" in first.drop_reason
    assert first.chosen == "A1"


def test_drops_warn_once_per_session_with_the_count():
    rec = Recording()
    rec.initiate()
    rec.choose("A1", held="A2")
    rec.iti().state("wait_init")
    rec.initiate()
    rec.choose("B1", held="B3")
    rec.iti().state("wait_init")
    full_trial(rec, choice="C1")
    with pytest.warns(UserWarning, match="dropped 2 trials") as caught:
        parse_session(session_record([rec.to_frame()]))
    assert len(caught) == 1


def test_clean_session_does_not_warn():
    rec = Recording()
    full_trial(rec, choice="A1")
    full_trial(rec, choice="B1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse_session(session_record([rec.to_frame()]))


def test_cross_check_ignores_dropped_trials():
    rec = Recording()
    rec.initiate()
    rec.choose("A1", held="A2")
    rec.iti().state("wait_init")
    full_trial(rec, choice="B1")
    check = cross_check_long_pokes(parse_recording(rec.to_frame()))
    assert (check.n_trials, check.n_agree) == (1, 1)


def test_alignment_error_is_not_a_parse_error():
    assert issubclass(PokeAlignmentError, ValueError)
    assert not issubclass(PokeAlignmentError, PokeParseError)
    assert not issubclass(PokeParseError, PokeAlignmentError)


def test_print_with_no_initiation_since_the_previous_trial_drops_both_trials():
    rec = Recording()
    full_trial(rec, choice="A1", after_choice=["C3"])  # ends in wait_init, no initiation follows
    rec.choose("B1")  # a print without an initiation
    rec.iti().state("wait_init")
    full_trial(rec, choice="C1")
    first, second, third = parse_recording(rec.to_frame())
    assert first.drop_reason is not None and "without an initiation" in first.drop_reason
    assert second.drop_reason is not None and "without an initiation" in second.drop_reason
    assert third.drop_reason is None
    assert first.long_pokes == second.long_pokes == () and first.window_pokes is None


# ------------------------------------------------- pokes of port-finding trials


def port_finding_around_task_trials():
    """Port finding (with pokes), a task trial, more port finding, a task trial, in file order."""
    rec = Recording()
    rec.port_finding_trial("A1", long_before=["C3"], after_choice=["A1", "C2", "C2"])
    full_trial(rec, choice="B1", after_choice=["C3"])
    rec.port_finding_trial("C1")
    full_trial(rec, choice="A1")
    return rec


def test_port_finding_trials_keep_their_pokes_and_stay_in_file_order():
    trials = parse_recording(port_finding_around_task_trials().to_frame())
    assert [(is_port_finding(t), t.chosen) for t in trials] == [
        (True, "A1"),
        (False, "B1"),
        (True, "C1"),
        (False, "A1"),
    ]
    first, task, second, _ = trials
    assert not first.has_print and first.drop_reason is None
    assert first.long_pokes == ("C3", "A1")
    assert first.window_pokes == ("A1", "C2", "C2", "B2")  # closed by the task trial's initiation
    assert task.window_pokes == ("C3", "B2")  # closed by the port-finding trial's initiation
    assert second.window_pokes == ("B2",)


def test_port_finding_pokes_follow_the_same_two_counting_rules():
    first = parse_recording(port_finding_around_task_trials().to_frame())[0]
    assert pokes_after_choice(first.window_pokes, first.chosen) == ["C2", "B2"]
    assert pokes_after_initiation(first.long_pokes, "B2") == ["C3", "A1"]
    rec = Recording()
    rec.port_finding_trial("A1", long_before=["B2", "A3"])
    (only,) = parse_recording(rec.to_frame())
    assert pokes_after_initiation(only.long_pokes, "B2") == ["A3", "A1"]


def test_port_finding_does_not_change_the_task_trials():
    plain = Recording()
    full_trial(plain, choice="B1", after_choice=["C3"])
    full_trial(plain, choice="A1")
    with_port_finding = port_finding_around_task_trials()
    task = [t for t in parse_recording(with_port_finding.to_frame()) if not is_port_finding(t)]
    assert task == parse_recording(plain.to_frame())


def test_port_finding_trials_stay_in_file_order_across_recordings():
    first = Recording()
    first.port_finding_trial("A1")
    full_trial(first, choice="B1")
    second = Recording()
    second.port_finding_trial("C1")
    full_trial(second, choice="A1")
    session = session_record([first.to_frame(), second.to_frame()], choice=["B1", "A1"])
    trials = parse_session(session)
    assert [(is_port_finding(t), t.chosen) for t in trials] == [
        (True, "A1"),
        (False, "B1"),
        (True, "C1"),
        (False, "A1"),
    ]


def test_a_port_finding_trial_ended_by_recording_end_has_an_open_window():
    rec = Recording()
    rec.port_finding_trial("A1")
    (only,) = parse_recording(rec.to_frame())
    assert only.window_pokes is None and only.long_pokes == ("A1",)


def test_pretraining_recording_ignores_plain_task_prints():
    rec = Recording()
    rec.initiate()
    rec.raw_print({"num_t": 1, "init_dur": 3})
    rec.iti().state("wait_init")
    assert parse_recording(rec.to_frame()) == []


def test_port_finding_trial_with_the_wrong_held_port_is_dropped_and_warned_about():
    rec = Recording()
    rec.port_finding_trial("A1", held="C1")
    full_trial(rec, choice="B1")
    with pytest.warns(UserWarning, match=r"dropped 1 trial.*port finding.*choice poke was held"):
        trials = parse_session(session_record([rec.to_frame()], choice=["B1"]))
    first = trials[0]
    assert is_port_finding(first) and first.drop_reason
    assert first.long_pokes == () and first.window_pokes is None


def test_port_finding_print_without_a_chosen_tower_is_dropped_not_guessed():
    rec = Recording()
    rec.port_finding_trial("A1", chosen=False)
    full_trial(rec, choice="B1")
    with pytest.warns(UserWarning, match="port finding"):
        first = parse_session(session_record([rec.to_frame()], choice=["B1"]))[0]
    assert is_port_finding(first) and first.chosen is None
    assert first.drop_reason and first.long_pokes == ()


def test_a_timed_out_port_finding_trial_has_no_pokes_to_count():
    rec = Recording()
    rec.port_finding_timeout()
    rec.port_finding_trial("A1")
    full_trial(rec, choice="C1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a port-finding timeout is expected, not a drop
        timed_out, found, task = parse_session(session_record([rec.to_frame()], choice=["C1"]))
    assert is_port_finding(timed_out) and timed_out.drop_reason == NO_PRINT
    assert timed_out.long_pokes == () and timed_out.window_pokes is None
    assert is_port_finding(found) and found.drop_reason is None
    assert not is_port_finding(task)


def test_an_unreadable_port_finding_only_recording_does_not_stop_the_session():
    pretraining = Recording(init="B3")  # initiates from B3 where the session's tower is B2
    pretraining.port_finding_trial("A1")
    pretraining.port_finding_trial("C1")
    rec = Recording()
    full_trial(rec, choice="B1")
    session = session_record([pretraining.to_frame(), rec.to_frame()], choice=["B1"])
    with pytest.warns(UserWarning, match=r"dropped 2 trials.*port finding.*initiation"):
        trials = parse_session(session)
    unreadable, task = trials[:2], trials[2:]
    assert all(is_port_finding(t) and t.drop_reason and t.long_pokes == () for t in unreadable)
    assert [t.chosen for t in task] == ["B1"]


def test_an_unreadable_recording_that_holds_task_trials_still_raises():
    rec = Recording(init="B3")
    rec.port_finding_trial("A1")
    full_trial(rec, choice="B1")
    with pytest.raises(PokeParseError, match="initiat"):
        parse_recording(rec.to_frame())
