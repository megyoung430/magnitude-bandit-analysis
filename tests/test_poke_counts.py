"""Per-port poke counts per session, mouse and problem, and previous-problem choice ports."""

import pytest

from src.behavior_analysis.poke_counts import (
    mouse_poke_counts,
    previous_choice_towers_for_problem,
    previous_problem_choice_towers,
    session_choice_ports,
    session_poke_counts,
    total_counts,
    total_port_finding_counts,
)
from src.behavior_analysis.poke_events import ARMS9
from tests.poke_fixtures import Recording, full_trial, session_record


def small_session(**extra):
    """Two trials with known pokes, plus an open final window.

    Trial 1: long pokes A3 then choice A1; after the choice: A1 (dropped), C3, C3, B2, A1,
    then the initiating B2 of trial 2.
    Trial 2: choice B1 only; its window is open (recording ends in wait_init).
    """
    rec = Recording()
    full_trial(rec, long_before=["A3"], choice="A1", after_choice=["A1", "C3", "C3", "B2", "A1"])
    full_trial(rec, choice="B1")
    return session_record([rec.to_frame()], **extra)


def test_session_counts_cover_all_nine_ports():
    counts = session_poke_counts(small_session())
    assert set(counts.after_initiation) == set(ARMS9) == set(counts.after_choice)


def test_after_initiation_counts_long_pokes_with_the_choice_exactly_once():
    counts = session_poke_counts(small_session())
    assert {p: n for p, n in counts.after_initiation.items() if n} == {"A3": 1, "A1": 1, "B1": 1}


def test_after_choice_counts_repeats_the_initiating_poke_and_drops_leading_choice_pokes():
    counts = session_poke_counts(small_session())
    assert {p: n for p, n in counts.after_choice.items() if n} == {"C3": 1, "B2": 2, "A1": 1}


def test_trials_with_an_open_window_are_excluded_from_after_choice_and_reported():
    counts = session_poke_counts(small_session())
    assert (counts.n_trials, counts.n_open_windows) == (2, 1)


def test_long_poke_check_is_attached():
    counts = session_poke_counts(small_session())
    assert counts.long_poke_check.n_agree == counts.long_poke_check.n_trials == 2


def test_port_finding_trials_are_counted_apart_from_task_trials_and_drops():
    rec = Recording()
    rec.port_finding_trial("A1")
    rec.port_finding_trial("C1")
    rec.port_finding_trial("B3")
    full_trial(rec, choice="A1")
    full_trial(rec, choice="B1")
    counts = session_poke_counts(session_record([rec.to_frame()], choice=["A1", "B1"]))
    assert counts.n_port_finding_trials == 3
    assert (counts.n_trials, counts.n_valid) == (2, 2)
    assert (counts.n_dropped_no_print, counts.n_dropped_inconsistent) == (0, 0)


def test_session_without_port_finding_reports_zero_port_finding_trials():
    counts = session_poke_counts(small_session())
    assert counts.n_port_finding_trials == 0
    assert counts.port_finding.n_valid == 0
    assert sum(counts.port_finding.after_choice.values()) == 0


def port_finding_session():
    """Port finding (with a timeout among it) followed by the two task trials of small_session."""
    rec = Recording()
    rec.port_finding_trial("A1", long_before=["C3"], after_choice=["A1", "C2", "C2"])
    rec.port_finding_trial("C1", long_before=["B2"], after_choice=["C1", "A2"])
    rec.port_finding_timeout()
    rec.port_finding_trial("B3")
    full_trial(rec, long_before=["A3"], choice="A1", after_choice=["A1", "C3", "C3", "B2", "A1"])
    full_trial(rec, choice="B1")
    return session_record([rec.to_frame()], choice=["A1", "B1"])


def nonzero(counts):
    return {port: n for port, n in counts.items() if n}


def test_port_finding_pokes_are_counted_under_the_two_task_rules():
    found = session_poke_counts(port_finding_session()).port_finding
    # long pokes: a leading B2 is dropped and the choice poke counts
    assert nonzero(found.after_initiation) == {"C3": 1, "A1": 1, "C1": 1, "B3": 1}
    # visits after the choice, leading visits to the chosen tower dropped, initiating B2 last
    assert nonzero(found.after_choice) == {"C2": 1, "A2": 1, "B2": 3}
    assert (found.n_valid, found.n_open_windows, found.n_dropped) == (3, 0, 1)


def test_port_finding_leaves_every_task_count_as_without_it():
    with_port_finding = session_poke_counts(port_finding_session())
    task_only = session_poke_counts(small_session())
    task_fields = (
        "after_initiation",
        "after_choice",
        "n_trials",
        "n_open_windows",
        "n_valid",
        "n_dropped_no_print",
        "n_dropped_inconsistent",
    )
    for field in task_fields:
        assert getattr(with_port_finding, field) == getattr(task_only, field), field
    assert with_port_finding.long_poke_check == task_only.long_poke_check
    assert with_port_finding.n_port_finding_trials == 4


def test_port_finding_totals_sum_over_the_sessions_of_a_mouse():
    sessions = {
        "ses-1_date-20260101": port_finding_session(),
        "ses-2_date-20260102": small_session(),  # no port finding: adds nothing
        "ses-3_date-20260103": port_finding_session(),
    }
    per_session = mouse_poke_counts("m1", sessions)
    assert nonzero(total_port_finding_counts(per_session, "after_choice")) == {
        "C2": 2,
        "A2": 2,
        "B2": 6,
    }
    assert nonzero(total_port_finding_counts(per_session, "after_initiation"))["B3"] == 2
    with pytest.raises(ValueError, match="kind"):
        total_port_finding_counts(per_session, "long_pokes")


def test_mouse_counts_sum_over_sessions_in_session_order_not_dict_order():
    late, early = small_session(), small_session()
    sessions = {"ses-10_date-20260110": late, "ses-9_date-20260109": early}
    per_session = mouse_poke_counts("m1", sessions)
    assert list(per_session) == ["ses-9_date-20260109", "ses-10_date-20260110"]
    assert total_counts(per_session, "after_initiation")["B1"] == 2
    assert total_counts(per_session, "after_choice")["C3"] == 2


# ---------------------------------------------------------------- choice ports


def test_choice_ports_from_a_set_and_from_per_trial_lists():
    assert session_choice_ports({"choice_towers": {"A1", "B1", "C1"}}) == {"A1", "B1", "C1"}
    per_trial = [["A1", "B1", "C1"], ["A1", "B1", "C1"]]
    assert session_choice_ports({"choice_towers": per_trial}) == {"A1", "B1", "C1"}


@pytest.mark.parametrize("sess", [{}, {"choice_towers": []}, {"choice_towers": [[]]}])
def test_choice_ports_missing_or_empty_raises(sess):
    with pytest.raises(ValueError, match="choice towers"):
        session_choice_ports(sess)


def ports_session(ports):
    return {"choice_towers": set(ports)}


def three_problems():
    """Mouse m1 has problems 2, 3, 10 (dict order scrambled); m2 only problem 3."""
    return {
        10: {"m1": {"ses-3_date-20260103": ports_session(["C1", "C2", "C3"])}},
        3: {
            "m1": {"ses-2_date-20260102": ports_session(["A1", "B1", "C1"])},
            "m2": {"ses-2_date-20260102": ports_session(["A1", "B1", "C1"])},
        },
        2: {"m1": {"ses-1_date-20260101": ports_session(["A3", "B3", "C3"])}},
    }


def test_previous_problem_follows_problem_order_not_dict_order():
    previous = previous_problem_choice_towers(three_problems())
    assert previous["m1"][10] == {"A1", "B1", "C1"}
    assert previous["m1"][3] == {"A3", "B3", "C3"}


def test_mouse_without_an_earlier_problem_has_none():
    previous = previous_problem_choice_towers(three_problems())
    assert previous["m1"][2] is None
    assert previous["m2"][3] is None


def test_previous_problem_skips_a_problem_the_mouse_did_not_run():
    data = three_problems()
    del data[3]["m1"]
    assert previous_problem_choice_towers(data)["m1"][10] == {"A3", "B3", "C3"}


def test_previous_choice_towers_for_a_problem_uses_each_sessions_overall_problem():
    previous = previous_problem_choice_towers(three_problems())
    # A task-type regrouped problem: m1's overall problem 10 next to m2's overall problem 3.
    regrouped = {
        "m1": {"ses-3_date-20260103": {**ports_session("C1"), "problem": 10}},
        "m2": {"ses-2_date-20260102": {**ports_session("A1"), "problem": 3}},
    }
    assert previous_choice_towers_for_problem(regrouped, previous) == {
        "m1": {"A1", "B1", "C1"},
        "m2": None,
    }


def test_mouse_or_problem_missing_from_previous_raises_a_clear_error():
    previous = previous_problem_choice_towers(three_problems())
    unknown_mouse = {"m9": {"ses-1_date-20260101": {**ports_session("C1"), "problem": 3}}}
    with pytest.raises(ValueError, match="no entry for m9 problem 3"):
        previous_choice_towers_for_problem(unknown_mouse, previous)
    unknown_problem = {"m1": {"ses-1_date-20260101": {**ports_session("C1"), "problem": 7}}}
    with pytest.raises(ValueError, match="no entry for m1 problem 7"):
        previous_choice_towers_for_problem(unknown_problem, previous)


def test_sessions_of_one_mouse_and_problem_must_share_one_problem_number():
    previous = previous_problem_choice_towers(three_problems())
    mixed = {
        "m1": {
            "ses-1_date-20260101": {**ports_session("C1"), "problem": 10},
            "ses-2_date-20260102": {**ports_session("C1"), "problem": 3},
        }
    }
    with pytest.raises(ValueError, match="more than one problem"):
        previous_choice_towers_for_problem(mixed, previous)
