"""The shared problem-numbering rule, on synthetic per-session facts."""

import warnings

import pytest

from src.behavior_import.problem_numbering import (
    SUSPECTED_SETTINGS_ERROR,
    SessionFacts,
    assign_problems,
    number_subject_sessions,
    tower_layout,
)

A = tower_layout(["A1", "B1", "C1"], "B2")
B = tower_layout(["A2", "B2", "C2"], "B2")
C = tower_layout(["A3", "B3", "C3"], "B2")


def clean(layout):
    """A changed session with full evidence: has num_t_found and starts with found_ports false."""
    return SessionFacts(layout, has_num_t_found=True, starts_found_ports_false=True)


def numbers(facts):
    return [a.problem for a in assign_problems(facts)]


def flags(facts):
    return [a.flag for a in assign_problems(facts)]


def test_layout_includes_the_initiation_tower_and_ignores_order_and_case():
    assert tower_layout(["C1", "a1", "B1"], "B2") == tower_layout(["A1", "B1", "C1"], "b2")
    assert tower_layout(["A1", "B1", "C1"], "B2") != tower_layout(["A1", "B1", "C1"], "A2")


def test_missing_initiation_tower_means_the_default_b2():
    assert tower_layout(["A1", "B1", "C1"], None) == tower_layout(["A1", "B1", "C1"], "B2")


def test_no_choice_towers_gives_no_layout():
    assert tower_layout([], "B2") is None


def test_same_layout_stays_one_problem():
    assert numbers([clean(A), clean(A), clean(A)]) == [1, 1, 1]


def test_change_of_choice_towers_starts_a_new_problem():
    assert numbers([clean(A), clean(B), clean(B)]) == [1, 2, 2]


def test_change_of_only_the_initiation_tower_starts_a_new_problem():
    other_initiation = tower_layout(["A1", "B1", "C1"], "A2")
    assert numbers([clean(A), clean(A), clean(other_initiation), clean(other_initiation)]) == [
        1,
        1,
        2,
        2,
    ]


def test_return_to_an_earlier_real_layout_is_a_new_problem():
    assert numbers([clean(A), clean(B), clean(B), clean(A)]) == [1, 2, 2, 3]
    assert flags([clean(A), clean(B), clean(B), clean(A)]) == [None] * 4


def test_a_layout_that_lasts_one_session_before_changing_again_is_flagged():
    facts = [clean(A), clean(B), clean(C), clean(C)]
    assert numbers(facts) == [1, None, 2, 2]
    assert flags(facts) == [None, SUSPECTED_SETTINGS_ERROR, None, None]


def test_flagged_session_between_equal_layouts_leaves_the_original_problem_pooled():
    facts = [clean(A), clean(A), clean(B), clean(A)]
    assert numbers(facts) == [1, 1, None, 1]


def test_a_single_session_at_the_end_is_not_flagged_for_lasting_one_session():
    assert numbers([clean(A), clean(B)]) == [1, 2]


def test_changed_session_without_num_t_found_is_flagged():
    no_num_t_found = SessionFacts(B, has_num_t_found=False, starts_found_ports_false=True)
    facts = [clean(A), no_num_t_found, clean(B)]
    assert numbers(facts) == [1, None, 2]


def test_changed_session_not_starting_with_found_ports_false_is_flagged():
    found_ports_true = SessionFacts(B, has_num_t_found=True, starts_found_ports_false=False)
    assert numbers([clean(A), found_ports_true, clean(B)]) == [1, None, 2]


def test_later_sessions_of_a_flagged_layout_are_judged_against_the_last_unflagged_layout():
    found_ports_true = SessionFacts(B, has_num_t_found=True, starts_found_ports_false=False)
    # B's first session is flagged; the following B sessions are not "changes", so they form
    # a new problem relative to A.
    assert numbers([clean(A), found_ports_true, clean(B), clean(B)]) == [1, None, 2, 2]


def test_evidence_is_not_checked_for_sessions_that_do_not_change_the_layout():
    unchecked = SessionFacts(A, has_num_t_found=False, starts_found_ports_false=False)
    assert numbers([clean(A), unchecked, unchecked]) == [1, 1, 1]


def test_first_session_is_never_flagged_whatever_its_evidence():
    unchecked = SessionFacts(A, has_num_t_found=False, starts_found_ports_false=False)
    assert numbers([unchecked, clean(A)]) == [1, 1]


def test_sessions_without_choice_towers_keep_the_current_problem_and_are_not_flagged():
    none = SessionFacts(None, has_num_t_found=False, starts_found_ports_false=False)
    assert numbers([clean(A), none, clean(A)]) == [1, 1, 1]
    assert numbers([clean(A), clean(B), none, clean(B)]) == [1, 2, 2, 2]
    assert numbers([none, clean(A)]) == [1, 1]
    assert flags([clean(A), none, clean(A)]) == [None] * 3


def test_a_session_without_towers_does_not_hide_a_layout_change_from_the_flag_checks():
    none = SessionFacts(None, has_num_t_found=True, starts_found_ports_false=True)
    # B lasts one session (the None session in between is ignored), then A returns.
    assert numbers([clean(A), clean(B), none, clean(A)]) == [1, None, 1, 1]


def test_no_sessions():
    assert assign_problems([]) == []


def test_number_subject_sessions_warns_naming_subject_and_session():
    facts = [clean(A), clean(B), clean(A)]
    with pytest.warns(UserWarning, match=r"M1.*ses-2.*suspected settings error"):
        out = number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)
    assert [a.problem for a in out] == [1, None, 1]


def test_number_subject_sessions_is_silent_without_flags():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        number_subject_sessions("M1", ["ses-1", "ses-2"], [clean(A), clean(B)])
