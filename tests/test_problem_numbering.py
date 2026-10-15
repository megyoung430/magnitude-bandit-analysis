"""The shared problem-numbering rule, on synthetic per-session facts."""

import warnings

import pytest

from src.behavior_import.problem_numbering import (
    NO_CHOICE_TOWERS,
    PORT_FINDING,
    PRETRAINING,
    REVIEW_SETTINGS,
    SUSPECTED_SETTINGS_ERROR,
    ProblemNumberingWarning,
    SessionFacts,
    assign_problems,
    is_pretraining,
    number_subject_sessions,
    session_sort_key,
    tower_layout,
)

A = tower_layout(["A1", "B1", "C1"], "B2")
B = tower_layout(["A2", "B2", "C2"], "B2")
C = tower_layout(["A3", "B3", "C3"], "B2")
D = tower_layout(["A4", "B4", "C4"], "B2")


def clean(layout):
    """The start of a problem: found_ports false, and port-finding trials were printed."""
    return SessionFacts(layout, starts_found_ports_false=True, has_port_finding_trials=True)


def resumed(layout):
    """A later session of an ongoing problem: found_ports already true, no port finding."""
    return SessionFacts(layout, starts_found_ports_false=False, has_port_finding_trials=False)


def numbers(facts):
    return [a.problem for a in assign_problems(facts)]


def flags(facts):
    return [a.flag for a in assign_problems(facts)]


# --- layouts ---


def test_layout_includes_the_initiation_tower_and_ignores_order_and_case():
    assert tower_layout(["C1", "a1", "B1"], "B2") == tower_layout(["A1", "B1", "C1"], "b2")
    assert tower_layout(["A1", "B1", "C1"], "B2") != tower_layout(["A1", "B1", "C1"], "A2")


def test_missing_initiation_tower_means_the_default_b2():
    assert tower_layout(["A1", "B1", "C1"], None) == tower_layout(["A1", "B1", "C1"], "B2")


def test_no_choice_towers_gives_no_layout():
    assert tower_layout([], "B2") is None


# --- numbering ---


def test_same_layout_stays_one_problem():
    assert numbers([clean(A), resumed(A), resumed(A)]) == [1, 1, 1]


def test_change_of_choice_towers_starts_a_new_problem():
    assert numbers([clean(A), clean(B), resumed(B)]) == [1, 2, 2]


def test_change_of_only_the_initiation_tower_starts_a_new_problem():
    other = tower_layout(["A1", "B1", "C1"], "A2")
    assert numbers([clean(A), resumed(A), clean(other), resumed(other)]) == [1, 1, 2, 2]


def test_return_to_an_earlier_layout_after_a_real_problem_is_a_new_problem():
    facts = [clean(A), clean(B), resumed(B), clean(A)]
    assert numbers(facts) == [1, 2, 2, 3]
    assert flags(facts) == [None] * 4


def test_no_sessions():
    assert assign_problems([]) == []


# --- single-session layouts are settings errors ---


def test_new_layout_lasting_one_session_is_a_settings_error():
    facts = [clean(A), clean(B), clean(C), resumed(C)]
    assert numbers(facts) == [1, None, 2, 2]
    assert flags(facts) == [None, SUSPECTED_SETTINGS_ERROR, None, None]


def test_resumed_problem_after_a_settings_error_continues_whatever_its_evidence():
    facts = [clean(A), clean(B), resumed(A), resumed(A)]
    assert numbers(facts) == [1, None, 1, 1]
    assert flags(facts) == [None, SUSPECTED_SETTINGS_ERROR, None, None]


def test_resumed_problem_after_a_settings_error_at_the_end_of_the_data():
    assert numbers([clean(A), clean(B), resumed(A)]) == [1, None, 1]


def test_consecutive_one_session_layouts_are_all_settings_errors():
    assert numbers([clean(A), clean(B), clean(C), resumed(A)]) == [1, None, None, 1]
    facts = [clean(A), clean(B), clean(C), clean(D), resumed(A), resumed(A)]
    assert numbers(facts) == [1, None, None, None, 1, 1]
    assert flags(facts)[1:4] == [SUSPECTED_SETTINGS_ERROR] * 3


def test_a_layout_is_compared_with_the_last_unflagged_layout_not_the_previous_session():
    # The flagged B must not make the resumed A look like a new layout needing evidence.
    assert flags([clean(A), clean(B), resumed(A)]) == [None, SUSPECTED_SETTINGS_ERROR, None]


def test_a_final_one_session_layout_is_a_real_problem():
    # Its duration is not known yet, so it is not a settings error.
    assert numbers([clean(A), clean(B)]) == [1, 2]


def test_a_one_session_first_layout_is_problem_1():
    assert numbers([clean(A), clean(B), resumed(B)]) == [1, 2, 2]
    assert numbers([resumed(A), clean(B), resumed(B)]) == [1, 2, 2]


# --- multi-session layouts failing the evidence checks are kept for review ---


def test_multi_session_layout_failing_found_ports_keeps_its_number_with_a_review_flag():
    facts = [clean(A), resumed(B), resumed(B), resumed(B)]
    assert numbers(facts) == [1, 2, 2, 2]
    assert flags(facts) == [None, REVIEW_SETTINGS, None, None]
    assert assign_problems(facts)[1].reasons == ("does not start with found_ports false",)


def test_found_ports_false_but_no_printed_port_finding_needs_review():
    no_trials = SessionFacts(B, starts_found_ports_false=True, has_port_finding_trials=False)
    facts = [clean(A), no_trials, resumed(B)]
    assert numbers(facts) == [1, 2, 2]
    assert flags(facts) == [None, REVIEW_SETTINGS, None]
    assert "num_t_found" in assign_problems(facts)[1].reasons[0]


def test_port_finding_check_is_not_applied_when_found_ports_already_failed():
    reasons = assign_problems([clean(A), resumed(B), resumed(B)])[1].reasons
    assert reasons == ("does not start with found_ports false",)


def test_review_flag_also_applies_to_a_returning_layout_and_a_final_single_session():
    assert flags([clean(A), clean(B), resumed(B), resumed(A)]) == [
        None,
        None,
        None,
        REVIEW_SETTINGS,
    ]
    assert numbers([clean(A), resumed(B)]) == [1, 2]
    assert flags([clean(A), resumed(B)]) == [None, REVIEW_SETTINGS]


def test_evidence_is_not_checked_for_the_first_run_or_continuing_sessions():
    assert flags([resumed(A), resumed(A), clean(B), resumed(B), resumed(B)]) == [None] * 5


# --- sessions that are not bandit problems ---


def pretraining(task_name="mag"):
    return SessionFacts(None, True, True, task_name=task_name)


def port_finding():
    return SessionFacts(None, True, True)


def no_towers():
    return SessionFacts(None, False, False)


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("mag", True),
        ("Meg\\MAG", True),
        ("Meg/initiation_pretraining", True),
        ("Meg\\3x3_field_pretraining_v2", True),
        ("Meg\\3x3_field_drifting_value_bandit", False),
        ("Meg\\3x3_field_magnitude_bandit", False),
        ("", False),
        (None, False),
    ],
)
def test_is_pretraining_matches_the_task_name_not_substrings_of_mag(task_name, expected):
    assert is_pretraining(task_name) is expected


def test_pretraining_is_skipped_silently_and_marked():
    facts = [clean(A), pretraining(), resumed(A)]
    assert numbers(facts) == [1, None, 1]
    assert flags(facts) == [None, PRETRAINING, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)


def test_pretraining_by_task_name_wins_even_with_choice_towers():
    with_towers = SessionFacts(B, True, True, task_name="initiation_pretraining")
    assert numbers([clean(A), with_towers, resumed(A)]) == [1, None, 1]
    assert numbers([with_towers, clean(A)]) == [None, 1]


def test_port_finding_is_skipped_marked_and_warned():
    facts = [clean(A), port_finding(), resumed(A)]
    assert numbers(facts) == [1, None, 1]
    assert flags(facts) == [None, PORT_FINDING, None]
    with pytest.warns(ProblemNumberingWarning, match=r"M1 ses-2.*port finding"):
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)


def test_towerless_session_that_is_neither_is_flagged_and_warned():
    facts = [clean(A), no_towers(), resumed(A)]
    assert numbers(facts) == [1, None, 1]
    assert flags(facts) == [None, NO_CHOICE_TOWERS, None]
    with pytest.warns(ProblemNumberingWarning, match=r"M1 ses-2.*no choice towers"):
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)


def test_skipped_sessions_before_the_first_layout_do_not_take_a_number():
    facts = [pretraining(), port_finding(), clean(A), clean(B), resumed(B)]
    assert numbers(facts) == [None, None, 1, 2, 2]


def test_skipped_sessions_are_ignored_when_comparing_the_sessions_around_them():
    assert numbers([clean(A), pretraining(), clean(B), resumed(B)]) == [1, None, 2, 2]
    # B lasts one session: the skipped session does not make it two.
    assert numbers([clean(A), clean(B), port_finding(), resumed(A)]) == [1, None, None, 1]


# --- warnings ---


def test_settings_error_warns_naming_subject_and_session():
    facts = [clean(A), clean(B), resumed(A)]
    with pytest.warns(ProblemNumberingWarning, match=r"M1.*ses-2.*suspected settings error"):
        out = number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)
    assert [a.problem for a in out] == [1, None, 1]


def test_review_warning_says_the_session_keeps_its_number():
    facts = [clean(A), resumed(B), resumed(B)]
    with pytest.warns(ProblemNumberingWarning, match=r"M1 ses-2.*review settings.*problem 2"):
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)


def test_warnings_point_at_the_caller():
    with pytest.warns(ProblemNumberingWarning) as record:
        number_subject_sessions("M1", ["s1", "s2", "s3"], [clean(A), clean(B), resumed(A)])
    assert record[0].filename == __file__


def test_warn_false_is_silent_and_gives_the_same_numbers():
    facts = [clean(A), clean(B), resumed(A)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = number_subject_sessions("M1", ["s1", "s2", "s3"], facts, warn=False)
    assert out == assign_problems(facts)


def test_no_warning_without_flags():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        number_subject_sessions("M1", ["ses-1", "ses-2"], [clean(A), clean(B)])


# --- session order ---


def test_session_sort_key_orders_by_date_then_session_number():
    keys = ["ses-9_date-20260101", "ses-2_date-20260102", "ses-1_date-20260102"]
    assert sorted(keys, key=session_sort_key) == [
        "ses-9_date-20260101",
        "ses-1_date-20260102",
        "ses-2_date-20260102",
    ]


def test_session_sort_key_puts_keys_without_date_or_number_last_and_keeps_segment_order():
    keys = ["other", "ses-2_date-20260101_part-02", "ses-2_date-20260101_part-01", "ses-1"]
    assert sorted(keys, key=session_sort_key) == [
        "ses-2_date-20260101_part-01",
        "ses-2_date-20260101_part-02",
        "ses-1",
        "other",
    ]
