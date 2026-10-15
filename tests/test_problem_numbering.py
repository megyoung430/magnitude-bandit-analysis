"""The shared problem-numbering rule, on synthetic per-session facts."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

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

# Two sessions of A: a first layout needs two sessions to be a real problem.
START = [clean(A), resumed(A)]


def test_same_layout_stays_one_problem():
    assert numbers([clean(A), resumed(A), resumed(A)]) == [1, 1, 1]


def test_change_of_choice_towers_starts_a_new_problem():
    assert numbers(START + [clean(B), resumed(B)]) == [1, 1, 2, 2]


def test_change_of_only_the_initiation_tower_starts_a_new_problem():
    other = tower_layout(["A1", "B1", "C1"], "A2")
    assert numbers([clean(A), resumed(A), clean(other), resumed(other)]) == [1, 1, 2, 2]


def test_return_to_an_earlier_layout_after_a_real_problem_is_a_new_problem():
    facts = START + [clean(B), resumed(B), clean(A)]
    assert numbers(facts) == [1, 1, 2, 2, 3]
    assert flags(facts) == [None] * 5


def test_no_sessions():
    assert assign_problems([]) == []


# --- single-session layouts are settings errors ---


def test_new_layout_lasting_one_session_is_a_settings_error():
    facts = START + [clean(B), clean(C), resumed(C)]
    assert numbers(facts) == [1, 1, None, 2, 2]
    assert flags(facts) == [None, None, SUSPECTED_SETTINGS_ERROR, None, None]


def test_resumed_problem_after_a_settings_error_continues_whatever_its_evidence():
    facts = START + [clean(B), resumed(A), resumed(A)]
    assert numbers(facts) == [1, 1, None, 1, 1]
    assert flags(facts) == [None, None, SUSPECTED_SETTINGS_ERROR, None, None]


def test_resumed_problem_after_a_settings_error_at_the_end_of_the_data():
    assert numbers(START + [clean(B), resumed(A)]) == [1, 1, None, 1]


def test_consecutive_one_session_layouts_are_all_settings_errors():
    assert numbers(START + [clean(B), clean(C), resumed(A)]) == [1, 1, None, None, 1]
    facts = START + [clean(B), clean(C), clean(D), resumed(A), resumed(A)]
    assert numbers(facts) == [1, 1, None, None, None, 1, 1]
    assert flags(facts)[2:5] == [SUSPECTED_SETTINGS_ERROR] * 3


def test_a_layout_is_compared_with_the_last_unflagged_layout_not_the_previous_session():
    # The flagged B must not make the resumed A look like a new layout needing evidence.
    facts = START + [clean(B), resumed(A)]
    assert flags(facts) == [None, None, SUSPECTED_SETTINGS_ERROR, None]


def test_a_final_one_session_layout_is_a_real_but_provisional_problem():
    # Its duration is not known yet, so it is not a settings error.
    out = assign_problems(START + [clean(B)])
    assert [a.problem for a in out] == [1, 1, 2]
    assert [a.provisional for a in out] == [False, False, True]


def test_a_final_run_of_two_sessions_is_not_provisional():
    assert not any(a.provisional for a in assign_problems(START + [clean(B), resumed(B)]))


# --- the first run ---


def test_a_first_layout_lasting_one_session_is_a_settings_error():
    facts = [clean(B), clean(A), resumed(A)]
    assert numbers(facts) == [None, 1, 1]
    assert flags(facts) == [SUSPECTED_SETTINGS_ERROR, None, None]


def test_consecutive_one_session_layouts_at_the_start_are_all_flagged():
    facts = [clean(B), clean(C), clean(A), resumed(A)]
    assert numbers(facts) == [None, None, 1, 1]


def test_the_only_layout_in_the_data_is_problem_1_even_for_one_session():
    out = assign_problems([clean(A)])
    assert [a.problem for a in out] == [1]
    assert out[0].flag is None
    assert out[0].provisional


def test_a_first_layout_of_one_session_followed_by_a_last_layout_is_flagged():
    assert numbers([clean(B), clean(A)]) == [None, 1]


def test_skipped_sessions_do_not_stop_a_layout_being_the_only_one():
    facts = [clean(A), SessionFacts(None, True, True, task_name="mag")]
    assert numbers(facts) == [1, None]


def test_the_first_problem_is_not_evidence_checked():
    assert flags([resumed(A), resumed(A), clean(B), resumed(B), resumed(B)]) == [None] * 5
    # After a flagged first run, the first real problem is still the first problem.
    assert flags([clean(B), resumed(A), resumed(A)]) == [SUSPECTED_SETTINGS_ERROR, None, None]


# --- multi-session layouts failing the evidence checks are kept for review ---


def test_multi_session_layout_failing_found_ports_keeps_its_number_with_a_review_flag():
    facts = START + [resumed(B), resumed(B), resumed(B)]
    assert numbers(facts) == [1, 1, 2, 2, 2]
    assert flags(facts) == [None, None, REVIEW_SETTINGS, None, None]
    assert assign_problems(facts)[2].reasons == ("does not start with found_ports false",)


def test_found_ports_false_but_no_printed_port_finding_needs_review():
    no_trials = SessionFacts(B, starts_found_ports_false=True, has_port_finding_trials=False)
    facts = START + [no_trials, resumed(B)]
    assert numbers(facts) == [1, 1, 2, 2]
    assert flags(facts) == [None, None, REVIEW_SETTINGS, None]
    assert "num_t_found" in assign_problems(facts)[2].reasons[0]


def test_port_finding_check_is_not_applied_when_found_ports_already_failed():
    reasons = assign_problems(START + [resumed(B), resumed(B)])[2].reasons
    assert reasons == ("does not start with found_ports false",)


def test_review_flag_also_applies_to_a_returning_layout_and_a_final_single_session():
    facts = START + [clean(B), resumed(B), resumed(A)]
    assert flags(facts) == [None, None, None, None, REVIEW_SETTINGS]
    assert flags(START + [resumed(B)]) == [None, None, REVIEW_SETTINGS]


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
    facts = [pretraining(), port_finding(), clean(A), resumed(A), clean(B), resumed(B)]
    assert numbers(facts) == [None, None, 1, 1, 2, 2]


def test_skipped_sessions_are_ignored_when_comparing_the_sessions_around_them():
    assert numbers(START + [pretraining(), clean(B), resumed(B)]) == [1, 1, None, 2, 2]
    # B lasts one session: the skipped session does not make it two.
    facts = START + [clean(B), port_finding(), resumed(A)]
    assert numbers(facts) == [1, 1, None, None, 1]


# --- warnings ---


def test_settings_error_warns_naming_subject_and_session():
    facts = START + [clean(B), resumed(A)]
    with pytest.warns(ProblemNumberingWarning, match=r"M1.*ses-3.*suspected settings error"):
        out = number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3", "ses-4"], facts)
    assert [a.problem for a in out] == [1, 1, None, 1]


def test_review_warning_says_the_session_keeps_its_number():
    facts = START + [resumed(B), resumed(B)]
    with pytest.warns(ProblemNumberingWarning, match=r"M1 ses-3.*review settings.*problem 2"):
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3", "ses-4"], facts)


def test_a_provisional_review_warning_says_a_later_session_can_change_it():
    facts = START + [resumed(B)]
    with pytest.warns(ProblemNumberingWarning, match=r"provisional.*later session"):
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], facts)


def test_warnings_point_at_the_caller():
    with pytest.warns(ProblemNumberingWarning) as record:
        number_subject_sessions("M1", ["s1", "s2", "s3", "s4"], START + [clean(B), resumed(A)])
    assert record[0].filename == __file__
    assert len(record) == 1


def test_warn_false_is_silent_and_gives_the_same_numbers():
    facts = [clean(A), clean(B), resumed(A)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = number_subject_sessions("M1", ["s1", "s2", "s3"], facts, warn=False)
    assert out == assign_problems(facts)


def test_no_warning_without_flags():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        number_subject_sessions("M1", ["ses-1", "ses-2", "ses-3"], START + [clean(B)])


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


def test_a_rerun_warns_again_from_the_same_line():
    # Python's default filter shows a warning once per source line, but a notebook cell
    # rerun must show it again. Run in a fresh interpreter so no pytest filter is in play.
    root = str(Path(__file__).resolve().parents[1])
    script = "\n".join(
        [
            "import sys, warnings",
            f"sys.path.insert(0, {root!r})",
            "from src.behavior_import.problem_numbering import SessionFacts, number_subject_sessions",
            "def facts(layout):",
            "    return SessionFacts(layout, True, True)",
            "rows = [facts(('A',)), facts(('A',)), facts(('B',)), facts(('A',))]",
            "with warnings.catch_warnings(record=True) as record:",
            "    for _ in range(2):",
            "        number_subject_sessions('M1', ['s1', 's2', 's3', 's4'], rows)",
            "print(len(record))",
        ]
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "2"
