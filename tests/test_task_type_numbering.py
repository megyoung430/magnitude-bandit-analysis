"""A change of task type starts a new problem, like a change of tower layout."""

import warnings

import pytest

from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.problem_numbering import (
    SUSPECTED_SETTINGS_ERROR,
    ProblemNumberingWarning,
    SessionFacts,
    assign_problems,
    tower_layout,
)
from tests.test_import_data import A, flags_of, import_sessions, problems

LAYOUT = tower_layout(["A1", "B1", "C1"], "B2")
BLOCKED_TASK = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_TASK = "Meg\\3x3_field_drifting_value_bandit"
FIRST_SESSION = "ses-1_date-20260101"
THIRD_SESSION = "ses-3_date-20260103"


def facts(task_type, *, start):
    return SessionFacts(
        LAYOUT,
        starts_found_ports_false=start,
        has_port_finding_trials=start,
        task_type=task_type,
    )


def numbers_and_flags(task_types):
    """Numbering of sessions of one layout; a session starts clean when its type changes."""
    sessions = [
        facts(kind, start=n == 0 or task_types[n - 1] != kind) for n, kind in enumerate(task_types)
    ]
    assignments = assign_problems(sessions)
    return [a.problem for a in assignments], [a.flag for a in assignments]


# --- the rule on synthetic facts ---


def test_a_change_of_task_type_with_the_same_layout_starts_a_new_problem():
    assert numbers_and_flags(["blocked", "blocked", "drifting", "drifting"]) == (
        [1, 1, 2, 2],
        [None] * 4,
    )


def test_one_session_of_another_task_type_is_a_settings_error_pooled_around():
    kinds = ["drifting", "drifting", "blocked", "drifting", "drifting"]
    numbers, flags = numbers_and_flags(kinds)
    assert numbers == [1, 1, None, 1, 1]
    assert flags[2] == SUSPECTED_SETTINGS_ERROR


def test_sessions_without_a_task_type_number_by_layout_alone():
    assert numbers_and_flags([None, None, None]) == ([1, 1, 1], [None] * 3)


# --- both entry points ---


def items(task_names):
    """Sessions of one layout: a clean start whenever the task name changes."""
    entries = []
    for n, name in enumerate(task_names):
        starts = n == 0 or task_names[n - 1] != name
        entries.append((A if starts else {**A, "found_ports": True}, name, starts))
    return entries


def grouped_numbers(sessions):
    grouped = extract_trials_grouped_by_problem({"M1": sessions})
    return {key: problem for problem, subjects in grouped.items() for key in subjects["M1"]}


def test_import_and_grouping_agree_on_a_task_type_change_with_an_unchanged_layout(tmp_path):
    names = [BLOCKED_TASK, BLOCKED_TASK, DRIFTING_TASK, DRIFTING_TASK]
    sessions = import_sessions(tmp_path, items(names), classify_task_types=True)
    assert problems(sessions) == [1, 1, 2, 2]
    assert [s["task_type"] for s in sessions.values()] == ["blocked"] * 2 + ["drifting"] * 2
    assert grouped_numbers(sessions) == dict(zip(sessions, [1, 1, 2, 2], strict=True))


def test_import_and_grouping_agree_on_a_one_session_task_type_error(tmp_path):
    names = [DRIFTING_TASK, DRIFTING_TASK, BLOCKED_TASK, DRIFTING_TASK, DRIFTING_TASK]
    with pytest.warns(ProblemNumberingWarning, match=rf"{THIRD_SESSION}.*settings error"):
        sessions = import_sessions(tmp_path, items(names), classify_task_types=True)
    assert problems(sessions) == [1, 1, None, 1, 1]
    assert flags_of(sessions)[2] == SUSPECTED_SETTINGS_ERROR
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # import_data already warned about this flag
        numbers = grouped_numbers(sessions)
    assert THIRD_SESSION not in numbers
    assert set(numbers.values()) == {1}


def test_task_type_is_ignored_unless_asked_for(tmp_path):
    names = [BLOCKED_TASK, BLOCKED_TASK, DRIFTING_TASK, DRIFTING_TASK]
    sessions = import_sessions(tmp_path, items(names))
    assert problems(sessions) == [1, 1, 1, 1]
    assert all(s["task_type"] is None for s in sessions.values())


def test_pretraining_is_skipped_before_the_task_type_is_read(tmp_path):
    resumed = {**A, "found_ports": True}
    entries = [
        (A, BLOCKED_TASK, True),
        (resumed, BLOCKED_TASK, False),
        (A, "initiation_pretraining", True),
        (resumed, BLOCKED_TASK, False),
    ]
    sessions = import_sessions(tmp_path, entries, classify_task_types=True)
    assert problems(sessions) == [1, 1, None, 1]
    assert flags_of(sessions)[2] == "pretraining"
    assert sessions["ses-3_date-20260103"]["task_type"] is None


@pytest.mark.parametrize("task_name", ["Meg\\mystery_task", "blocked_drifting"])
def test_an_unrecognised_task_name_raises_before_numbering_naming_the_file(tmp_path, task_name):
    entries = [(A, BLOCKED_TASK, True), (A, task_name, True)]
    with pytest.raises(ValueError, match=r"M1 ses-2_date-20260102.*M1-2026-01-02-100000\.tsv"):
        import_sessions(tmp_path, entries, classify_task_types=True)


def test_a_missing_task_name_stops_the_run_and_names_the_file(tmp_path):
    entries = [(A, BLOCKED_TASK, True), (A, None, True)]
    with pytest.raises(ValueError) as err:
        import_sessions(tmp_path, entries, classify_task_types=True)
    message = str(err.value)
    assert "M1 ses-2_date-20260102" in message and "M1-2026-01-02-100000.tsv" in message
    assert "no task name" in message and "stops" in message
