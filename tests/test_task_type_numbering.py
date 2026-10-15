"""A change of task type starts a new problem, like a change of tower layout."""

import warnings

import pytest

from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.behavior_import.problem_numbering import (
    SUSPECTED_SETTINGS_ERROR,
    ProblemNumberingWarning,
    SessionFacts,
    assign_problems,
    tower_layout,
)
from tests.test_import_data import (
    A,
    flags_of,
    import_sessions,
    problems,
    run_start_for,
    session_dir,
    write_tsv,
)

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


# --- sessions split into segments ---

B_LAYOUT = run_start_for(["A2", "B2", "C2"])
RESUMED_A = {**A, "found_ports": True}
RESUMED_B = {**B_LAYOUT, "found_ports": True}


def write_session(root, number, recordings):
    """One session folder with a recording per ``(run_start, task_name, port_finding)``."""
    folder = session_dir(root, session=f"ses-{number}_date-202601{number:02d}")
    for n, (run_start, task_name, port_finding) in enumerate(recordings):
        write_tsv(
            folder,
            f"M1-2026-01-{number:02d}-{10 + n}0000.tsv",
            run_start=run_start,
            task_name=task_name,
            port_finding=port_finding,
        )


def segment_numbers(root):
    imported = import_data(root, classify_task_types=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProblemNumberingWarning)
        grouped = extract_trials_grouped_by_problem(imported)
    return {key: problem for problem, subjects in grouped.items() for key in subjects["M1"]}


def test_a_segment_is_numbered_by_its_own_task_type_on_a_switch_day(tmp_path):
    write_session(tmp_path, 1, [(A, BLOCKED_TASK, True)])
    write_session(tmp_path, 2, [(RESUMED_A, BLOCKED_TASK, False)])
    # blocked on layout A, then a drifting recording on layout B, in one session
    write_session(tmp_path, 3, [(RESUMED_A, BLOCKED_TASK, False), (B_LAYOUT, DRIFTING_TASK, True)])
    write_session(tmp_path, 4, [(RESUMED_B, DRIFTING_TASK, False)])
    assert segment_numbers(tmp_path) == {
        "ses-1_date-20260101": 1,
        "ses-2_date-20260102": 1,
        "ses-3_date-20260103_part-01": 1,
        "ses-3_date-20260103_part-02": 2,
        "ses-4_date-20260104": 2,
    }


def test_a_segment_after_a_pretraining_first_recording_is_classified_itself(tmp_path):
    write_session(tmp_path, 1, [(A, BLOCKED_TASK, True)])
    write_session(tmp_path, 2, [(RESUMED_A, BLOCKED_TASK, False)])
    write_session(
        tmp_path, 3, [(A, "initiation_pretraining", True), (B_LAYOUT, BLOCKED_TASK, True)]
    )
    write_session(tmp_path, 4, [(RESUMED_B, BLOCKED_TASK, False)])
    numbers = segment_numbers(tmp_path)
    assert "ses-3_date-20260103_part-01" not in numbers  # the pretraining recording is skipped
    assert numbers["ses-3_date-20260103_part-02"] == numbers["ses-4_date-20260104"] == 2


def test_segments_carry_no_task_type_when_task_types_are_off(tmp_path):
    write_session(tmp_path, 1, [(A, BLOCKED_TASK, True)])
    write_session(tmp_path, 2, [(RESUMED_A, BLOCKED_TASK, False)])
    write_session(tmp_path, 3, [(RESUMED_A, BLOCKED_TASK, False), (B_LAYOUT, DRIFTING_TASK, True)])
    write_session(tmp_path, 4, [(RESUMED_B, DRIFTING_TASK, False)])
    grouped = extract_trials_grouped_by_problem(import_data(tmp_path))
    segments = [s for subjects in grouped.values() for s in subjects["M1"].values()]
    assert len(segments) == 5 and {s["task_type"] for s in segments} == {None}


def test_the_settings_error_warning_names_only_the_layout_without_task_types(tmp_path):
    entries = [(A, BLOCKED_TASK, True), (A, BLOCKED_TASK, True), (B_LAYOUT, BLOCKED_TASK, True)]
    entries += [(A, BLOCKED_TASK, True)]
    with pytest.warns(ProblemNumberingWarning) as caught:
        import_sessions(tmp_path, entries)
    text = " ".join(str(w.message) for w in caught)
    assert "layout lasts one session" in text and "task type" not in text


def test_the_settings_error_warning_names_the_task_type_when_classified(tmp_path):
    names = [DRIFTING_TASK, DRIFTING_TASK, BLOCKED_TASK, DRIFTING_TASK, DRIFTING_TASK]
    with pytest.warns(ProblemNumberingWarning, match="layout or task type lasts one session"):
        import_sessions(tmp_path, items(names), classify_task_types=True)
