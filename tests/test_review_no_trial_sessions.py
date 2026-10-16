"""The table of skipped files, recordings and sessions, and the problem context of its rows."""

import pandas as pd

from src.behavior_analysis.no_trial_sessions import COLUMNS, no_trial_sessions_table
from src.behavior_import import import_log
from src.behavior_import.import_log import SkipRecord
from tests.review_fixtures import BLOCKED_NAME, DRIFTING_NAME, cohort, session

S1, S2, S3, S4 = (f"ses-{n}_date-2026010{n}" for n in (1, 2, 3, 4))


def two_problem_cohort():
    """M1: blocked problem 1 (ses 1, 2), drifting problem 2 (ses 4)."""
    return cohort(
        {
            1: {"M1": {S1: session(task_name=BLOCKED_NAME), S2: session(task_name=BLOCKED_NAME)}},
            2: {"M1": {S4: session(task_name=DRIFTING_NAME)}},
        }
    )


def skip(session_key, **kw):
    fields = {
        "mouse": "M1",
        "session": session_key,
        "file": f"/raw/M1/{session_key}/behav/M1-x.tsv",
        "level": import_log.SESSION,
        "kind": import_log.NO_TRIAL_PRINTS,
        "reason": "it prints no trials",
        "task_name": None,
    }
    return SkipRecord(**{**fields, **kw})


def test_a_skipped_session_sits_in_the_problem_the_mouse_was_running():
    _, index, _, _ = two_problem_cohort()
    table = no_trial_sessions_table([skip(S3, task_name=DRIFTING_NAME)], index)
    (row,) = table.to_dict("records")
    assert row["mouse"] == "M1"
    assert row["session"] == S3
    assert row["file"] == "M1-x.tsv"
    assert row["skipped"] == "session"
    assert row["kind"] == import_log.NO_TRIAL_PRINTS
    assert row["reason"] == "it prints no trials"
    # own task type from its name; the problem is the one before it (blocked 1), whose number
    # within its type does not carry over to a drifting session
    assert (row["task_type"], row["overall_problem"]) == ("drifting", 1)
    assert pd.isna(row["problem_in_type"])


def test_a_skipped_session_of_the_same_type_gets_the_number_within_the_type():
    _, index, _, _ = two_problem_cohort()
    table = no_trial_sessions_table([skip(S3, task_name=BLOCKED_NAME)], index)
    (row,) = table.to_dict("records")
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("blocked", 1, 1)


def test_a_session_in_a_problem_with_a_skipped_recording_uses_its_own_problem():
    _, index, _, _ = two_problem_cohort()
    table = no_trial_sessions_table(
        [skip(S4, level=import_log.RECORDING, kind=import_log.NO_BANDIT_FIELDS)], index
    )
    (row,) = table.to_dict("records")
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("drifting", 1, 2)
    assert row["skipped"] == "recording"


def test_files_without_a_mouse_or_session_still_get_a_row_with_blanks():
    _, index, _, _ = two_problem_cohort()
    lost = SkipRecord(None, None, "/raw/odd/x.tsv", "file", import_log.NO_SUBJECT, "no subject")
    (row,) = no_trial_sessions_table([lost], index).to_dict("records")
    assert pd.isna(row["mouse"]) and pd.isna(row["session"])
    assert row["file"] == "x.tsv"
    assert row["skipped"] == "file"


def test_rows_are_ordered_by_mouse_then_session_and_the_columns_are_fixed():
    _, index, _, _ = two_problem_cohort()
    skips = [skip(S3), skip(S2, level=import_log.RECORDING), skip(S3, mouse="M0")]
    table = no_trial_sessions_table(skips, index)
    assert list(table.columns) == list(COLUMNS)
    assert list(zip(table["mouse"], table["session"], strict=True)) == [
        ("M0", S3),
        ("M1", S2),
        ("M1", S3),
    ]


def test_no_skips_give_an_empty_table_with_the_columns():
    _, index, _, _ = two_problem_cohort()
    table = no_trial_sessions_table([], index)
    assert table.empty
    assert list(table.columns) == list(COLUMNS)
