"""The log of the Python warnings raised during a run, with mouse and session read from the text."""

import pandas as pd

from src.behavior_analysis.warnings_log import (
    COLUMNS,
    parse_warning_message,
    warnings_log_table,
)
from src.behavior_import.warning_records import WarningRecord
from tests.review_fixtures import cohort, session

S1, S2, S3 = "ses-1_date-20260101", "ses-2_date-20260102", "ses-3_date-20260103"
FLAGGED = (
    f"M1 {S2}: suspected settings error (layout lasts one session before changing again); "
    "excluded from problem numbering."
)


def record(message, category="UserWarning", filename="/src/x.py", lineno=7):
    return WarningRecord(category, message, filename, lineno)


def index_of_cohort():
    _, index, _, _ = cohort({1: {"M1": {S1: session(), S3: session()}, "M10": {S1: session()}}})
    return index


def test_mouse_session_and_problem_are_read_from_a_numbering_warning():
    parsed = parse_warning_message(
        f"M1 {S2}: review settings (...); kept as problem 3.", ["M1", "M10"]
    )
    assert parsed == {"mouse": "M1", "session": S2, "problem_in_message": 3}


def test_a_mouse_name_inside_another_name_is_not_a_match():
    assert parse_warning_message("M10 did something", ["M1"])["mouse"] is None
    assert parse_warning_message("subject 'M1': dropped 2 reversals", ["M1"])["mouse"] == "M1"


def test_two_different_mice_in_one_message_leave_the_mouse_blank():
    assert parse_warning_message("M1 and M10 disagree", ["M1", "M10"])["mouse"] is None


def test_the_same_mouse_named_twice_is_still_that_mouse():
    assert parse_warning_message("M1 ... M1", ["M1", "M10"])["mouse"] == "M1"


def test_a_part_session_key_is_read_whole():
    parsed = parse_warning_message(f"M1 {S1}_part-02: split", ["M1"])
    assert parsed["session"] == f"{S1}_part-02"


def test_a_message_without_names_has_blank_fields():
    parsed = parse_warning_message("fewer than two non-NaN values", ["M1"])
    assert parsed == {"mouse": None, "session": None, "problem_in_message": None}


def test_the_log_has_one_row_per_distinct_warning_with_its_place_in_the_problems():
    records = [
        record(FLAGGED, "ProblemNumberingWarning", "/src/problem_numbering.py", 430),
        record("fewer than two non-NaN values"),
    ]
    table = warnings_log_table(records, index_of_cohort())
    assert list(table.columns) == list(COLUMNS)
    flagged, plain = table.to_dict("records")
    assert flagged["category"] == "ProblemNumberingWarning"
    assert flagged["message"] == FLAGGED
    assert (flagged["mouse"], flagged["session"]) == ("M1", S2)
    assert flagged["source"] == "problem_numbering.py:430"
    # S2 is in no problem: placed after the problem the mouse was running
    assert (flagged["task_type"], flagged["overall_problem"]) == ("drifting", 1)
    assert pd.isna(plain["mouse"]) and pd.isna(plain["task_type"])


def test_a_session_in_a_problem_uses_its_own_problem():
    (row,) = warnings_log_table([record(f"M1 {S3}: odd")], index_of_cohort()).to_dict("records")
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("drifting", 1, 1)


def test_repeated_warnings_are_counted_once_per_distinct_text_in_first_seen_order():
    records = [record("b"), record("a"), record("b"), record("b", category="ReversalWarning")]
    table = warnings_log_table(records, index_of_cohort())
    assert [(r.category, r.message, r.n_times) for r in table.itertuples()] == [
        ("UserWarning", "b", 2),
        ("UserWarning", "a", 1),
        ("ReversalWarning", "b", 1),
    ]


def test_no_warnings_give_an_empty_table_with_the_columns():
    table = warnings_log_table([], index_of_cohort())
    assert table.empty and list(table.columns) == list(COLUMNS)
