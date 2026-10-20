"""Sessions per drifting problem: the table and the summary behind the figure (pure, no files).

Sessions are minimal dicts (task name, choice towers, initiation tower), which is all the
numbering and the port-finding assignment read.
"""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.port_finding_sessions import assign_port_finding_sessions
from src.behavior_analysis.sessions_per_problem import (
    TABLE_COLUMNS,
    session_without_part,
    sessions_per_problem_table,
    summarize_sessions_per_problem,
    with_cohort,
)

BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"
A = ["A1", "B1", "C1"]
B = ["A3", "B3", "C3"]


def key(number):
    return f"ses-{number}_date-202601{number:02d}"


def session(towers, task=DRIFTING_NAME, provisional=False):
    return {
        "task_name": task,
        "choice_towers": [sorted(towers)] * 3,
        "initiation_tower": "B2",
        "problem_provisional": provisional,
    }


def run_of(numbers, towers=A, task=DRIFTING_NAME, provisional=False):
    return {key(n): session(towers, task, provisional) for n in numbers}


def row_of(table, mouse, k):
    (row,) = table[(table["mouse"] == mouse) & (table["problem_in_type"] == k)].to_dict("records")
    return row


# --- session_without_part ---------------------------------------------------------------


def test_a_part_suffix_is_removed_and_other_keys_are_unchanged():
    assert session_without_part("ses-3_date-20260103_part-02") == "ses-3_date-20260103"
    assert session_without_part("ses-3_date-20260103") == "ses-3_date-20260103"


# --- counts -----------------------------------------------------------------------------


def test_each_mouse_and_drifting_problem_gets_its_number_of_sessions():
    trials = {
        1: {"m1": run_of([1, 2], task=BLOCKED_NAME), "m2": run_of([1], task=BLOCKED_NAME)},
        2: {"m1": run_of([3, 4, 5]), "m2": run_of([2, 3])},
        3: {"m1": run_of([6]), "m2": run_of([4, 5, 6, 7])},
    }
    table = sessions_per_problem_table(trials)
    assert list(table.columns) == list(TABLE_COLUMNS)
    counts = {(r.mouse, r.problem_in_type): r.n_sessions for r in table.itertuples()}
    assert counts == {("m1", 1): 3, ("m1", 2): 1, ("m2", 1): 2, ("m2", 2): 4}
    assert set(table["task_type"]) == {"drifting"}  # the blocked problems are not in the table


def test_the_overall_problem_and_first_and_last_session_are_reported():
    trials = {
        1: {"m1": run_of([1], task=BLOCKED_NAME)},
        2: {"m1": run_of([2, 3, 5])},
        3: {"m1": run_of([6])},
    }
    first = row_of(sessions_per_problem_table(trials), "m1", 1)
    assert first["overall_problem"] == 2
    assert (first["first_session"], first["last_session"]) == (key(2), key(5))


def test_the_parts_of_a_split_session_count_once():
    split = {
        f"{key(1)}_part-01": session(A),
        f"{key(1)}_part-02": session(A),
        key(2): session(A),
    }
    trials = {1: {"m1": split}, 2: {"m1": run_of([3])}}
    row = row_of(sessions_per_problem_table(trials), "m1", 1)
    assert row["n_sessions"] == 2
    assert (row["first_session"], row["last_session"]) == (key(1), key(2))


def test_a_session_split_across_two_problems_counts_once_in_each():
    trials = {
        1: {"m1": {key(1): session(A), f"{key(2)}_part-01": session(A)}},
        2: {"m1": {f"{key(2)}_part-02": session(B), key(3): session(B)}},
    }
    table = sessions_per_problem_table(trials)
    assert table["n_sessions"].tolist() == [2, 2]
    assert row_of(table, "m1", 1)["last_session"] == key(2)
    assert row_of(table, "m1", 2)["first_session"] == key(2)


def test_a_cohort_without_drifting_problems_gives_an_empty_table_with_the_columns():
    table = sessions_per_problem_table({1: {"m1": run_of([1, 2], task=BLOCKED_NAME)}})
    assert list(table.columns) == list(TABLE_COLUMNS) and table.empty


def test_rows_are_sorted_by_mouse_then_problem():
    trials = {
        1: {"m2": run_of([1]), "m1": run_of([1])},
        2: {"m2": run_of([2]), "m1": run_of([2])},
    }
    table = sessions_per_problem_table(trials)
    assert table[["mouse", "problem_in_type"]].values.tolist() == [
        ["m1", 1],
        ["m1", 2],
        ["m2", 1],
        ["m2", 2],
    ]


# --- complete --------------------------------------------------------------------------


def test_only_the_latest_problem_of_a_mouse_is_still_running():
    trials = {1: {"m1": run_of([1, 2])}, 2: {"m1": run_of([3, 4])}, 3: {"m1": run_of([5])}}
    table = sessions_per_problem_table(trials)
    assert table["complete"].tolist() == [True, True, False]


def test_each_mouse_has_its_own_latest_problem():
    trials = {
        1: {"m1": run_of([1]), "m2": run_of([1])},
        2: {"m1": run_of([2]), "m2": run_of([2])},
        3: {"m2": run_of([3])},  # m1 stopped after drifting problem 2
    }
    table = sessions_per_problem_table(trials)
    done = {(r.mouse, r.problem_in_type): r.complete for r in table.itertuples()}
    assert done == {
        ("m1", 1): True,
        ("m1", 2): False,
        ("m2", 1): True,
        ("m2", 2): True,
        ("m2", 3): False,
    }


def test_a_mouse_whose_last_problem_is_blocked_has_only_complete_drifting_problems():
    trials = {
        1: {"m1": run_of([1, 2])},
        2: {"m1": run_of([3, 4])},
        3: {"m1": run_of([5], task=BLOCKED_NAME)},
    }
    table = sessions_per_problem_table(trials)
    assert table["complete"].tolist() == [True, True]


def test_a_drifting_problem_followed_by_a_blocked_one_is_complete():
    trials = {1: {"m1": run_of([1])}, 2: {"m1": run_of([2], task=BLOCKED_NAME)}}
    assert sessions_per_problem_table(trials)["complete"].tolist() == [True]


# --- provisional problems --------------------------------------------------------------------


def test_a_provisional_latest_problem_leaves_the_previous_one_not_yet_complete():
    # a final one-session run on a new layout may turn out to be a settings error, which would
    # leave the previous problem still running
    trials = {
        1: {"m1": run_of([1, 2])},
        2: {"m1": run_of([3, 4], B)},
        3: {"m1": run_of([5], A, provisional=True)},
    }
    table = sessions_per_problem_table(trials)
    assert table["complete"].tolist() == [True, False, False]
    assert table["provisional"].tolist() == [False, False, True]


def test_a_provisional_problem_after_a_blocked_one_changes_nothing_else():
    trials = {
        1: {"m1": run_of([1, 2], task=BLOCKED_NAME)},
        2: {"m1": run_of([3], B, provisional=True)},
    }
    table = sessions_per_problem_table(trials)
    assert table["complete"].tolist() == [False] and table["provisional"].tolist() == [True]


def test_a_provisional_session_earlier_in_the_run_does_not_matter():
    trials = {1: {"m1": run_of([1, 2], provisional=True)}, 2: {"m1": run_of([3, 4], B)}}
    table = sessions_per_problem_table(trials)
    assert table["complete"].tolist() == [True, False]
    assert table["provisional"].tolist() == [True, False]


def test_a_mouse_whose_last_session_was_left_out_keeps_its_last_problem_running():
    # a last session flagged as a settings error is not in any problem, so the problem before it
    # is still the mouse's latest
    trials = {1: {"m1": run_of([1, 2])}, 2: {"m1": run_of([3, 4], B)}}
    assert sessions_per_problem_table(trials)["complete"].tolist() == [True, False]


# --- port-finding-only sessions ----------------------------------------------------------


def port_finding(towers):
    return {"df": "frames", "choice_towers": frozenset(towers), "initiation_tower": "B2"}


def test_port_finding_only_sessions_are_counted_in_their_own_column_and_the_total():
    trials = {
        1: {"m1": run_of([1, 2], A)},
        2: {"m1": run_of([5, 6], B)},
    }
    found = {"m1": {key(3): port_finding(B), key(4): port_finding(B)}}
    assignment = assign_port_finding_sessions(trials, found)
    table = sessions_per_problem_table(trials, assignment)
    first, second = row_of(table, "m1", 1), row_of(table, "m1", 2)
    assert (first["n_sessions"], first["n_port_finding_only_sessions"]) == (2, 0)
    assert (second["n_sessions"], second["n_port_finding_only_sessions"]) == (2, 2)
    assert (first["n_sessions_total"], second["n_sessions_total"]) == (2, 4)
    # the session range is that of the bandit sessions
    assert (second["first_session"], second["last_session"]) == (key(5), key(6))


def test_an_unassigned_port_finding_session_is_counted_nowhere():
    trials = {1: {"m1": run_of([2, 3], A)}}
    found = {"m1": {key(1): port_finding(B)}}  # other towers than the next problem
    with pytest.warns(UserWarning, match="port-finding-only"):
        assignment = assign_port_finding_sessions(trials, found)
    row = row_of(sessions_per_problem_table(trials, assignment), "m1", 1)
    assert (row["n_port_finding_only_sessions"], row["n_sessions_total"]) == (0, 2)


def test_a_newest_unassigned_port_finding_session_leaves_the_last_problem_running():
    # port finding on new towers after the last problem may be the start of the next problem, but
    # no problem has taken it: the table keeps the conservative reading, the last problem running
    trials = {1: {"m1": run_of([1, 2], A)}, 2: {"m1": run_of([3, 4], B)}}
    found = {"m1": {key(5): port_finding(["A2", "B2", "C2"])}}
    with pytest.warns(UserWarning, match="port-finding-only"):
        assignment = assign_port_finding_sessions(trials, found)
    table = sessions_per_problem_table(trials, assignment)
    assert table["complete"].tolist() == [True, False]
    assert table["n_port_finding_only_sessions"].tolist() == [0, 0]


def test_without_an_assignment_there_are_no_port_finding_sessions():
    row = row_of(sessions_per_problem_table({1: {"m1": run_of([1])}}), "m1", 1)
    assert (row["n_port_finding_only_sessions"], row["n_sessions_total"]) == (0, 1)


# --- with_cohort ----------------------------------------------------------------------------


def test_with_cohort_puts_the_cohort_name_in_front():
    table = sessions_per_problem_table({1: {"m1": run_of([1])}})
    labelled = with_cohort(table, "cohort-03")
    assert list(labelled.columns) == ["cohort", *TABLE_COLUMNS]
    assert labelled["cohort"].tolist() == ["cohort-03"]


# --- summary --------------------------------------------------------------------------------


def table_of(rows):
    """Rows ``(mouse, k, n_sessions, complete)`` as a table (port finding 1 each, to be told)."""
    return pd.DataFrame(
        [
            {
                "mouse": mouse,
                "task_type": "drifting",
                "problem_in_type": k,
                "overall_problem": k + 1,
                "n_sessions": n,
                "n_port_finding_only_sessions": 1,
                "n_sessions_total": n + 1,
                "first_session": key(1),
                "last_session": key(2),
                "complete": complete,
                "provisional": False,
            }
            for mouse, k, n, complete in rows
        ],
        columns=list(TABLE_COLUMNS),
    )


ROWS = [
    ("m1", 1, 4, True),
    ("m1", 2, 6, True),
    ("m1", 3, 9, False),
    ("m2", 1, 6, True),
    ("m2", 2, 2, False),
    ("m3", 1, 8, True),
]


def test_the_summary_has_one_series_per_mouse_over_the_problems_present():
    summary = summarize_sessions_per_problem(table_of(ROWS))
    assert summary.mice == ["m1", "m2", "m3"]
    assert summary.problems == [1, 2, 3]
    np.testing.assert_array_equal(summary.by_mouse["m1"], [4, 6, 9])
    np.testing.assert_array_equal(summary.by_mouse["m2"], [6, 2, np.nan])
    np.testing.assert_array_equal(summary.complete["m2"], [True, False, False])


def test_the_mean_and_se_use_the_completed_problems_only():
    summary = summarize_sessions_per_problem(table_of(ROWS))
    np.testing.assert_array_equal(summary.n, [3, 1, 0])
    assert summary.mean[0] == pytest.approx(6.0)  # (4 + 6 + 8) / 3
    assert summary.se[0] == pytest.approx(np.std([4, 6, 8], ddof=1) / np.sqrt(3))
    assert summary.mean[1] == pytest.approx(6.0)  # m2's problem 2 is still running
    assert np.isnan(summary.se[1])  # one mouse: no standard error, never 0
    assert np.isnan(summary.mean[2])  # only m1's running problem


def test_the_total_replaces_the_bandit_count_on_request():
    summary = summarize_sessions_per_problem(table_of(ROWS), include_port_finding=True)
    np.testing.assert_array_equal(summary.by_mouse["m1"], [5, 7, 10])
    assert summary.mean[0] == pytest.approx(7.0)


def test_the_problem_axis_can_be_given():
    summary = summarize_sessions_per_problem(table_of(ROWS[:1]), problems=[1, 2, 3, 4])
    assert summary.problems == [1, 2, 3, 4]
    np.testing.assert_array_equal(summary.by_mouse["m1"], [4, np.nan, np.nan, np.nan])


def test_an_empty_table_raises():
    with pytest.raises(ValueError, match="no drifting problems"):
        summarize_sessions_per_problem(table_of([]))


def test_a_table_missing_a_column_raises():
    with pytest.raises(ValueError, match="n_sessions_total"):
        summarize_sessions_per_problem(
            table_of(ROWS).drop(columns="n_sessions_total"), include_port_finding=True
        )


def test_a_mouse_and_problem_listed_twice_raises():
    with pytest.raises(ValueError, match=r"m1.*problem 1"):
        summarize_sessions_per_problem(table_of([*ROWS, ("m1", 1, 3, True)]))


def test_given_problems_must_cover_those_of_the_table():
    with pytest.raises(ValueError, match="problems"):
        summarize_sessions_per_problem(table_of(ROWS), problems=[1, 2])
