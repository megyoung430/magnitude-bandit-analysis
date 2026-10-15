"""Task type from the pyControl task name, and per-mouse numbering within a type."""

import pandas as pd
import pytest

from src.behavior_import.task_types import (
    BLOCKED,
    DRIFTING,
    classify_task_name,
    number_within_task_type,
    problem_map_table,
    regroup_by_task_type,
    session_task_type,
)

BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"


def session(task_name, path="/raw/sub-01_id-A/ses-1_date-20260101/A-1.tsv", problem=None):
    frame = pd.DataFrame({"type": []})
    frame.attrs["source_path"] = path
    return {"task_name": task_name, "df": frame, "problem": problem}


def mouse(task_names, subject="A"):
    """``{problem: {subject: {session_key: session}}}`` with one session per problem."""
    return {
        p: {subject: {f"ses-{p}_date-2026010{p}": session(name, problem=p)}}
        for p, name in enumerate(task_names, start=1)
    }


def merge(*by_problem):
    out = {}
    for one in by_problem:
        for p, subjects in one.items():
            out.setdefault(p, {}).update(subjects)
    return out


@pytest.mark.parametrize(
    "name, expected",
    [
        (BLOCKED_NAME, BLOCKED),
        ("Meg/3x3_Field_STRUCTURED_bandit", BLOCKED),
        (DRIFTING_NAME, DRIFTING),
        ("DRIFTING", DRIFTING),
        ("3x3_field_drifting_value_bandit", DRIFTING),
    ],
)
def test_classify_task_name(name, expected):
    assert classify_task_name(name) == expected


def test_only_the_last_path_component_counts():
    # the folder "blocked_lab" must not make a drifting task blocked
    assert classify_task_name("blocked_lab\\drifting_task") == DRIFTING


@pytest.mark.parametrize("name", ["Meg\\something_else", "", None])
def test_unknown_or_missing_name_raises(name):
    with pytest.raises(ValueError, match="task name"):
        classify_task_name(name)


def test_name_matching_both_raises():
    with pytest.raises(ValueError, match="both"):
        classify_task_name("blocked_drifting_bandit")


def test_session_error_names_subject_session_file_and_task_name():
    sess = session("Meg\\odd_task", path="/raw/x/odd.tsv")
    with pytest.raises(ValueError) as err:
        session_task_type("MY_1", "ses-3_date-20260105", sess)
    for text in ("MY_1", "ses-3_date-20260105", "/raw/x/odd.tsv", "odd_task"):
        assert text in str(err.value)


def test_session_without_a_file_path_still_classifies():
    sess = {"task_name": DRIFTING_NAME}
    assert session_task_type("A", "s", sess) == DRIFTING


def test_mixed_types_within_a_problem_raise_naming_the_sessions():
    by_problem = {
        1: {
            "A": {
                "ses-1_date-20260101": session(BLOCKED_NAME),
                "ses-2_date-20260102": session(DRIFTING_NAME),
            }
        }
    }
    with pytest.raises(ValueError, match="ses-1_date-20260101.*ses-2_date-20260102"):
        number_within_task_type(by_problem)


def test_numbering_is_per_mouse_within_type():
    a = mouse([BLOCKED_NAME, BLOCKED_NAME, DRIFTING_NAME, DRIFTING_NAME], subject="A")
    b = mouse([BLOCKED_NAME] * 3 + [DRIFTING_NAME], subject="B")
    numbering = number_within_task_type(merge(a, b))
    assert numbering[("A", 3)] == (DRIFTING, 1)
    assert numbering[("A", 4)] == (DRIFTING, 2)
    assert numbering[("B", 3)] == (BLOCKED, 3)
    assert numbering[("B", 4)] == (DRIFTING, 1)


def test_drifting_one_lines_up_across_mice_after_regrouping():
    a = mouse([BLOCKED_NAME, BLOCKED_NAME, DRIFTING_NAME], subject="A")
    b = mouse([BLOCKED_NAME] * 3 + [DRIFTING_NAME], subject="B")
    grouped = regroup_by_task_type(merge(a, b))
    assert list(grouped) == [(BLOCKED, 1), (BLOCKED, 2), (BLOCKED, 3), (DRIFTING, 1)]
    assert set(grouped[(BLOCKED, 3)]) == {"B"}
    drifting = grouped[(DRIFTING, 1)]
    assert set(drifting) == {"A", "B"}
    # each mouse keeps its own sessions, still tagged with its overall problem number
    assert [s["problem"] for s in drifting["A"].values()] == [3]
    assert [s["problem"] for s in drifting["B"].values()] == [4]


def test_regrouping_does_not_change_the_input():
    by_problem = mouse([BLOCKED_NAME, DRIFTING_NAME])
    regroup_by_task_type(by_problem)
    assert list(by_problem) == [1, 2]
    assert list(by_problem[1]["A"]) == ["ses-1_date-20260101"]


def test_problem_map_table_columns_and_rows():
    table = problem_map_table(mouse([BLOCKED_NAME, DRIFTING_NAME], subject="A"))
    assert list(table.columns) == [
        "mouse",
        "task_type",
        "problem_in_type",
        "overall_problem",
        "sessions",
    ]
    assert table.to_dict("records") == [
        {
            "mouse": "A",
            "task_type": BLOCKED,
            "problem_in_type": 1,
            "overall_problem": 1,
            "sessions": "ses-1_date-20260101",
        },
        {
            "mouse": "A",
            "task_type": DRIFTING,
            "problem_in_type": 1,
            "overall_problem": 2,
            "sessions": "ses-2_date-20260102",
        },
    ]


def test_problem_map_lists_sessions_in_order():
    sessions = {
        "ses-2_date-20260102": session(DRIFTING_NAME),
        "ses-1_date-20260101": session(DRIFTING_NAME),
    }
    table = problem_map_table({1: {"A": sessions}})
    assert table["sessions"].tolist() == ["ses-1_date-20260101;ses-2_date-20260102"]
