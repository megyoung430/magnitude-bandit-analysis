"""Where a session sits in a mouse's problems, for sessions in and out of a problem."""

from tests.review_fixtures import BLOCKED_NAME, DRIFTING_NAME, cohort, session

S1, S2, S3, S4, S5 = (f"ses-{n}_date-2026010{n}" for n in (1, 2, 3, 4, 5))


def three_problem_index():
    """M1: blocked problem 1 (ses 1), drifting problems 2 and 3 (ses 2 and ses 4)."""
    _, index, _, _ = cohort(
        {
            1: {"M1": {S1: session(task_name=BLOCKED_NAME)}},
            2: {"M1": {S2: session(task_name=DRIFTING_NAME)}},
            3: {"M1": {S4: session(task_name=DRIFTING_NAME)}},
        }
    )
    return index


def test_a_session_in_a_problem_has_its_own_problem():
    columns = three_problem_index().columns("M1", S2)
    assert (columns["task_type"], columns["problem_in_type"], columns["overall_problem"]) == (
        "drifting",
        1,
        2,
    )


def test_a_session_in_no_problem_gets_the_latest_earlier_problem_not_the_first():
    columns = three_problem_index().columns("M1", S5)
    assert (columns["task_type"], columns["problem_in_type"], columns["overall_problem"]) == (
        "drifting",
        2,
        3,
    )


def test_a_session_between_problems_gets_the_problem_before_it():
    columns = three_problem_index().columns("M1", S3)
    assert columns["overall_problem"] == 2


def test_a_session_before_every_problem_has_no_problem_but_its_own_task_type():
    columns = three_problem_index().columns("M1", "ses-0_date-20251231", BLOCKED_NAME)
    assert columns == {"task_type": "blocked", "problem_in_type": None, "overall_problem": None}


def test_problem_columns_find_the_overall_number_of_a_problem_in_a_task_type():
    index = three_problem_index()
    assert index.problem_columns("M1", "drifting", 2)["overall_problem"] == 3
    assert index.problem_columns("M1", "drifting", 3)["overall_problem"] is None
    assert index.problem_columns(None, "drifting", 2) == {
        "task_type": "drifting",
        "problem_in_type": 2,
        "overall_problem": None,
    }
