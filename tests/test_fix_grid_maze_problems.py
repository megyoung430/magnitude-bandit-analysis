import pytest

from src.behavior_import.fix_grid_maze_cohort_02_problems import fix_grid_maze_cohort_02_problems


def by_problem(**problems):
    """``p2={'A': 'x'}`` -> ``{2: {'A': 'x'}}`` (keyword names are ``p<number>``)."""
    return {int(name[1:]): subjects for name, subjects in problems.items()}


def test_default_subjects_shift_down_and_last_problem_is_removed():
    data = by_problem(
        p1={"MY_04_R": "a1"}, p2={"MY_04_R": "a2"}, p3={"MY_04_R": "a3"}, p4={"MY_04_R": "a4"}
    )
    assert fix_grid_maze_cohort_02_problems(data) == by_problem(
        p1={"MY_04_R": "a1"}, p2={"MY_04_R": "a3"}, p3={"MY_04_R": "a4"}
    )


def test_both_default_subjects_are_fixed_and_others_are_untouched():
    data = by_problem(
        p1={"MY_04_R": 1, "MY_05_R": 1, "other": 1},
        p2={"MY_04_R": 2, "MY_05_R": 2, "other": 2},
        p3={"MY_04_R": 3, "MY_05_R": 3, "other": 3},
    )
    assert fix_grid_maze_cohort_02_problems(data) == by_problem(
        p1={"MY_04_R": 1, "MY_05_R": 1, "other": 1},
        p2={"MY_04_R": 3, "MY_05_R": 3, "other": 2},
        p3={"other": 3},
    )


def test_fix_subjects_and_start_problem_select_what_shifts():
    data = by_problem(p2={"A": 2, "B": 2}, p3={"A": 3, "B": 3}, p4={"A": 4, "B": 4})
    fixed = fix_grid_maze_cohort_02_problems(data, ["A"], 3)
    assert fixed == by_problem(p2={"A": 2, "B": 2}, p3={"A": 4, "B": 3}, p4={"B": 4})


def test_start_problem_is_inclusive():
    data = by_problem(p2={"A": 2}, p3={"A": 3})
    assert fix_grid_maze_cohort_02_problems(data, ["A"], 2) == by_problem(p2={"A": 3})


def test_subject_needs_a_problem_to_take_data_from():
    only_start = by_problem(p1={"A": 1}, p2={"A": 2})
    assert fix_grid_maze_cohort_02_problems(only_start, ["A"], 2) == only_start
    absent = by_problem(p2={"B": 2}, p3={"B": 3})
    assert fix_grid_maze_cohort_02_problems(absent, ["A"]) == absent


def test_shift_skips_problems_the_subject_never_did():
    data = by_problem(p2={"A": 2}, p3={"B": 3}, p5={"A": 5})
    assert fix_grid_maze_cohort_02_problems(data, ["A"]) == by_problem(p2={"A": 5}, p3={"B": 3})


def test_problem_stays_while_other_subjects_remain_and_empty_ones_are_removed():
    data = by_problem(p2={"A": 2}, p3={"A": 3, "B": 3}, p4={})
    assert fix_grid_maze_cohort_02_problems(data, ["A"]) == by_problem(
        p2={"A": 3}, p3={"B": 3}, p4={}
    )


def test_copy_by_default_and_mutate_in_place_on_request():
    data = by_problem(p2={"A": 2}, p3={"A": 3})
    copied = fix_grid_maze_cohort_02_problems(data, ["A"])
    assert data == by_problem(p2={"A": 2}, p3={"A": 3})
    assert copied is not data

    returned = fix_grid_maze_cohort_02_problems(data, ["A"], in_place=True)
    assert returned is data
    assert data == by_problem(p2={"A": 3})


def test_copy_does_not_share_session_data_with_the_input():
    sessions = {"ses-1": {"trial": [1]}}
    data = by_problem(p1={"A": sessions}, p2={"A": sessions})
    fixed = fix_grid_maze_cohort_02_problems(data, ["A"], 1)
    fixed[1]["A"]["ses-1"]["trial"].append(2)
    assert sessions == {"ses-1": {"trial": [1]}}


def test_unsortable_problem_keys_raise():
    with pytest.raises(TypeError):
        fix_grid_maze_cohort_02_problems({"a": {}, 2: {}})
