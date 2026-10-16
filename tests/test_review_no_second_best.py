"""Blocks where more than one arm is at 0, so there is no second best arm to choose."""

import pandas as pd

from src.behavior_analysis.no_second_best import COLUMNS, no_second_best_table
from tests.review_fixtures import BLOCKED_NAME, cohort, session

S1, S2 = "ses-1_date-20260101", "ses-2_date-20260102"


def mags(a, b, c):
    """``{tower: [magnitude per trial]}`` from three per-trial lists."""
    return {"A1": a, "B1": b, "C1": c}


def table_of(problems):
    _, index, grouped, _ = cohort(problems)
    return no_second_best_table(grouped, index)


def test_a_block_with_two_zero_arms_is_listed_with_its_only_rewarded_arm():
    one_session = session(
        magnitudes=mags([4, 4, 4, 4, 4, 3, 3], [2, 2, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 2]),
        blocks=[1, 1, 2, 2, 2, 3, 3],
    )
    table = table_of({1: {"M1": {S1: one_session}}})
    assert list(table.columns) == list(COLUMNS)
    (row,) = table.to_dict("records")
    assert (row["mouse"], row["task_type"], row["problem_in_type"], row["overall_problem"]) == (
        "M1",
        "drifting",
        1,
        1,
    )
    assert (row["sessions"], row["block"]) == (S1, 2)
    assert (row["first_trial"], row["last_trial"], row["n_trials"]) == (3, 5, 3)
    assert (row["first_trial_in_session"], row["last_trial_in_session"]) == (3, 5)
    assert (row["rewarded_arm"], row["rewarded_magnitude"]) == ("A1", "4")
    assert row["magnitudes"] == "A1=4, B1=0, C1=0"
    assert row["n_trials_multiple_zero"] == 3


def test_a_block_that_continues_into_the_next_session_names_both_sessions():
    first = session(
        magnitudes=mags([3, 3, 3], [0, 0, 0], [0, 0, 0]),
        blocks=[1, 1, 1],
    )
    second = session(
        magnitudes=mags([3, 3, 1], [0, 0, 4], [0, 0, 2]),
        blocks=[1, 1, 2],
    )
    (row,) = table_of({1: {"M1": {S1: first, S2: second}}}).to_dict("records")
    assert row["sessions"] == f"{S1};{S2}"
    assert (row["block"], row["first_trial"], row["last_trial"], row["n_trials"]) == (1, 1, 5, 5)
    assert (row["first_trial_in_session"], row["last_trial_in_session"]) == (1, 2)


def test_only_some_trials_of_a_block_may_have_two_zero_arms():
    mixed = session(
        magnitudes=mags([4, 4, 4, 4], [2, 0, 0, 2], [0, 0, 0, 0]),
        blocks=[1, 1, 1, 1],
    )
    (row,) = table_of({1: {"M1": {S1: mixed}}}).to_dict("records")
    assert (row["first_trial"], row["last_trial"], row["n_trials"]) == (1, 4, 4)
    # the extent is the block's, not that of the trials with two zero arms (trials 2 and 3)
    assert (row["first_trial_in_session"], row["last_trial_in_session"]) == (1, 4)
    assert row["n_trials_multiple_zero"] == 2
    # described at the first trial with two zero arms
    assert row["magnitudes"] == "A1=4, B1=0, C1=0"


def test_all_arms_at_zero_has_no_rewarded_arm():
    dead = session(magnitudes=mags([0, 0], [0, 0], [0, 0]), blocks=[1, 1])
    (row,) = table_of({1: {"M1": {S1: dead}}}).to_dict("records")
    assert pd.isna(row["rewarded_arm"]) and pd.isna(row["rewarded_magnitude"])


def test_trials_with_a_missing_magnitude_are_not_counted():
    gaps = session(magnitudes=mags([4, 4], [None, 0], [0, 0]), blocks=[1, 1])
    (row,) = table_of({1: {"M1": {S1: gaps}}}).to_dict("records")
    assert row["n_trials_multiple_zero"] == 1


def test_a_trial_with_a_missing_magnitude_is_never_a_two_zero_trial_even_with_two_zeros():
    gaps = session(magnitudes=mags([4, 4], [0, 0], [None, 0]), blocks=[1, 1])
    (row,) = table_of({1: {"M1": {S1: gaps}}}).to_dict("records")
    assert row["n_trials_multiple_zero"] == 1
    assert row["magnitudes"] == "A1=4, B1=0, C1=0"


def test_exactly_two_zero_arms_qualify_and_one_does_not():
    two = session(magnitudes=mags([4], [0], [0]), blocks=[1])
    one = session(magnitudes=mags([4], [1], [0]), blocks=[1])
    assert len(table_of({1: {"M1": {S1: two}}})) == 1
    assert table_of({1: {"M1": {S1: one}}}).empty


def test_blocks_with_one_zero_arm_give_an_empty_table_with_the_columns():
    normal = session(magnitudes=mags([4, 4], [2, 2], [0, 0]), blocks=[1, 1])
    table = table_of({1: {"M1": {S1: normal}}})
    assert table.empty
    assert list(table.columns) == list(COLUMNS)


def test_each_mouse_and_problem_is_checked_on_its_own():
    dead = session(magnitudes=mags([4, 4], [0, 0], [0, 0]), blocks=[1, 1])
    dead_blocked = {**dead, "task_name": BLOCKED_NAME}
    fine = session(magnitudes=mags([4, 4], [2, 2], [0, 0]), blocks=[1, 1])
    table = table_of(
        {1: {"M1": {S1: dead_blocked}, "M2": {S1: fine}}, 2: {"M1": {S2: fine}, "M2": {S2: dead}}}
    )
    got = {(r["mouse"], r["task_type"], r["overall_problem"]) for r in table.to_dict("records")}
    assert got == {("M1", "blocked", 1), ("M2", "drifting", 2)}
