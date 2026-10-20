"""Dropping mice from the drifting problems before they are pooled (no files, no plotting)."""

from datetime import datetime

import pandas as pd
import pytest

from src.behavior_analysis.exclude_mice import (
    check_mouse_ids,
    exclude_mice_from_problems,
    exclude_mice_from_table,
    excluded_mice_description,
    excluded_scope_text,
    excluded_set_name,
)

# {problem: {mouse: sessions}}; the sessions are opaque to the helper
PROBLEMS = {
    1: {"MY_1_A": {"s1": "a1"}, "MY_2_B": {"s1": "b1"}, "MY_3_C": {"s1": "c1"}},
    2: {"MY_2_B": {"s2": "b2"}, "MY_3_C": {"s2": "c2"}},
    3: {"MY_3_C": {"s3": "c3"}},
}


def test_the_named_mice_are_dropped_from_every_problem():
    kept = exclude_mice_from_problems(PROBLEMS, ["MY_2_B"])
    assert {k: sorted(mice) for k, mice in kept.items()} == {
        1: ["MY_1_A", "MY_3_C"],
        2: ["MY_3_C"],
        3: ["MY_3_C"],
    }
    assert kept[1]["MY_1_A"] is PROBLEMS[1]["MY_1_A"]  # the sessions are passed on untouched


def test_the_input_is_not_modified():
    exclude_mice_from_problems(PROBLEMS, ["MY_2_B"])
    assert sorted(PROBLEMS[1]) == ["MY_1_A", "MY_2_B", "MY_3_C"]


def test_a_problem_left_without_mice_is_dropped():
    kept = exclude_mice_from_problems(PROBLEMS, ["MY_3_C"])
    assert list(kept) == [1, 2]  # problem 3 had only MY_3_C
    assert {k: sorted(mice) for k, mice in kept.items()} == {
        1: ["MY_1_A", "MY_2_B"],
        2: ["MY_2_B"],
    }


def test_a_mouse_that_is_not_in_the_problems_raises_and_lists_the_mice_that_are():
    with pytest.raises(ValueError, match="MY_9_Z") as err:
        exclude_mice_from_problems(PROBLEMS, ["MY_2_B", "MY_9_Z"])
    assert "MY_1_A, MY_2_B, MY_3_C" in str(err.value)


def test_excluding_every_mouse_raises():
    with pytest.raises(ValueError, match="no mice left"):
        exclude_mice_from_problems(PROBLEMS, ["MY_1_A", "MY_2_B", "MY_3_C"])


def test_keys_that_are_not_integers_work_too():
    # the all-cohorts pool keys its problems (cohort id, k)
    problems = {(1, 1): {"a": {}, "b": {}}, (3, 1): {"c": {}}}
    assert list(exclude_mice_from_problems(problems, ["c"])) == [(1, 1)]


# --- ids -----------------------------------------------------------------------------------


def test_ids_come_back_sorted_and_once_each():
    assert check_mouse_ids(["MY_66_LL", "MY_175_N", "MY_66_LL"]) == ("MY_175_N", "MY_66_LL")


@pytest.mark.parametrize("bad", [[], [""], ["MY 1"], ["../x"], ["a/b"], [3]])
def test_an_empty_list_or_an_id_that_is_not_a_plain_name_raises(bad):
    with pytest.raises(ValueError):
        check_mouse_ids(bad)


# --- names and texts -----------------------------------------------------------------------


def test_the_folder_name_lists_the_sorted_ids():
    assert (
        excluded_set_name("all-drifting-problems", ["MY_66_LL", "MY_175_N"])
        == "all-drifting-problems-without-MY_175_N-MY_66_LL"
    )


def test_the_scope_text_lists_the_sorted_ids():
    assert excluded_scope_text(["MY_66_LL", "MY_175_N"]) == "without MY_175_N, MY_66_LL"


def test_the_description_names_the_mice_the_scope_and_the_run_time():
    text = excluded_mice_description(
        "cohort-03", ["MY_66_LL", "MY_175_N"], datetime(2026, 10, 20, 9, 30, 5)
    )
    assert "cohort-03" in text
    assert "MY_175_N" in text and "MY_66_LL" in text
    assert "2026-10-20 09:30:05" in text
    assert text.endswith("\n")


# --- tables --------------------------------------------------------------------------------


def test_a_table_loses_the_rows_of_the_excluded_mice():
    table = pd.DataFrame({"mouse": ["a", "b", "a", "c"], "n": [1, 2, 3, 4]})
    kept = exclude_mice_from_table(table, ["a"])
    assert kept.to_dict("list") == {"mouse": ["b", "c"], "n": [2, 4]}


def test_a_table_without_a_mouse_column_raises():
    with pytest.raises(ValueError, match="mouse"):
        exclude_mice_from_table(pd.DataFrame({"x": [1]}), ["a"])
