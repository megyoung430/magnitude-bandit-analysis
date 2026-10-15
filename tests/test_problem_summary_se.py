import math

import numpy as np
import pytest

from src.behavior_analysis.problem_summary import (
    compute_problem_goodrev_summary,
    compute_problem_length_summary,
)

nan = float("nan")


def test_length_summary_se_is_nan_for_a_single_mouse():
    out = compute_problem_length_summary({"a": [3, 5], "b": [4]})
    assert out["ns"] == [2, 1]
    assert out["ses"][0] == pytest.approx(np.std([3, 4], ddof=1) / np.sqrt(2))
    assert math.isnan(out["ses"][1])  # never 0 for one value


def test_length_summary_counts_only_non_nan_mice():
    out = compute_problem_length_summary({"a": [3.0], "b": [5.0], "c": [nan]})
    assert out["ns"] == [2]
    assert out["means"] == pytest.approx([4.0])
    assert out["ses"] == pytest.approx([np.std([3, 5], ddof=1) / np.sqrt(2)])


def test_length_summary_of_only_nan_is_nan_with_zero_n():
    out = compute_problem_length_summary({"a": [nan]})
    assert out["ns"] == [0]
    assert math.isnan(out["means"][0]) and math.isnan(out["ses"][0])


def test_goodrev_summary_se_is_nan_for_a_single_mouse():
    out = compute_problem_goodrev_summary({1: {"a": 2.0, "b": 4.0}, 2: {"a": 3.0}})
    assert out["ns"] == [2, 1]
    assert out["ses"][0] == pytest.approx(np.std([2, 4], ddof=1) / np.sqrt(2))
    assert math.isnan(out["ses"][1])


def test_summarize_by_problem_se_is_nan_for_one_mouse_and_skips_nan():
    from src.behavior_analysis.problem_utils import summarize_by_problem, summarize_by_problem_safe

    metric = {"a": {1: 2.0, 2: 3.0}, "b": {1: 4.0, 2: nan}, "c": {1: 6.0}}
    out = summarize_by_problem(metric)
    assert out["ns"] == [3, 1]
    assert out["ses"][0] == pytest.approx(np.std([2, 4, 6], ddof=1) / np.sqrt(3))
    assert math.isnan(out["ses"][1])
    safe = summarize_by_problem_safe({"a": {1: 2.0}, "b": {1: 4.0, 2: 1.0}})
    assert safe[1]["se"] == pytest.approx(np.std([2, 4], ddof=1) / np.sqrt(2))
    assert math.isnan(safe[2]["se"])


def test_first_leave_se_is_nan_for_one_subject():
    from src.behavior_analysis.get_first_leave_after_good_reversals import (
        average_first_leave_across_subjects,
    )

    counts = {"a": {"new_best": 3, "third": 1, "total": 4}}
    mean, se, n = average_first_leave_across_subjects(counts)
    assert n == 1 and math.isnan(se["new_best"]) and math.isnan(se["third"])
    counts["b"] = {"new_best": 1, "third": 1, "total": 4}
    _, se, _ = average_first_leave_across_subjects(counts)
    assert se["new_best"] == pytest.approx(np.std([0.75, 0.25], ddof=1) / np.sqrt(2))


def test_block_length_se_is_nan_below_two_values_and_skips_nan():
    from src.behavior_analysis.get_task_statistics import _standard_error

    assert _standard_error([1.0, 3.0]) == pytest.approx(np.sqrt(2) / np.sqrt(2))
    assert math.isnan(_standard_error([5.0]))
    assert math.isnan(_standard_error([5.0, nan]))
