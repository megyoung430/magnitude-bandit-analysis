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
