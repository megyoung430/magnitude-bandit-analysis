import numpy as np

from src.behavior_analysis.blocklen_distributions import (
    collect_blocklens,
    drop_last_problem_per_mouse_raw,
    pooled_blocklens,
)

nan = float("nan")


def block_stats():
    return {
        2: {"block_stats": {"per_mouse_blocklens": {"A": {1: 30.0, 2: 40.0}}}},
        1: {"block_stats": {"per_mouse_blocklens": {"B": {1: 20.0}, "A": {1: 10.0, 2: nan}}}},
    }


def test_collect_blocklens_sorts_keys_fills_missing_mice_and_drops_nan():
    problems, mice, data = collect_blocklens(block_stats())
    assert problems == [1, 2]
    assert mice == ["A", "B"]
    np.testing.assert_array_equal(data[1]["A"], [10.0])  # the NaN block is dropped
    np.testing.assert_array_equal(data[2]["A"], [30.0, 40.0])
    assert data[2]["B"].size == 0  # B has no entry in problem 2


def test_drop_last_problem_removes_each_mouses_highest_problem_with_data():
    data = {
        1: {"A": np.array([1.0]), "B": np.array([2.0])},
        2: {"A": np.array([3.0]), "B": np.array([])},
        3: {"A": np.array([5.0]), "B": np.array([])},
    }
    problems, mice, out = drop_last_problem_per_mouse_raw([1, 2, 3], ["A", "B"], data)
    # A loses problem 3 (its last), B loses problem 1 (its only one) and so drops out entirely.
    assert problems == [1, 2]
    assert mice == ["A"]
    np.testing.assert_array_equal(out[1]["A"], [1.0])
    np.testing.assert_array_equal(out[2]["A"], [3.0])
    assert 3 not in out


def test_drop_last_problem_does_not_modify_its_input():
    data = {1: {"A": np.array([1.0])}, 2: {"A": np.array([2.0])}}
    drop_last_problem_per_mouse_raw([1, 2], ["A"], data)
    np.testing.assert_array_equal(data[2]["A"], [2.0])


def test_drop_last_problem_can_remove_everything():
    data = {1: {"A": np.array([1.0])}}
    assert drop_last_problem_per_mouse_raw([1], ["A"], data) == ([], [], {})


def test_pooled_blocklens_selects_problems_and_mice():
    data = {
        1: {"A": np.array([1.0, 2.0]), "B": np.array([]), "C": np.array([3.0])},
        2: {"A": np.array([4.0]), "B": np.array([5.0]), "C": np.array([])},
    }
    np.testing.assert_array_equal(pooled_blocklens(data, [1], ["A", "B", "C"]), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(pooled_blocklens(data, [1, 2], ["A"]), [1.0, 2.0, 4.0])
    assert pooled_blocklens(data, [1], ["B"]).size == 0
    assert pooled_blocklens(data, [], ["A"]).size == 0


def test_collect_blocklens_with_no_problems_is_empty():
    assert collect_blocklens({}) == ([], [], {})
