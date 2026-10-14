import math

import numpy as np
import pytest

from src.behavior_analysis.problem_summary import compute_problem_blocklen_summary


def stats(**problems):
    """``p1={'m': {1: 10.0}}`` -> block stats for problem 1 with that per-mouse table."""
    return {
        int(name[1:]): {"block_stats": {"per_mouse_blocklens": per_mouse}}
        for name, per_mouse in problems.items()
    }


def test_averages_blocks_per_mouse_then_summarises_across_mice():
    data = stats(p1={"a": {1: 10.0, 2: 20.0}, "b": {1: 30.0}})
    out = compute_problem_blocklen_summary(data)
    assert out["per_mouse_blocklens"] == {"a": {1: 15.0}, "b": {1: 30.0}}
    assert out["blocks"] == [1]
    assert out["means"] == [22.5]
    assert out["ns"] == [2]
    assert out["ses"] == pytest.approx([np.std([15.0, 30.0], ddof=1) / math.sqrt(2)])
    assert out["mice"] == ["a", "b"]


def test_problems_are_sorted_and_mice_missing_from_one_are_skipped_there():
    data = stats(
        p3={"a": {1: 9.0}},
        p1={"a": {1: 1.0}, "b": {1: 3.0}},
    )
    out = compute_problem_blocklen_summary(data)
    assert out["blocks"] == [1, 3]
    assert out["means"] == [2.0, 9.0]
    assert out["ns"] == [2, 1]
    assert out["ses"][1] == 0.0  # a single mouse has no spread
    assert out["per_mouse_blocklens"] == {"a": {1: 1.0, 3: 9.0}, "b": {1: 3.0}}


def test_values_are_plain_floats():
    out = compute_problem_blocklen_summary(stats(p1={"a": {1: 4}, "b": {1: 6}}))
    assert [type(v) for v in out["means"] + out["ses"]] == [float, float]
    assert type(out["ns"][0]) is int


def test_a_mouse_with_no_blocks_in_a_problem_does_not_count_there():
    out = compute_problem_blocklen_summary(stats(p1={"a": {1: 5.0}, "b": {}}))
    assert out["ns"] == [1]
    assert out["mice"] == ["a"]


def test_a_problem_nobody_has_blocks_for_is_nan():
    out = compute_problem_blocklen_summary(stats(p1={"a": {}}, p2={"a": {1: 2.0}}))
    assert out["blocks"] == [1, 2]
    assert out["ns"] == [0, 1]
    assert math.isnan(out["means"][0]) and math.isnan(out["ses"][0])


def test_drop_last_problem_removes_each_mouses_highest_problem():
    data = stats(p1={"a": {1: 1.0}, "b": {1: 3.0}}, p2={"a": {1: 5.0}}, p3={"b": {1: 7.0}})
    out = compute_problem_blocklen_summary(data, drop_last_problem_per_mouse=True)
    assert out["per_mouse_blocklens"] == {"a": {1: 1.0}, "b": {1: 3.0}}
    assert out["blocks"] == [1]
    assert out["ns"] == [2]


def test_drop_last_problem_removes_mice_left_with_nothing():
    data = stats(p1={"a": {1: 1.0}, "b": {1: 3.0}}, p2={"b": {1: 7.0}})
    out = compute_problem_blocklen_summary(data, True)
    assert out["mice"] == ["b"]
    assert out["per_mouse_blocklens"] == {"b": {1: 3.0}}
    assert out["blocks"] == [1]


def test_nothing_is_dropped_by_default():
    data = stats(p1={"a": {1: 1.0}}, p2={"a": {1: 5.0}})
    assert compute_problem_blocklen_summary(data)["blocks"] == [1, 2]


def test_no_problems_gives_an_empty_summary():
    assert compute_problem_blocklen_summary({}) == {
        "blocks": [],
        "means": [],
        "ses": [],
        "ns": [],
        "per_mouse_blocklens": {},
        "mice": [],
    }


def test_input_is_not_modified():
    data = stats(p1={"a": {1: 1.0}}, p2={"a": {1: 5.0}})
    before = {p: dict(v["block_stats"]["per_mouse_blocklens"]) for p, v in data.items()}
    compute_problem_blocklen_summary(data, True)
    assert {p: v["block_stats"]["per_mouse_blocklens"] for p, v in data.items()} == before


def test_malformed_stats_raise():
    with pytest.raises(KeyError):
        compute_problem_blocklen_summary({1: {}})
