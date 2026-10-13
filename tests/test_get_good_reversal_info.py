"""Unit tests for the pure helpers behind get_good_reversal_info (synthetic inputs only)."""

import pytest

from src.behavior_analysis import get_good_reversal_info as mod
from src.behavior_analysis.get_good_reversal_info import _SubjectSeries


def series(**fields):
    return _SubjectSeries.from_merged(fields)


def test_from_merged_turns_missing_and_none_fields_into_empty():
    s = series(num_blocks=None, choices_by_tower=None)
    assert s.good is None
    assert s.num_blocks == [] and s.blocks == []
    assert s.reward_by_tower == {} and s.choices_by_tower == {} and s.choices_by_rank == {}


def test_find_change_indices():
    assert mod._find_change_indices([1, 1, 2, 2, 3]) == [2, 4]
    assert mod._find_change_indices([1]) == []


def test_count_trials_prefers_num_blocks_then_blocks_then_rewards_then_choices():
    assert mod._count_trials(series(num_blocks=[0, 1], blocks=[1, 1, 1])) == 2
    assert (
        mod._count_trials(series(blocks=[1, 1, 1], reward_magnitudes_by_tower={"A": [1] * 9})) == 3
    )
    assert mod._count_trials(series(reward_magnitudes_by_tower={"A": [1] * 4, "B": [1] * 6})) == 6
    assert mod._count_trials(series(choices_by_tower={"A": [1] * 5})) == 5
    assert mod._count_trials(series()) == 0


def test_list_towers_prefers_reward_keys():
    both = series(reward_magnitudes_by_tower={"A": [1]}, choices_by_tower={"B": [1]})
    assert mod._list_towers(both) == ["A"]
    assert mod._list_towers(series(choices_by_tower={"B": [1]})) == ["B"]


@pytest.mark.parametrize(
    "fields, any_good, expected",
    [
        (
            dict(good_reversals=[0, 0, 1], num_blocks=[0, 1, 1], blocks=[1, 2, 2]),
            True,
            ([2], "good_reversals"),
        ),
        (
            dict(good_reversals=[0, 0, 1], num_blocks=[0, 1, 1], blocks=[1, 2, 2]),
            False,
            ([1], "num_blocks"),
        ),
        (dict(good_reversals=[0], blocks=[1, 1, 2]), True, ([2], "blocks")),
        (dict(blocks=[1]), False, ([], None)),
    ],
)
def test_find_boundaries_priority(fields, any_good, expected):
    assert mod._find_boundaries(series(**fields), any_good) == expected


def test_block_id_at_returns_none_past_the_end():
    assert mod._block_id_at([1, 2], 1) == 2
    assert mod._block_id_at([1, 2], 2) is None


def test_window_indices_clip_to_next_boundary_and_trial_count():
    # post=None runs to the next boundary; the last boundary runs to n_trials.
    assert mod._window_indices([3, 6], 0, 10, pre=2, post=None) == ([1, 2], [3, 4, 5])
    assert mod._window_indices([3, 6], 1, 10, pre=5, post=None) == ([1, 2, 3, 4, 5], [6, 7, 8, 9])
    # an explicit post is clipped by both the next boundary and the trial count.
    assert mod._window_indices([3, 6], 0, 10, pre=0, post=2) == ([], [3, 4])
    assert mod._window_indices([3, 6], 0, 10, pre=1, post=100) == ([2], [3, 4, 5])
    assert mod._window_indices([3, 6], 1, 8, pre=1, post=100) == ([5], [6, 7])


def test_slice_choices_drops_indices_past_array_end():
    sliced = mod._slice_choices({"A": [0, 1, 0], "B": []}, [1, 2], [2, 3, 4])
    assert sliced == {"A": {"pre": [1, 0], "post": [0]}, "B": {"pre": [], "post": []}}


def test_before_trial_clamps_at_zero():
    assert mod._before_trial(0) == 0
    assert mod._before_trial(4) == 3


def test_rewards_at_is_all_none_when_any_array_is_too_short():
    rewards = {"A": [1, 2, 3], "B": [4, 5]}
    assert mod._rewards_at(rewards, ["A", "B"], 1) == {"A": 2, "B": 5}
    assert mod._rewards_at(rewards, ["A", "B"], 2) == {"A": None, "B": None}


def test_skip_message_reports_each_reason_in_order():
    rewards = {"A": [0, 0, 1], "B": [0, 1, 1], "C": [1, 1]}
    s = series(blocks=[1, 1, 2], reward_magnitudes_by_tower=rewards)
    assert "missing towers" in mod._skip_message("s", 1, "blocks", s, ["A", "Z"], 1)
    assert "too short" in mod._skip_message("s", 3, "blocks", s, ["A", "B", "C"], 1)
    zero_msg = mod._skip_message("s", 1, "blocks", s, ["A", "B"], 0)
    assert "more than 0 zero rewards in [0, 0]" in zero_msg and "block 1" in zero_msg
    assert mod._skip_message("s", 1, "blocks", s, ["A", "B"], 2) is None


def test_good_reversal_number_prefers_good_array():
    s = series(good_reversals=[0, 1])
    assert mod._good_reversal_number(s, 1, 9) == 1
    assert mod._good_reversal_number(s, 5, 3) == 2  # past the array: block_id - 1


def test_good_reversal_number_raises_when_blocks_does_not_reach_idx():
    # Documents a latent bug kept for backwards compatibility.
    with pytest.raises(TypeError):
        mod._good_reversal_number(series(), 3, None)


def test_get_good_reversal_info_end_to_end_from_blocks(monkeypatch):
    merged = {
        "x": dict(
            blocks=[1, 1, 2, 2, 2],
            reward_magnitudes_by_tower={t: [1] * 5 for t in "ABC"},
            choices_by_tower={t: [0, 1, 0, 1, 0] for t in "ABC"},
        )
    }
    monkeypatch.setattr(mod, "get_vars_across_all_sessions", lambda data: (merged, None))
    out = mod.get_good_reversal_info({"x": {"s": {"has_good": False}}}, pre=1)
    (rev,) = out["x"]
    assert rev["reversal_idx"] == 2 and rev["boundary_source"] == "blocks"
    assert rev["good_reversal_number"] == 1 and rev["block_id"] == 2
    assert rev["trial_window_idx"] == {"pre": [1], "post": [2, 3, 4]}
    assert rev["choices_by_tower"]["A"] == {"pre": [1], "post": [0, 1, 0]}
