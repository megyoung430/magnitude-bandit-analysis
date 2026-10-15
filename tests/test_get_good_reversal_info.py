"""Tests for get_good_reversal_info on synthetic sessions, plus edge cases of its helpers."""

import json
import os
import subprocess
import sys

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


def test_rewards_at_is_all_none_when_any_array_is_too_short():
    rewards = {"A": [1, 2, 3], "B": [4, 5]}
    assert mod._rewards_at(rewards, ["A", "B"], 1) == {"A": 2, "B": 5}
    assert mod._rewards_at(rewards, ["A", "B"], 2) == {"A": None, "B": None}


def test_skip_message_reports_each_reason_in_order():
    rewards = {"A": [0, 0, 1], "B": [0, 1, 1], "C": [1, 1]}
    s = series(blocks=[1, 1, 2], reward_magnitudes_by_tower=rewards)
    assert "missing towers" in mod._skip_message("s", 1, "blocks", s, ["A", "Z"], max_zero_count=1)
    assert "too short" in mod._skip_message("s", 3, "blocks", s, ["A", "B", "C"], max_zero_count=1)
    zero_msg = mod._skip_message("s", 1, "blocks", s, ["A", "B"], max_zero_count=0)
    assert "more than 0 zero rewards in [0, 0]" in zero_msg and "block 1" in zero_msg
    assert mod._skip_message("s", 1, "blocks", s, ["A", "B"], max_zero_count=2) is None


def test_good_reversal_number_prefers_good_array():
    s = series(good_reversals=[0, 1])
    assert mod._good_reversal_number(s, 1, 9) == 1
    assert mod._good_reversal_number(s, 5, 3) == 2  # past the array: block_id - 1


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


# --- public function, end to end on synthetic sessions (merged by the real merge code) ---

N = 15


def make_session(n=N, *, good=None, blocks=None, rewards=None, has_good=True):
    """One session whose per-trial values equal their trial index, so slices are easy to read."""
    rewards = rewards or {
        "A": [10 + i for i in range(n)],
        "B": [20 + i for i in range(n)],
        "C": [30 + i for i in range(n)],
    }
    sess = dict(
        trial=list(range(1, n + 1)),
        has_good=has_good,
        blocks=blocks if blocks is not None else [1] * n,
        reward_magnitudes_by_tower=rewards,
        choices_by_tower={t: [100 + i for i in range(n)] for t in "ABC"},
        choices_by_rank={"best": list(range(n))},
    )
    if good is not None:
        sess["good_reversals"] = good
    return sess


def test_next_kept_reversal_post_window_stops_at_skipped_boundary(capsys):
    # Boundaries 0 (first block), 5 and 10; the middle one has two zero rewards on trial 4.
    rewards = {
        "A": [10 + i for i in range(N)],
        "B": [20 + i for i in range(N)],
        "C": [30 + i for i in range(N)],
    }
    rewards["A"][4] = rewards["B"][4] = 0
    good = [0] * 5 + [1] * 5 + [2] * 5
    data = {"m": {"ses-1": make_session(good=good, rewards=rewards)}}

    first, last = mod.get_good_reversal_info(data, pre=2, post=8, include_first_block=True)["m"]

    assert "[SKIP] m boundary@5" in capsys.readouterr().out
    # post=8 would reach trial 7 for the first block; it must stop at the skipped boundary (5).
    assert first["reversal_idx"] == 0 and first["good_reversal_number"] == 0
    assert first["trial_window_idx"] == {"pre": [], "post": [0, 1, 2, 3, 4]}
    # Index 0 has no preceding trial, so before and after both describe trial 0.
    assert first["reward_magnitudes_by_tower_before"] == {"A": 10, "B": 20, "C": 30}
    assert first["reward_magnitudes_by_tower_after"] == {"A": 10, "B": 20, "C": 30}
    assert first["choices_by_rank"] == {"best": {"pre": [], "post": [0, 1, 2, 3, 4]}}
    # The reversal after the skipped one is windowed normally and clipped by the last trial.
    assert last["reversal_idx"] == 10 and last["good_reversal_number"] == 2
    assert last["trial_window_idx"] == {"pre": [8, 9], "post": [10, 11, 12, 13, 14]}
    assert last["reward_magnitudes_by_tower_before"] == {"A": 19, "B": 29, "C": 39}
    assert last["reward_magnitudes_by_tower_after"] == {"A": 20, "B": 30, "C": 40}
    assert last["choices_by_rank"] == {"best": {"pre": [8, 9], "post": [10, 11, 12, 13, 14]}}
    assert last["choices_by_tower"]["B"] == {"pre": [108, 109], "post": [110, 111, 112, 113, 114]}


def test_good_reversals_counter_takes_priority_over_block_changes():
    good = [0, 0, 0, 1, 1, 1, 1, 1]
    blocks = [1, 1, 1, 1, 1, 2, 2, 2]
    data = {"m": {"ses-1": make_session(8, good=good, blocks=blocks, has_good=True)}}
    (rev,) = mod.get_good_reversal_info(data, pre=1)["m"]
    assert rev["boundary_source"] == "good_reversals" and rev["reversal_idx"] == 3
    assert rev["good_reversal_number"] == 1 and rev["block_id"] == 1
    assert rev["towers"] == ["A", "B", "C"]
    assert rev["trial_window_idx"] == {"pre": [2], "post": [3, 4, 5, 6, 7]}

    # Without has_good the same data is windowed by block changes instead.
    data["m"]["ses-1"]["has_good"] = False
    (rev,) = mod.get_good_reversal_info(data, pre=1)["m"]
    assert rev["boundary_source"] == "blocks" and rev["reversal_idx"] == 5


def test_reversal_at_the_last_trial_has_a_one_trial_post_window():
    good = [0] * 7 + [1]
    data = {"m": {"ses-1": make_session(8, good=good)}}
    (rev,) = mod.get_good_reversal_info(data, pre=2, post=5)["m"]
    assert rev["trial_window_idx"] == {"pre": [5, 6], "post": [7]}
    assert rev["reward_magnitudes_by_tower_after"] == {"A": 17, "B": 27, "C": 37}


def test_subject_without_towers_gets_an_empty_list(capsys):
    empty = dict(
        trial=[1, 2],
        has_good=False,
        blocks=[1, 2],
        reward_magnitudes_by_tower={},
        choices_by_tower={},
        choices_by_rank={},
    )
    assert mod.get_good_reversal_info({"m": {"ses-1": empty}}) == {"m": []}
    assert "[WARN] m: no towers" in capsys.readouterr().out


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="known bug: boundaries from num_blocks with a shorter blocks array make "
    "block_id None, so the block_id - 1 fallback raises instead of yielding None",
)
def test_boundary_beyond_blocks_array_yields_no_good_reversal_number(monkeypatch):
    merged = {
        "m": dict(
            num_blocks=[0, 0, 1, 1],
            reward_magnitudes_by_tower={t: [1] * 4 for t in "ABC"},
        )
    }
    monkeypatch.setattr(mod, "get_vars_across_all_sessions", lambda data: (merged, None))
    (rev,) = mod.get_good_reversal_info({"m": {"ses-1": {}}})["m"]
    assert rev["good_reversal_number"] is None


# --- classify_towers_at_good_reversals ---


def reversal_with(before, after):
    return {
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
    }


def test_classify_towers_gives_the_remaining_tower_when_best_towers_differ():
    rev = reversal_with({"A": 4, "B": 1, "C": 0}, {"A": 0, "B": 4, "C": 1})
    assert mod.classify_towers_at_good_reversals(rev) == ("A", "B", "C")


@pytest.mark.parametrize(
    "order, expected_third",
    [(("A", "B", "C"), "B"), (("C", "B", "A"), "C"), (("B", "A", "C"), "B")],
)
def test_classify_towers_same_best_picks_first_other_tower_in_session_order(order, expected_third):
    # prev_best == next_best == A leaves two candidates; the one listed first in the
    # session's tower order is the third.
    before = {t: {"A": 4, "B": 1, "C": 0}[t] for t in order}
    after = {t: {"A": 4, "B": 0, "C": 1}[t] for t in order}
    assert mod.classify_towers_at_good_reversals(reversal_with(before, after)) == (
        "A",
        "A",
        expected_third,
    )


def test_classify_towers_raises_when_no_tower_is_left():
    rev = reversal_with({"A": 4, "B": 1}, {"A": 1, "B": 4})
    with pytest.raises(ValueError, match="no remaining tower"):
        mod.classify_towers_at_good_reversals(rev)


def test_classify_towers_does_not_depend_on_hash_seed():
    # PYTHONHASHSEED is fixed at interpreter start, so each seed needs its own process.
    code = (
        "import json;"
        "from src.behavior_analysis.get_good_reversal_info import "
        "classify_towers_at_good_reversals as f;"
        "r = {'reward_magnitudes_by_tower_before': {'T1': 4, 'T2': 1, 'T3': 0, 'T4': 0},"
        " 'reward_magnitudes_by_tower_after': {'T1': 4, 'T2': 0, 'T3': 1, 'T4': 0}};"
        "print(json.dumps(f(r)))"
    )
    results = set()
    for seed in range(8):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        ).stdout
        results.add(tuple(json.loads(out)))
    assert results == {("T1", "T1", "T2")}


# --- subjects that log bad but not good reversals ---


def test_subject_logging_only_bad_reversals_has_no_good_reversals():
    # Block changes of such a subject are bad reversals (or unrecorded), so they must not
    # stand in for good reversals.
    sess = make_session(8, blocks=[1] * 3 + [2] * 5, has_good=False)
    sess["has_bad"] = True
    sess["bad_reversals"] = [0] * 3 + [1] * 5
    assert mod.get_good_reversal_info({"m": {"ses-1": sess}}, pre=1) == {"m": []}


def test_mixed_subject_takes_good_reversals_only_from_sessions_that_log_them():
    logged = make_session(8, good=[0, 0, 0, 1, 1, 1, 1, 1], blocks=[1] * 3 + [2] * 5)
    unlogged = make_session(8, blocks=[1] * 4 + [2] * 4, has_good=False)
    unlogged["has_bad"] = True
    unlogged["bad_reversals"] = [0] * 4 + [1] * 4
    data = {"m": {"ses-1": logged, "ses-2": unlogged}}
    revs = mod.get_good_reversal_info(data, pre=1)["m"]
    assert [(r["reversal_idx"], r["boundary_source"]) for r in revs] == [(3, "good_reversals")]


def test_subject_with_no_reversal_counters_still_uses_block_changes():
    # Sessions from before the good/bad counters existed have only block numbers.
    sess = make_session(8, blocks=[1] * 5 + [2] * 3, has_good=False)
    (rev,) = mod.get_good_reversal_info({"m": {"ses-1": sess}}, pre=1)["m"]
    assert rev["boundary_source"] == "blocks" and rev["reversal_idx"] == 5
