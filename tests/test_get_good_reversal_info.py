"""Tests for get_good_reversal_info on synthetic sessions, plus edge cases of its helpers."""

import json
import os
import subprocess
import sys
import warnings

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
    assert mod._find_boundaries(series(**fields), any_good, []) == expected


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


def test_classify_towers_invents_no_third_when_the_best_tower_is_unchanged():
    rev = reversal_with({"A": 4, "B": 1, "C": 0}, {"A": 4, "B": 0, "C": 1})
    assert mod.classify_towers_at_good_reversals(rev) == ("A", "A", None)


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
        "same = {'reward_magnitudes_by_tower_before': {'T1': 4, 'T2': 1, 'T3': 0, 'T4': 0},"
        " 'reward_magnitudes_by_tower_after': {'T1': 4, 'T2': 0, 'T3': 1, 'T4': 0}};"
        "diff = {'reward_magnitudes_by_tower_before': {'T1': 4, 'T2': 1, 'T3': 0},"
        " 'reward_magnitudes_by_tower_after': {'T1': 0, 'T2': 4, 'T3': 1}};"
        "print(json.dumps([f(same), f(diff)]))"
    )
    results = set()
    for seed in range(8):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        ).stdout
        results.add(out)
    assert len(results) == 1
    same, diff = json.loads(results.pop())
    assert same == ["T1", "T1", None]
    assert diff == ["T1", "T2", "T3"]


# --- records whose best tower does not change ---


def rewards_swapping_best_at(trial, n=N):
    """A is best before *trial* and B is best from it on."""
    a = [4 if i < trial else 0 for i in range(n)]
    b = [0 if i < trial else 4 for i in range(n)]
    return {"A": a, "B": b, "C": [1] * n}


def test_reversal_where_best_tower_changes_is_not_flagged():
    sess = make_session(
        8, good=[0] * 4 + [1] * 4, rewards=rewards_swapping_best_at(4, 8), blocks=[1] * 8
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (rev,) = mod.get_good_reversal_info({"m": {"ses-1": sess}}, pre=1)["m"]
    assert rev["best_unchanged"] is False and rev["first_block"] is False
    assert mod.classify_towers_at_good_reversals(rev) == ("A", "B", "C")


def test_reversal_where_best_tower_is_unchanged_is_flagged_and_warns():
    # Default rewards keep C best throughout, so the reversal at trial 3 of ses-2 does not
    # change the best tower. ses-1 has 5 trials, so its merged index is 8.
    first = make_session(5, good=[0] * 5, blocks=[1] * 5)
    second = make_session(8, good=[0] * 3 + [1] * 5, blocks=[1] * 8)
    data = {"m": {"ses-1": first, "ses-2": second}}
    with pytest.warns(UserWarning, match=r"m.*ses-2.*trial 3.*best tower.*unchanged") as caught:
        (rev,) = mod.get_good_reversal_info(data, pre=1)["m"]
    assert len(caught) == 1
    assert rev["reversal_idx"] == 8
    assert rev["best_unchanged"] is True and rev["first_block"] is False
    assert mod.classify_towers_at_good_reversals(rev)[2] is None


def test_first_block_record_is_not_flagged_even_though_its_best_is_unchanged():
    # Trial 0 supplies both "before" and "after", so prev_best == next_best by construction.
    sess = make_session(
        8, good=[0] * 4 + [1] * 4, rewards=rewards_swapping_best_at(4, 8), blocks=[1] * 8
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first, second = mod.get_good_reversal_info(
            {"m": {"ses-1": sess}}, pre=1, include_first_block=True
        )["m"]
    assert first["reversal_idx"] == 0
    assert first["first_block"] is True and first["best_unchanged"] is False
    assert mod.classify_towers_at_good_reversals(first) == ("A", "A", None)
    assert second["first_block"] is False and second["best_unchanged"] is False


# --- sessions that log bad but not good reversals ---


def bad_only_session(n, blocks):
    sess = make_session(n, blocks=blocks, has_good=False)
    sess["has_bad"] = True
    sess["bad_reversals"] = [b - 1 for b in blocks]
    return sess


def boundaries(data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        revs = mod.get_good_reversal_info(data, pre=1, **kwargs)["m"]
    return [(r["reversal_idx"], r["boundary_source"]) for r in revs]


def test_subject_logging_only_bad_reversals_has_no_good_reversals():
    # Block changes of such a subject are bad reversals, not good ones.
    data = {"m": {"ses-1": bad_only_session(8, [1] * 3 + [2] * 5)}}
    assert boundaries(data) == []


def test_bad_only_session_warns_with_subject_and_session():
    data = {"m": {"ses-1": bad_only_session(8, [1] * 3 + [2] * 5)}}
    with pytest.warns(UserWarning, match=r"m.*ses-1.*no good reversals"):
        mod.get_good_reversal_info(data, pre=1)


def test_legacy_session_keeps_block_changes_next_to_a_bad_only_session():
    legacy = make_session(8, blocks=[1] * 5 + [2] * 3, has_good=False)
    for first, second, expected in [
        (legacy, bad_only_session(8, [1] * 3 + [2] * 5), [(5, "blocks")]),
        (bad_only_session(8, [1] * 3 + [2] * 5), legacy, [(13, "blocks")]),
    ]:
        data = {"m": {"ses-1": first, "ses-2": second}}
        assert boundaries(data) == expected
        with pytest.warns(UserWarning, match="no good reversals"):
            mod.get_good_reversal_info(data, pre=1)


def test_bad_only_session_yields_no_first_block_record():
    # The first block of a bad-only session is bounded by a bad reversal, which this
    # function does not treat as a boundary, so it has no valid post window.
    data = {"m": {"ses-1": bad_only_session(8, [1] * 3 + [2] * 5)}}
    assert boundaries(data, include_first_block=True) == []


def test_first_block_record_is_kept_for_a_legacy_session_beside_a_bad_only_one():
    legacy = make_session(8, blocks=[1] * 5 + [2] * 3, has_good=False)
    data = {"m": {"ses-1": legacy, "ses-2": bad_only_session(8, [1] * 3 + [2] * 5)}}
    assert boundaries(data, include_first_block=True) == [(0, "blocks"), (5, "blocks")]


def test_mixed_subject_takes_good_reversals_only_from_sessions_that_log_them():
    logged = make_session(8, good=[0, 0, 0, 1, 1, 1, 1, 1], blocks=[1] * 3 + [2] * 5)
    data = {"m": {"ses-1": logged, "ses-2": bad_only_session(8, [1] * 4 + [2] * 4)}}
    assert boundaries(data) == [(3, "good_reversals")]
    with pytest.warns(UserWarning, match=r"m.*ses-2.*no good reversals"):
        mod.get_good_reversal_info(data, pre=1)


def test_subject_with_no_reversal_counters_still_uses_block_changes():
    # Sessions from before the good/bad counters existed have only block numbers.
    sess = make_session(
        8, blocks=[1] * 5 + [2] * 3, rewards=rewards_swapping_best_at(5, 8), has_good=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (rev,) = mod.get_good_reversal_info({"m": {"ses-1": sess}}, pre=1)["m"]
    assert rev["boundary_source"] == "blocks" and rev["reversal_idx"] == 5


def test_find_boundaries_requires_the_masked_spans():
    with pytest.raises(TypeError):
        mod._find_boundaries(series(blocks=[1, 1, 2]), False)


def test_find_boundaries_drops_block_changes_inside_masked_spans():
    s = series(blocks=[1, 1, 2, 2, 3, 3])
    assert mod._find_boundaries(s, False, [(0, 3)]) == ([4], "blocks")
    # The good counter is not masked: only sessions without it can fall back to blocks.
    s = series(good_reversals=[0, 1, 1], blocks=[1, 2, 2])
    assert mod._find_boundaries(s, True, [(0, 3)]) == ([1], "good_reversals")
