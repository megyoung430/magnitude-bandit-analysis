"""A choice series one entry short would silently shift every later session's choices."""

import pytest

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info

TOWERS = "ABC"
RANKS = ("best", "second", "third")


def session(n):
    """A blocks-only session of ``n`` trials; the best tower rotates at each block change."""
    blocks = [1 + (i >= n // 2) for i in range(n)]
    best = [TOWERS[b - 1] for b in blocks]
    rewards = {
        t: [
            4 if best[i] == t else (1 if t == TOWERS[(TOWERS.index(best[i]) + 1) % 3] else 0)
            for i in range(n)
        ]
        for t in TOWERS
    }
    return {
        "trial": list(range(1, n + 1)),
        "blocks": blocks,
        "reward_magnitudes_by_tower": rewards,
        "choices_by_tower": {t: [int(best[i] == t) for i in range(n)] for t in TOWERS},
        "choices_by_rank": {r: [int(i % 3 == k) for i in range(n)] for k, r in enumerate(RANKS)},
    }


def two_sessions(mutate=lambda s: None):
    first, second = session(6), session(6)
    mutate(second)
    return {"m": {"ses-1": first, "ses-2": second}}


def test_aligned_choice_series_pass():
    assert get_good_reversal_info(two_sessions())["m"]


def test_a_tower_choice_series_one_short_raises_naming_the_series():
    data = two_sessions(lambda s: s["choices_by_tower"]["B"].pop())
    with pytest.raises(ValueError, match=r"m: choices_by_tower\['B'\] has 11 entries.* 12"):
        get_good_reversal_info(data)


def test_a_rank_choice_series_one_short_raises_naming_the_series():
    data = two_sessions(lambda s: s["choices_by_rank"]["second"].pop())
    with pytest.raises(ValueError, match=r"m: choices_by_rank\['second'\] has 11 entries.* 12"):
        get_good_reversal_info(data)


def test_sessions_without_choices_are_not_checked():
    # An empty series counts as "not logged", as for the other series.
    data = two_sessions()
    for sess in data["m"].values():
        sess["choices_by_tower"], sess["choices_by_rank"] = {}, {}
    assert get_good_reversal_info(data)["m"]
