"""Good reversals whose best tower does not change are dropped before any analysis sees them."""

import pytest

from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_good_reversal
from src.behavior_analysis.split_early_late_good_reversals import split_good_reversals_early_late
from src.behavior_analysis.split_good_reversals_by_best_change import (
    split_good_reversals_by_best_change,
)

N = 15
REVERSALS = [3, 6, 9, 12]
# Best tower per block. The reversals at 3 and 9 do not change it, so they are suspect.
BEST_PER_BLOCK = ["A", "A", "B", "B", "C"]


def rewards_for(best_per_block):
    """4 for the block's best tower, then 1 and 0 for the other two in alphabetical order."""
    rewards = {t: [] for t in "ABC"}
    for best in best_per_block:
        others = [t for t in "ABC" if t != best]
        for tower, value in zip([best, *others], (4, 1, 0), strict=True):
            rewards[tower].extend([value] * 3)
    return rewards


def subject_sessions():
    good = [sum(i >= r for r in REVERSALS) for i in range(N)]
    return {
        "ses-1": dict(
            trial=list(range(1, N + 1)),
            has_good=True,
            good_reversals=good,
            blocks=[1 + i // 3 for i in range(N)],
            reward_magnitudes_by_tower=rewards_for(BEST_PER_BLOCK),
            choices_by_tower={t: [1 if t == "A" else 0] * N for t in "ABC"},
            choices_by_rank={"best": [1] * N, "second": [0] * N, "third": [0] * N},
        )
    }


@pytest.fixture
def windows():
    with pytest.warns(ReversalWarning) as caught:
        info = get_good_reversal_info({"m": subject_sessions()}, pre=1)
    (warning,) = caught
    assert "dropped 2 good reversal" in str(warning.message)
    return info


def test_windows_hold_only_the_valid_reversals(windows):
    assert [r["reversal_idx"] for r in windows["m"]] == [6, 12]


def test_dropped_reversals_still_end_the_windows_of_their_neighbours(windows):
    first, second = windows["m"]
    # The suspect reversals at 3 and 9 are real block changes: they bound the windows.
    assert first["trial_window_idx"]["post"] == [6, 7, 8]
    assert second["trial_window_idx"]["post"] == [12, 13, 14]


def test_rank_counts_cover_only_the_valid_reversals(windows):
    counts = get_rank_counts_by_good_reversal(windows)
    assert len(counts["m"]) == 2


def test_early_late_split_counts_only_valid_reversals(windows):
    early, late = split_good_reversals_early_late(windows, first_n=1)
    assert [r["reversal_idx"] for r in early["m"]] == [6]
    assert [r["reversal_idx"] for r in late["m"]] == [12]


def test_best_change_split_covers_only_the_valid_reversals(windows):
    best_to_second, best_to_third = split_good_reversals_by_best_change(windows)
    kept = best_to_second["m"] + best_to_third["m"]
    assert sorted(r["reversal_idx"] for r in kept) == [6, 12]
