"""Post-only analyses must not depend on how many pre-window trials exist.

A reversal near the start of the data has a short (or empty) pre window.
First-leave and rank counts read only the post window, indexed from the
reversal trial, so the pre length must not change them.
"""

import pytest

from src.behavior_analysis.get_first_leave_after_good_reversals import (
    get_first_leave_after_good_reversals,
)
from src.behavior_analysis.get_rank_counts_by_good_reversal import (
    get_rank_counts_by_good_reversal,
)

TOWERS = ("A", "B", "C")


def reversal(n_pre, post):
    """Good reversal A -> B whose pre window has *n_pre* trials on A and post lists *post*."""
    return {
        "reward_magnitudes_by_tower_before": {"A": 4, "B": 0, "C": 0},
        "reward_magnitudes_by_tower_after": {"A": 0, "B": 4, "C": 0},
        "choices_by_tower": {
            t: {"pre": [int(t == "A")] * n_pre, "post": [int(c == t) for c in post]} for t in TOWERS
        },
        "choices_by_rank": {
            rank: {"pre": [1] * n_pre, "post": [int(c == t) for c in post]}
            for rank, t in zip(("best", "second", "third"), ("B", "A", "C"), strict=True)
        },
    }


@pytest.mark.parametrize("n_pre", [0, 3, 10])
def test_first_leave_does_not_depend_on_the_pre_window_length(n_pre):
    windows = {"m": [reversal(n_pre, ["A", "A", "B"]), reversal(n_pre, ["A", "C"])]}
    assert get_first_leave_after_good_reversals(windows) == {
        "m": {"new_best": 1, "third": 1, "total": 2}
    }


@pytest.mark.parametrize("n_pre", [0, 3, 10])
def test_rank_counts_do_not_depend_on_the_pre_window_length(n_pre):
    (row,) = get_rank_counts_by_good_reversal({"m": [reversal(n_pre, ["A", "B", "B", "C"])]})["m"]
    assert (row["best"], row["second"], row["third"], row["n_trials"]) == (2, 1, 1, 4)
