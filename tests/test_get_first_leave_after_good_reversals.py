"""First-leave counting: which arm is left to, and which reversals are excluded."""

import pytest

from src.behavior_analysis.get_first_leave_after_good_reversals import (
    get_first_leave_after_good_reversals,
)


def reversal(prev_best, next_best, first_choices, **flags):
    """Reversal over towers A, B, C whose post window holds *first_choices* (tower names)."""
    towers = ("A", "B", "C")
    before = {t: 4 if t == prev_best else 0 for t in towers}
    after = {t: 4 if t == next_best else 0 for t in towers}
    return {
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
        "choices_by_tower": {
            t: {"pre": [], "post": [int(c == t) for c in first_choices]} for t in towers
        },
        **flags,
    }


def test_first_leave_counts_new_best_and_third():
    windows = {
        "mouse1": [
            reversal("A", "B", ["A", "B"]),  # leaves to the new best
            reversal("A", "B", ["A", "C"]),  # leaves to the third tower
            reversal("A", "B", ["A", "A"]),  # never leaves
        ]
    }
    assert get_first_leave_after_good_reversals(windows) == {
        "mouse1": {"new_best": 1, "third": 1, "total": 3}
    }


def test_reversals_with_unchanged_best_tower_are_excluded_and_counted():
    windows = {
        "mouse1": [
            reversal("A", "B", ["A", "B"]),
            reversal("A", "A", ["A", "B"], best_unchanged=True),
            reversal("A", "A", ["A", "C"], best_unchanged=True),
        ],
        "mouse2": [reversal("A", "A", ["A", "B"], best_unchanged=True)],
    }
    with pytest.warns(UserWarning) as caught:
        out = get_first_leave_after_good_reversals(windows)
    messages = [str(w.message) for w in caught]
    assert any("excluded 2 reversal" in m and "mouse1" in m and "unchanged" in m for m in messages)
    assert any("excluded 1 reversal" in m and "mouse2" in m for m in messages)
    assert out == {
        "mouse1": {"new_best": 1, "third": 0, "total": 1},
        "mouse2": {"new_best": 0, "third": 0, "total": 0},
    }


def test_unflagged_hand_built_reversal_with_unchanged_best_is_still_excluded():
    # The exclusion follows the classification, not only the flag on the record.
    windows = {"mouse1": [reversal("A", "A", ["A", "B"])]}
    with pytest.warns(UserWarning, match="excluded 1 reversal"):
        out = get_first_leave_after_good_reversals(windows)
    assert out == {"mouse1": {"new_best": 0, "third": 0, "total": 0}}


def test_first_block_records_are_dropped_without_a_warning(recwarn):
    windows = {"mouse1": [reversal("A", "A", ["A", "B"], first_block=True)]}
    out = get_first_leave_after_good_reversals(windows)
    assert out == {"mouse1": {"new_best": 0, "third": 0, "total": 0}}
    assert len(recwarn) == 0
