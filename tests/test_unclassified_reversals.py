"""Reversals of unknown kind: block changes in sessions that log no good/bad counters."""

from src.behavior_analysis.get_total_reversals import (
    get_all_reversal_indices,
    get_reversal_indices_by_kind,
    get_unclassified_reversal_indices,
)


def blocks_only(blocks):
    return {
        "trial": list(range(1, len(blocks) + 1)),
        "blocks": blocks,
        "reward_magnitudes_by_tower": {},
        "choices_by_tower": {},
        "choices_by_rank": {},
    }


def with_counters(blocks, good):
    return {**blocks_only(blocks), "has_good": True, "good_reversals": good}


def test_blocks_only_session_reports_every_block_change():
    data = {"m": {"ses-1": blocks_only([1, 1, 2, 2, 2, 3])}}
    assert get_unclassified_reversal_indices(data) == {"m": [2, 5]}
    assert get_all_reversal_indices(data)[:2] == ({"m": []}, {"m": []})  # nothing is good or bad


def test_sessions_with_counters_contribute_no_unclassified_reversals():
    data = {"m": {"ses-1": with_counters([1, 1, 2, 2], [0, 0, 1, 1])}}
    assert get_unclassified_reversal_indices(data) == {"m": []}


def test_mixed_sessions_are_classified_per_session_on_the_merged_trial_axis():
    # the counters session ends in block 2, so the blocks-only one continues from it
    data = {
        "m": {
            "ses-2": blocks_only([1, 1, 2, 2]),
            "ses-1": with_counters([1, 1, 2, 2], [0, 0, 1, 1]),
        },
        "n": {"ses-1": blocks_only([1, 2])},
    }
    kinds = get_reversal_indices_by_kind(data)
    assert kinds["good"] == {"m": [2], "n": []}
    assert kinds["bad"] == {"m": [], "n": []}
    assert kinds["unclassified"] == {"m": [6], "n": [1]}


def test_sessions_are_ordered_by_number_not_by_name():
    data = {"m": {"ses-10": blocks_only([1, 2]), "ses-2": with_counters([1, 1], [0, 0])}}
    # ses-2 comes first (2 trials), so ses-10's change sits at merged index 3
    assert get_unclassified_reversal_indices(data) == {"m": [3]}
