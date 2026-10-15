"""A subject whose sessions mix the good-counter format with blocks-only sessions raises."""

import numpy as np
import pytest

from src.behavior_analysis.cumulative_reversal_removal import removal_curves_by_kind
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_total_reversals import (
    get_reversal_indices_by_kind,
    get_unclassified_reversal_indices,
)
from src.behavior_analysis.reversal_format import check_formats_agree


def blocks_only(blocks):
    return {
        "trial": list(range(1, len(blocks) + 1)),
        "blocks": blocks,
        "reward_magnitudes_by_tower": {},
        "choices_by_tower": {},
        "choices_by_rank": {},
    }


def with_good(blocks, good):
    return {**blocks_only(blocks), "has_good": True, "good_reversals": good}


def bad_only(blocks, bad):
    return {**blocks_only(blocks), "has_bad": True, "bad_reversals": bad}


MIXED = {
    "m": {
        "ses-1_date-20260101": with_good([1, 1, 2, 2], [0, 0, 1, 1]),
        "ses-2_date-20260102": blocks_only([1, 1, 2]),
    }
}

ENTRY_POINTS = {
    "check": lambda data: check_formats_agree("m", data["m"]),
    "windows": get_good_reversal_info,
    "unclassified indices": get_unclassified_reversal_indices,
    "indices by kind": get_reversal_indices_by_kind,
}


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_mixing_good_counter_and_blocks_only_sessions_raises_at_every_entry_point(entry_point):
    with pytest.raises(ValueError) as error:
        ENTRY_POINTS[entry_point](MIXED)
    message = str(error.value)
    assert message.startswith("m: ")
    assert "good-counter sessions ['ses-1_date-20260101']" in message
    assert "blocks-only sessions ['ses-2_date-20260102']" in message


def test_the_message_lists_every_session_of_each_kind_in_session_order():
    data = {
        "m": {
            "ses-3": blocks_only([1, 2]),
            "ses-2": with_good([1, 2], [0, 1]),
            "ses-1": blocks_only([1, 2]),
            "ses-4": with_good([1, 2], [0, 1]),
        }
    }
    with pytest.raises(
        ValueError, match=r"good-counter sessions \['ses-2', 'ses-4'\].*\['ses-1', 'ses-3'\]"
    ):
        check_formats_agree("m", data["m"])


def test_one_format_per_subject_is_fine_and_other_subjects_are_not_affected():
    data = {
        "m": {"ses-1": with_good([1, 1, 2, 2], [0, 0, 1, 1]), "ses-2": with_good([1, 2], [0, 1])},
        "n": {"ses-1": blocks_only([1, 2]), "ses-2": blocks_only([1, 2])},
    }
    for subject, sessions in data.items():
        check_formats_agree(subject, sessions)
    assert get_unclassified_reversal_indices(data) == {"m": [], "n": [1, 3]}


def test_sessions_without_trials_do_not_count_as_blocks_only():
    data = {"m": {"ses-1": with_good([1, 2], [0, 1]), "ses-2": {"blocks": [], "trial": []}}}
    check_formats_agree("m", data["m"])


def test_a_bad_only_session_beside_blocks_only_sessions_is_consistent_and_allowed():
    # Without a good counter the windows and the unclassified indices both fall back to block
    # changes, masking the bad-only session's, so they agree.
    data = {
        "m": {
            "ses-1_date-20260101": bad_only([1, 1, 2, 2], [0, 0, 1, 1]),
            "ses-2_date-20260102": blocks_only([1, 1, 2]),
        }
    }
    check_formats_agree("m", data["m"])
    kinds = get_reversal_indices_by_kind(data)
    assert kinds["bad"] == {"m": [2]} and kinds["unclassified"] == {"m": [6]}


def test_a_good_counter_session_with_a_bad_only_session_is_allowed():
    data = {
        "m": {
            "ses-1_date-20260101": with_good([1, 1, 2, 2], [0, 0, 1, 1]),
            "ses-2_date-20260102": bad_only([1, 1, 2], [0, 0, 1]),
        }
    }
    check_formats_agree("m", data["m"])


# --- an unclassified reversal is never also a good or bad one ---


@pytest.mark.parametrize("kind", ["good", "bad"])
def test_removal_curves_reject_a_trial_that_is_also_an_unclassified_reversal(kind):
    windows = {"m": [{"reversal_idx": 1}]}
    indices = {"good_idx": {"m": []}, "bad_idx": {"m": []}, "unclassified_idx": {"m": [5]}}
    indices[f"{kind}_idx"] = {"m": [5]}
    with pytest.raises(ValueError, match=rf"m: {kind} and unclassified reversals share trials"):
        removal_curves_by_kind(windows, np.arange(10), **indices)


def test_a_good_and_a_bad_event_on_one_trial_count_once_as_good():
    # Events of both kinds can enter the counters on the same trial; the anchor is removed
    # once, and the good kind takes it.
    windows = {"m": [{"reversal_idx": 1}]}
    curves = removal_curves_by_kind(
        windows, np.arange(10), good_idx={"m": [5]}, bad_idx={"m": [5]}, unclassified_idx={"m": []}
    )
    assert curves["good"][0][-1] == 1 and curves["bad"][0][-1] == 0


def test_removal_curves_accept_disjoint_kinds():
    windows = {"m": [{"reversal_idx": 1}]}
    curves = removal_curves_by_kind(
        windows,
        np.arange(10),
        good_idx={"m": [5]},
        bad_idx={"m": [7]},
        unclassified_idx={"m": [9]},
    )
    assert curves["good"][0][-1] == 1
