"""The boundary description: what a change logged between two recordings looks like.

A boundary runs from the last trial of one recording to the first trial of the next (a later
recording of the same session, or the first recording of the next session). It is *consistent*
when the counters and the reward magnitudes tell the same story, *no change* when neither shows
a change, *inconsistent* when only one does, and *unverifiable* when the magnitudes needed to
tell are missing.
"""

import pytest

from src.behavior_import.run_boundaries import (
    CONSISTENT,
    INCONSISTENT,
    NO_CHANGE,
    UNVERIFIABLE,
    best_changed,
    describe_boundary,
    event_matches,
    magnitude_change,
)
from tests.boundary_fixtures import (
    B1_BEST,
    B3_BEST,
    SECOND_SWAPPED,
    ZERO_B1,
    ZERO_B3,
    ZERO_B3_DRIFTED,
)


def side(mags, **totals):
    """One side of a boundary; ``blocks=2`` is the block total, ``good=1`` the good total."""
    keys = {"blocks": "block", "good": "good", "bad": "bad"}
    full = {"block": None, "good": None, "bad": None}
    full.update({keys[k]: v for k, v in totals.items()})
    return {"magnitudes": mags, "totals": full, "counters": dict(full)}


# --- the magnitude rules ---


def test_drifting_block_change_is_a_move_of_the_zero_arm():
    assert magnitude_change(ZERO_B3, ZERO_B1, "drifting") is True


def test_drifting_drift_of_the_other_arms_is_not_a_block_change():
    assert magnitude_change(ZERO_B3, ZERO_B3_DRIFTED, "drifting") is False


def test_blocked_block_change_is_any_permutation():
    assert magnitude_change(B1_BEST, SECOND_SWAPPED, "blocked") is True
    assert magnitude_change(B1_BEST, dict(B1_BEST), "blocked") is False


@pytest.mark.parametrize(
    "before, after, mode",
    [
        (None, ZERO_B1, "drifting"),
        (ZERO_B3, {"B1": 1, "B3": None, "C2": 2}, "drifting"),
        (ZERO_B3, {"B1": 1, "C2": 2}, "drifting"),  # different towers
        (ZERO_B3, ZERO_B1, None),  # task type unknown
    ],
)
def test_magnitude_change_is_unknown_when_it_cannot_be_told(before, after, mode):
    assert magnitude_change(before, after, mode) is None


def test_best_changed_follows_the_first_listed_best_tower():
    assert best_changed(B1_BEST, B3_BEST) is True
    assert best_changed(B1_BEST, SECOND_SWAPPED) is False
    assert best_changed(B1_BEST, {"B1": 4, "B3": 4, "C2": 0}) is False
    assert best_changed(B1_BEST, None) is None


@pytest.mark.parametrize(
    "kind, after, expected",
    [
        ("good", B3_BEST, True),
        ("good", SECOND_SWAPPED, False),
        ("bad", SECOND_SWAPPED, True),
        ("bad", dict(B1_BEST), False),
        ("block", B3_BEST, True),
    ],
)
def test_event_matches_uses_the_rule_of_its_kind(kind, after, expected):
    assert event_matches(kind, B1_BEST, after, "blocked") is expected


def test_a_drifting_block_event_matches_a_zero_arm_move_only():
    assert event_matches("block", ZERO_B3, ZERO_B1, "drifting") is True
    assert event_matches("block", ZERO_B3, ZERO_B3_DRIFTED, "drifting") is False


# --- the boundary description, drifting ---


def test_drifting_logged_block_and_moved_zero_arm_are_consistent():
    out = describe_boundary(side(ZERO_B3, blocks=1), side(ZERO_B1, blocks=2), "drifting")
    assert out["verdict"] == CONSISTENT
    assert out["source"] == "totals"
    assert out["logged_changes"] == 1
    assert out["total_delta"] == {"block": 1, "good": None, "bad": None}
    assert out["magnitudes_before"] == ZERO_B3 and out["magnitudes_after"] == ZERO_B1


def test_drifting_logged_block_with_an_unmoved_zero_arm_is_inconsistent():
    out = describe_boundary(side(ZERO_B3, blocks=1), side(ZERO_B3_DRIFTED, blocks=2), "drifting")
    assert out["verdict"] == INCONSISTENT
    assert "zero arm" in out["reason"]


def test_drifting_moved_zero_arm_without_a_logged_block_is_inconsistent():
    out = describe_boundary(side(ZERO_B3, blocks=2), side(ZERO_B1, blocks=2), "drifting")
    assert out["verdict"] == INCONSISTENT
    assert out["logged_changes"] == 0


def test_drifting_restart_with_no_change_is_no_change():
    out = describe_boundary(side(ZERO_B3, blocks=2), side(ZERO_B3_DRIFTED, blocks=2), "drifting")
    assert out["verdict"] == NO_CHANGE


def test_counter_delta_is_reported_beside_the_total_delta():
    # The per-recording counter restarts at 1 in the next recording; the total does not.
    before = {**side(ZERO_B3, blocks=2), "counters": {"block": 2, "good": None, "bad": None}}
    after = {**side(ZERO_B1, blocks=3), "counters": {"block": 1, "good": None, "bad": None}}
    out = describe_boundary(before, after, "drifting")
    assert out["counter_delta"]["block"] == -1
    assert out["total_delta"]["block"] == 1
    assert out["verdict"] == CONSISTENT


def test_without_totals_a_moved_zero_arm_is_the_evidence():
    out = describe_boundary(side(ZERO_B3), side(ZERO_B1), "drifting")
    assert (out["verdict"], out["source"], out["logged_changes"]) == (
        CONSISTENT,
        "magnitudes",
        None,
    )


def test_without_totals_an_unmoved_zero_arm_is_no_change():
    out = describe_boundary(side(ZERO_B3), side(ZERO_B3_DRIFTED), "drifting")
    assert (out["verdict"], out["source"]) == (NO_CHANGE, "magnitudes")


def test_a_counter_that_goes_back_is_inconsistent():
    out = describe_boundary(side(ZERO_B3, blocks=3), side(ZERO_B3, blocks=2), "drifting")
    assert out["verdict"] == INCONSISTENT
    assert "rolled back" in out["reason"]


def test_missing_magnitudes_cannot_be_verified():
    out = describe_boundary(side(None, blocks=1), side(ZERO_B1, blocks=2), "drifting")
    assert out["verdict"] == UNVERIFIABLE


# --- the boundary description, blocked ---


def test_blocked_good_reversal_that_changes_the_best_tower_is_consistent():
    out = describe_boundary(side(B1_BEST, good=0, bad=0), side(B3_BEST, good=1, bad=0), "blocked")
    assert out["verdict"] == CONSISTENT
    assert out["total_delta"] == {"block": None, "good": 1, "bad": 0}


def test_blocked_good_reversal_that_leaves_the_best_tower_is_inconsistent():
    out = describe_boundary(
        side(B1_BEST, good=0, bad=0), side(SECOND_SWAPPED, good=1, bad=0), "blocked"
    )
    assert out["verdict"] == INCONSISTENT
    assert "best tower" in out["reason"]


def test_blocked_bad_reversal_needs_a_permutation_only():
    out = describe_boundary(
        side(B1_BEST, good=0, bad=0), side(SECOND_SWAPPED, good=0, bad=1), "blocked"
    )
    assert out["verdict"] == CONSISTENT


def test_blocked_without_totals_a_permutation_is_the_evidence():
    out = describe_boundary(side(B1_BEST), side(B3_BEST), "blocked")
    assert (out["verdict"], out["source"]) == (CONSISTENT, "magnitudes")
    assert describe_boundary(side(B1_BEST), side(dict(B1_BEST)), "blocked")["verdict"] == NO_CHANGE
