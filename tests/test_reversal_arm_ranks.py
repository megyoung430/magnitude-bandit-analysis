"""Tests for ranking the arms around a reversal, labelling them and typing reversals."""

import pytest

from src.behavior_analysis.reversal_arm_ranks import (
    arm_labels,
    classify_reversal_type,
    consistent_rankings,
    select_reversals,
)


def reversal(before, after, **extra):
    return {
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
        **extra,
    }


def abc(a, b, c):
    return {"A": a, "B": b, "C": c}


# -------------------------------------------------------------------- rankings


def test_consistent_rankings_is_a_single_order_without_ties():
    assert consistent_rankings(abc(1, 3, 2)) == [("B", "C", "A")]


def test_consistent_rankings_lists_every_order_a_tie_allows():
    assert sorted(consistent_rankings(abc(1, 2, 2))) == [("B", "C", "A"), ("C", "B", "A")]
    assert len(consistent_rankings(abc(2, 2, 2))) == 6


def test_consistent_rankings_rejects_a_missing_magnitude():
    with pytest.raises(ValueError, match="magnitude"):
        consistent_rankings(abc(1, None, 2))


def test_consistent_rankings_needs_three_arms():
    with pytest.raises(ValueError, match="three arms"):
        consistent_rankings({"A": 1, "B": 2})


# ------------------------------------------------------------------ arm labels


def test_prev_labels_follow_the_ranking_before_and_new_labels_the_ranking_after():
    rev = reversal(abc(3, 2, 1), abc(1, 2, 3))
    assert arm_labels(rev, "prev") == ("A", "B", "C")
    assert arm_labels(rev, "new") == ("C", "B", "A")


def test_labels_are_none_when_the_ranking_the_scheme_uses_has_a_tie():
    rev = reversal(abc(3, 2, 2), abc(1, 2, 3))
    assert arm_labels(rev, "prev") is None  # B and C are tied for second before
    assert arm_labels(rev, "new") == ("C", "B", "A")  # the tie does not touch the after ranking


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="scheme"):
        arm_labels(reversal(abc(3, 2, 1), abc(1, 2, 3)), "old")


# ------------------------------------------------------------ reversal types


@pytest.mark.parametrize(
    "before, after, expected",
    [
        (abc(3, 2, 1), abc(1, 2, 3), "best_to_third"),  # A: best -> third
        (abc(3, 2, 1), abc(2, 1, 3), "second_to_third"),  # B: second -> third, A -> second
        (abc(3, 2, 1), abc(1, 3, 2), "best_to_third"),
        (abc(3, 2, 1), abc(2, 3, 1), "other"),  # C stays third
    ],
)
def test_classify_reversal_type_by_the_prev_ranking(before, after, expected):
    assert classify_reversal_type(reversal(before, after)) == expected


def test_a_tie_that_does_not_change_the_type_is_not_ambiguous():
    # B and C tie for second before, but A (the unique best) becomes third either way.
    assert classify_reversal_type(reversal(abc(3, 2, 2), abc(1, 3, 2))) == "best_to_third"


def test_a_tie_that_changes_the_type_is_ambiguous():
    # B and C tie for second before; C becomes third after. By arm name B would be second and
    # the type "other"; with C second it is "second_to_third".
    rev = reversal(abc(3, 2, 2), abc(2, 3, 1))
    assert classify_reversal_type(rev) == "ambiguous"


def test_a_tie_for_third_after_can_make_the_type_ambiguous():
    # A is best before; B and C tie for the lowest value after, so the third arm is B or C.
    assert classify_reversal_type(reversal(abc(3, 2, 1), abc(3, 1, 1))) == "ambiguous"


# ------------------------------------------------------------------- selection


def windows_for_selection():
    return {
        "m1": [
            reversal(abc(3, 2, 1), abc(1, 2, 3)),  # best_to_third, clean
            reversal(abc(3, 2, 1), abc(2, 1, 3)),  # second_to_third, clean
            reversal(abc(3, 2, 2), abc(2, 3, 1)),  # tie before: prev labels and type ambiguous
            reversal(abc(3, 2, 1), abc(2, 3, 2)),  # tie after: new labels ambiguous, type fine
        ],
        "m2": [reversal(abc(3, 2, 1), abc(2, 3, 1))],  # other
    }


def test_select_all_drops_only_reversals_whose_scheme_labels_are_tied():
    prev, counts_prev = select_reversals(windows_for_selection(), "prev", "all")
    new, counts_new = select_reversals(windows_for_selection(), "new", "all")
    assert [len(v) for v in prev.values()] == [3, 1]
    assert [len(v) for v in new.values()] == [3, 1]
    assert counts_prev.candidates == counts_new.candidates == 5
    assert counts_prev.tied_labels == counts_new.tied_labels == 1
    assert counts_prev.tied_type == counts_new.tied_type == 0
    assert counts_prev.selected == counts_new.selected == 4


def test_select_type_counts_labels_ties_then_type_ties_then_other_types():
    sel, counts = select_reversals(windows_for_selection(), "new", "second_to_third")
    # tied labels (new): the 4th reversal; tied type: the 3rd; other type: best_to_third and other
    assert tuple(counts) == (
        5,
        1,
        1,
        2,
        1,
    )  # candidates, tied_labels, tied_type, wrong_type, selected
    assert list(sel) == ["m1"] and len(sel["m1"]) == 1


def test_select_omits_subjects_left_without_reversals():
    sel, _ = select_reversals(windows_for_selection(), "prev", "best_to_third")
    assert list(sel) == ["m1"]


def test_select_rejects_the_first_block_record_and_unknown_types():
    with pytest.raises(ValueError, match="first block"):
        select_reversals(
            {"m": [reversal(abc(3, 2, 1), abc(3, 2, 1), first_block=True)]}, "prev", "all"
        )
    with pytest.raises(ValueError, match="reversal_type"):
        select_reversals({"m": []}, "prev", "third_to_best")
