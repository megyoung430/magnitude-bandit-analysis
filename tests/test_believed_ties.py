"""Believed-rank shares: ties split evenly among the tied arms, independent of arm names."""

import pytest

from src.behavior_analysis.get_believed_value import (
    compute_believed_value,
    rank_weights,
    tie_category,
)


def shares(offers, choices):
    """(best_share, second_share) of the last trial's chosen arm."""
    last = compute_believed_value(choices, offers)[-1]
    return last["believed_best_share"], last["believed_second_share"]


def test_unseen_chosen_arm_has_no_shares():
    first = compute_believed_value(["A"], [{"A": 3, "B": 2, "C": 1}])[0]
    assert (first["believed_best_share"], first["believed_second_share"]) == (None, None)


def test_untied_choices_get_all_or_nothing_shares():
    offers = [{"A": 4, "B": 3, "C": 1}] * 4
    assert shares(offers, list("ABCA")) == (1.0, 0.0)
    assert shares(offers, list("ABCB")) == (0.0, 1.0)
    assert shares(offers, list("ABCC")) == (0.0, 0.0)


@pytest.mark.parametrize("names", [("A", "B", "C"), ("C", "B", "A"), ("B", "C", "A")])
def test_second_third_tie_splits_second_evenly_whatever_the_arm_names(names):
    top, tie1, tie2 = names
    offers = [{top: 4, tie1: 2, tie2: 2}] * 6
    for tied_arm in (tie1, tie2):
        assert shares(offers, [top, tie1, tie2, top, tie1, tied_arm]) == (0.0, 0.5)
    assert shares(offers, [top, tie1, tie2, top, tie1, top]) == (1.0, 0.0)


@pytest.mark.parametrize("chosen", ["A", "B", "C"])
def test_three_way_tie_gives_every_arm_one_third_of_best_and_of_second(chosen):
    offers = [{"A": 3, "B": 3, "C": 3}] * 4
    best, second = shares(offers, ["A", "B", "C", chosen])
    assert best == pytest.approx(1 / 3) and second == pytest.approx(1 / 3)


def test_best_second_tie_splits_best_and_second_evenly():
    offers = [{"A": 3, "B": 3, "C": 1}] * 4
    assert shares(offers, list("ABCA")) == (0.5, 0.5)
    assert shares(offers, list("ABCB")) == (0.5, 0.5)
    assert shares(offers, list("ABCC")) == (0.0, 0.0)


@pytest.mark.parametrize(
    "magnitudes, expected",
    [
        ([4, 3, 1], None),
        ([3, 3, 1], "best_second"),
        ([4, 2, 2], "second_third"),
        ([3, 3, 3], "all_three"),
        ([2, 2], "best_second"),
        ([4, 3], None),
        ([4], None),
        ([4, 3, 1, 1], None),  # a tie below the third value does not count
        ([4, 4, 3, 3], "best_second"),
    ],
)
def test_tie_category_names_which_top_three_values_tie(magnitudes, expected):
    assert tie_category(magnitudes) == expected


def test_believed_tie_is_reported_for_the_believed_values():
    offers = [{"A": 3, "B": 3, "C": 1}] * 4
    assert compute_believed_value(list("ABCA"), offers)[-1]["believed_tie"] == "best_second"
    assert compute_believed_value(["A"], offers[:1])[0]["believed_tie"] is None


@pytest.mark.parametrize(
    "chosen, magnitudes, expected",
    [
        (4, [4, 3, 1], (1.0, 0.0, 0.0)),
        (1, [4, 3, 1], (0.0, 0.0, 1.0)),
        (2, [4, 2, 2], (0.0, 0.5, 0.5)),
        (3, [3, 3, 3], (1 / 3, 1 / 3, 1 / 3)),
        (3, [3, 3], (0.5, 0.5, 0.0)),
    ],
)
def test_rank_weights_cover_all_three_positions(chosen, magnitudes, expected):
    assert rank_weights(chosen, magnitudes) == pytest.approx(expected)
