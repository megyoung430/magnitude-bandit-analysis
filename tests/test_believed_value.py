import pytest

from src.behavior_analysis.get_believed_value import compute_believed_value


def beliefs(choices, magnitudes, field):
    return [t[field] for t in compute_believed_value(choices, magnitudes)]


def test_nothing_is_believed_before_the_first_choice():
    first = compute_believed_value(["A"], [{"A": 3, "B": 2, "C": 1}])[0]
    assert first == {
        "believed_best_mag": None,
        "believed_second_mag": None,
        "believed_chosen_rank": None,
        "tied": False,
        "n_seen": 0,
    }


def test_belief_is_the_last_value_seen_not_the_current_one():
    offers = [{"A": 4, "B": 0}, {"A": 1, "B": 0}, {"A": 0, "B": 0}]
    # A is seen as 4 on trial 0 and as 1 on trial 1; the value on offer now never counts
    assert beliefs(["A", "A", "B"], offers, "believed_best_mag") == [None, 4, 1]


def test_only_arms_chosen_before_are_seen_and_ranked():
    offers = [{"A": 3, "B": 2, "C": 1}] * 3
    trials = compute_believed_value(["A", "B", "C"], offers)
    assert [t["n_seen"] for t in trials] == [0, 1, 2]
    assert trials[1]["believed_chosen_rank"] is None  # B is chosen before it is seen
    assert (trials[1]["believed_best_mag"], trials[1]["believed_second_mag"]) == (3, None)
    assert (trials[2]["believed_best_mag"], trials[2]["believed_second_mag"]) == (3, 2)


def test_chosen_arm_rank_among_seen_arms():
    offers = [{"A": 1, "B": 4, "C": 2, "D": 3}] * 8
    ranks = beliefs(list("ABCDABCD"), offers, "believed_chosen_rank")
    assert ranks[:4] == [None] * 4  # each arm is unseen when first chosen
    # believed order is B (4), D (3), C (2), A (1): rank follows value, not arm name,
    # and the fourth place is still labelled "third"
    assert ranks[4:] == ["third", "best", "third", "second"]


def test_ties_are_flagged_and_broken_by_arm_name():
    offers = [{"A": 3, "B": 3, "C": 0}] * 4
    trials = compute_believed_value(["B", "A", "B", "A"], offers)
    assert [t["tied"] for t in trials] == [False, False, True, True]
    assert [t["believed_chosen_rank"] for t in trials][2:] == ["second", "best"]


def test_untied_trials_report_best_above_second():
    offers = [{"A": 4, "B": 2, "C": 0}] * 3
    third = compute_believed_value(["A", "B", "C"], offers)[2]
    assert (third["believed_best_mag"], third["believed_second_mag"], third["tied"]) == (
        4,
        2,
        False,
    )


def test_unchosen_or_unoffered_choices_leave_beliefs_unchanged():
    offers = [{"A": 4}, {"A": 2}, {"A": 1}]
    # None, "" and an arm that is not on offer record nothing
    assert beliefs([None, "", "Z"], offers, "n_seen") == [0, 0, 0]
    assert beliefs(["A", "A", "A"], offers, "n_seen") == [0, 1, 1]


def test_seen_arms_missing_from_the_current_offer_are_not_believed():
    offers = [{"A": 4, "B": 2}, {"A": 4, "B": 2}, {"A": 4}]
    # by trial 2 both arms were chosen, but only A is on offer then
    assert beliefs(["A", "B", "A"], offers, "n_seen") == [0, 1, 1]
    assert beliefs(["A", "B", "A"], offers, "believed_second_mag") == [None, None, None]


def test_one_result_per_offer_and_extra_choices_are_ignored():
    assert compute_believed_value([], []) == []
    assert len(compute_believed_value(["A", "B", "C"], [{"A": 1}])) == 1


def test_fewer_choices_than_offers_raises():
    with pytest.raises(IndexError):
        compute_believed_value(["A"], [{"A": 1}, {"A": 2}])


def test_an_empty_choice_is_never_recorded_as_seen():
    # "" stands for no choice, even if an offer happens to contain that key
    offers = [{"": 3, "A": 1}] * 3
    assert beliefs(["", "A", "A"], offers, "n_seen") == [0, 0, 1]
