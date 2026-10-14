import pytest

from src.behavior_analysis.split_early_late_good_reversals import split_good_reversals_early_late


def reversals(*numbers):
    """One reversal dict per number, tagged with its list position."""
    return [{"good_reversal_number": n, "pos": i} for i, n in enumerate(numbers)]


def numbers(revs):
    return [r["good_reversal_number"] for r in revs]


def split(revs, **kw):
    early, late = split_good_reversals_early_late({"m": revs}, **kw)
    return numbers(early["m"]), numbers(late["m"])


def test_default_splits_at_the_midpoint():
    assert split(reversals(1, 2, 3, 4)) == ([1, 2], [3, 4])


def test_default_gives_the_odd_one_to_late():
    assert split(reversals(1, 2, 3, 4, 5)) == ([1, 2], [3, 4, 5])
    assert split(reversals(1)) == ([], [1])


def test_reversals_are_ordered_by_number_before_splitting():
    assert split(reversals(4, 1, 3, 2)) == ([1, 2], [3, 4])


def test_input_list_is_not_reordered():
    revs = reversals(2, 1)
    split_good_reversals_early_late({"m": revs})
    assert numbers(revs) == [2, 1]


def test_reversals_without_a_number_sort_as_zero_and_keep_their_order():
    revs = [{"pos": 0}, {"good_reversal_number": 1, "pos": 1}, {"pos": 2}]
    early, late = split_good_reversals_early_late({"m": revs})
    # sorted by number: pos 0 (0), pos 2 (0), pos 1 (1); the midpoint is after the first
    assert [r["pos"] for r in early["m"]] == [0]
    assert [r["pos"] for r in late["m"]] == [2, 1]


def test_first_n_takes_the_first_n_and_late_is_the_remainder():
    assert split(reversals(1, 2, 3, 4, 5), first_n=2) == ([1, 2], [3, 4, 5])


def test_last_n_takes_the_last_n_and_early_is_the_remainder():
    assert split(reversals(1, 2, 3, 4, 5), last_n=2) == ([1, 2, 3], [4, 5])


def test_both_given_use_each_independently_and_may_overlap_or_leave_a_gap():
    assert split(reversals(1, 2, 3), first_n=2, last_n=2) == ([1, 2], [2, 3])
    assert split(reversals(1, 2, 3, 4, 5), first_n=1, last_n=1) == ([1], [5])


def test_counts_beyond_the_list_are_clipped():
    assert split(reversals(1, 2), first_n=10) == ([1, 2], [])
    assert split(reversals(1, 2), last_n=10) == ([], [1, 2])


def test_zero_and_negative_counts_mean_none():
    assert split(reversals(1, 2), first_n=0) == ([], [1, 2])
    assert split(reversals(1, 2), first_n=-3) == ([], [1, 2])
    assert split(reversals(1, 2), last_n=-3) == ([1, 2], [])
    assert split(reversals(1, 2), last_n=0) == ([1, 2], [])


def test_counts_are_converted_to_int():
    assert split(reversals(1, 2, 3), first_n=1.9) == ([1], [2, 3])
    assert split(reversals(1, 2, 3), last_n="1") == ([1, 2], [3])


def test_a_count_that_is_not_a_number_raises_for_subjects_with_reversals():
    with pytest.raises(ValueError):
        split_good_reversals_early_late({"m": reversals(1, 2)}, first_n="two")


def test_each_subject_is_split_on_its_own_and_empty_ones_stay_empty():
    info = {"a": reversals(1, 2), "b": reversals(1, 2, 3, 4), "none": [], "missing": None}
    early, late = split_good_reversals_early_late(info)
    assert {s: numbers(v) for s, v in early.items()} == {
        "a": [1],
        "b": [1, 2],
        "none": [],
        "missing": [],
    }
    assert {s: numbers(v) for s, v in late.items()} == {
        "a": [2],
        "b": [3, 4],
        "none": [],
        "missing": [],
    }
    assert list(early) == list(late) == list(info)


def test_returned_reversals_are_the_same_dicts_as_the_input():
    revs = reversals(1, 2)
    early, late = split_good_reversals_early_late({"m": revs})
    assert early["m"][0] is revs[0] and late["m"][0] is revs[1]
