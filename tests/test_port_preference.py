"""Port-preference counting on synthetic sessions with hand-computed answers."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.port_preference import (
    add_ratio,
    category_summary,
    count_nan_ratios,
    count_port_preference,
    pool_problems,
    port_category,
)


def session(trials, *, initiation="B2", n_recordings=1):
    """A session whose recordings each hold *trials*, a list of ``(magnitudes, choice)``."""
    recording = {
        "reward_magnitudes": [mags for mags, _ in trials],
        "choice": [choice for _, choice in trials],
    }
    return {
        "initiation_tower": initiation,
        "trial_variables": [dict(recording) for _ in range(n_recordings)],
    }


def one_mouse(trials, **kwargs):
    return {1: {"m1": {"ses-1": session(trials, **kwargs)}}}


# Three trials: A1 (X) wins one, B1 (plus) wins one, and they tie on the third; C3 (X) is
# offered every time with magnitude 0, so it is never best.
TRIALS = [
    ({"A1": 4, "B1": 2, "C3": 0}, "A1"),
    ({"A1": 1, "B1": 3, "C3": 0}, "A1"),
    ({"A1": 2, "B1": 2, "C3": 0}, "B1"),
]


def row(table, **match):
    """The single row of *table* whose columns equal *match*."""
    mask = np.ones(len(table), dtype=bool)
    for column, value in match.items():
        mask &= (table[column] == value).to_numpy()
    assert mask.sum() == 1, match
    return table[mask].iloc[0]


# ---------------------------------------------------------------- geometry


@pytest.mark.parametrize(
    ("port", "expected"),
    [
        ("A1", "X"),
        ("C1", "X"),
        ("A3", "X"),
        ("C3", "X"),
        ("B1", "plus"),
        ("B3", "plus"),
        ("A2", "plus"),
        ("C2", "plus"),
        ("B2", "initiation"),
    ],
)
def test_categories_around_the_centre_initiation_port(port, expected):
    assert port_category(port, "B2") == expected


def test_categories_follow_the_initiation_port_not_the_centre():
    # From A1 only A2 and B1 touch it orthogonally and only B2 touches it diagonally
    assert {p: port_category(p, "A1") for p in ("A2", "B1", "B2", "C3", "C1", "A1")} == {
        "A2": "plus",
        "B1": "plus",
        "B2": "X",
        "C3": "other",
        "C1": "other",
        "A1": "initiation",
    }


def test_unknown_port_raises():
    with pytest.raises(ValueError, match="D4"):
        port_category("D4", "B2")


# ---------------------------------------------------------------- counting


def test_port_counts_with_a_tie_and_a_port_that_is_never_best():
    counts = count_port_preference(one_mouse(TRIALS)).by_port
    a1 = row(counts, port="A1")
    assert (a1.n_trials, a1.chosen, a1.best) == (3, 2, 1.5)
    b1 = row(counts, port="B1")
    assert (b1.n_trials, b1.chosen, b1.best) == (3, 1, 1.5)
    c3 = row(counts, port="C3")
    assert (c3.n_trials, c3.chosen, c3.best) == (3, 0, 0)
    never_offered = row(counts, port="A3")
    assert (never_offered.n_trials, never_offered.chosen, never_offered.best) == (0, 0, 0)


def test_best_shares_over_the_ports_of_a_trial_sum_to_one():
    counts = count_port_preference(one_mouse(TRIALS)).by_port
    assert counts.best.sum() == len(TRIALS)
    assert counts.chosen.sum() == len(TRIALS)


def test_port_ratio_is_chosen_fraction_over_best_fraction_and_nan_if_never_best():
    ratios = add_ratio(count_port_preference(one_mouse(TRIALS)).by_port)
    assert row(ratios, port="A1").ratio == pytest.approx((2 / 3) / (1.5 / 3))
    assert row(ratios, port="B1").ratio == pytest.approx((1 / 3) / (1.5 / 3))
    assert np.isnan(row(ratios, port="C3").ratio)  # offered but never best
    assert np.isnan(row(ratios, port="A3").ratio)  # never offered


def test_category_counts_use_trials_where_a_port_of_the_category_was_offered():
    cats = count_port_preference(one_mouse(TRIALS)).by_category
    x = row(cats, category="X")
    assert (x.n_trials, x.chosen, x.best) == (3, 2, 1.5)
    plus = row(cats, category="plus")
    assert (plus.n_trials, plus.chosen, plus.best) == (3, 1, 1.5)
    ratios = add_ratio(cats)
    assert row(ratios, category="X").ratio == pytest.approx(2 / 1.5)
    assert row(ratios, category="plus").ratio == pytest.approx(1 / 1.5)


def test_a_tie_across_categories_splits_the_best_between_them():
    # A1 (X) and B1 (plus) tie for best on every trial, so each category is best half the time
    trials = [({"A1": 3, "B1": 3, "C3": 1}, "A1")] * 4
    cats = count_port_preference(one_mouse(trials)).by_category
    assert row(cats, category="X").best == 2
    assert row(cats, category="plus").best == 2


def test_category_denominator_ignores_trials_without_a_port_of_the_category():
    # Trial 2 offers only X ports: it counts for X but not for plus
    trials = [
        ({"A1": 3, "B1": 1, "C3": 0}, "B1"),
        ({"A1": 3, "C1": 1, "C3": 0}, "A1"),
    ]
    cats = count_port_preference(one_mouse(trials)).by_category
    assert row(cats, category="plus").n_trials == 1
    assert row(cats, category="X").n_trials == 2


def test_category_ratio_is_nan_when_the_category_was_never_best():
    trials = [({"A1": 3, "B1": 1, "C3": 0}, "B1")] * 3
    ratios = add_ratio(count_port_preference(one_mouse(trials)).by_category)
    assert np.isnan(row(ratios, category="plus").ratio)
    assert row(ratios, category="X").ratio == 0.0  # best every time, never chosen


def test_different_initiation_port_changes_the_categories():
    trials = [({"B2": 3, "B1": 1, "C3": 0}, "B2")] * 2
    cats = count_port_preference(one_mouse(trials, initiation="A1")).by_category
    assert row(cats, category="X").best == 2  # B2 is diagonal to A1
    assert row(cats, category="plus").n_trials == 2  # B1 is orthogonal to A1
    assert row(cats, category="other").n_trials == 2  # C3 is neither
    assert row(cats, category="X").chosen == 2


def test_missing_initiation_defaults_to_b2():
    problems = one_mouse(TRIALS)
    del problems[1]["m1"]["ses-1"]["initiation_tower"]
    cats = count_port_preference(problems).by_category
    assert row(cats, category="X").n_trials == 3


def test_every_recording_of_a_session_is_counted():
    counts = count_port_preference(one_mouse(TRIALS, n_recordings=2)).by_port
    assert row(counts, port="A1").n_trials == 6


def test_trials_without_a_choice_or_magnitudes_are_skipped_and_counted():
    trials = [*TRIALS, ({"A1": 3, "B1": 1}, None), ({}, "A1"), (None, "A1")]
    result = count_port_preference(one_mouse(trials))
    assert row(result.by_port, port="A1").n_trials == 3
    assert result.skipped.to_dict() == {"m1": 3}


def test_choice_outside_the_offered_ports_raises():
    trials = [({"A1": 3, "B1": 1}, "C3")]
    with pytest.raises(ValueError, match="m1.*C3"):
        count_port_preference(one_mouse(trials))


def test_unknown_port_in_magnitudes_raises():
    trials = [({"A1": 3, "Z9": 1}, "A1")]
    with pytest.raises(ValueError, match="Z9"):
        count_port_preference(one_mouse(trials))


def test_initiation_port_that_changes_within_a_problem_raises():
    problems = one_mouse(TRIALS)
    problems[1]["m1"]["ses-2"] = session(TRIALS, initiation="A1")
    with pytest.raises(ValueError, match="m1.*initiation"):
        count_port_preference(problems)


def test_no_trials_at_all_raises():
    with pytest.raises(ValueError, match="no trials"):
        count_port_preference({1: {"m1": {"ses-1": session([])}}})


# ---------------------------------------------------------------- pooling and mice


def two_problem_mouse():
    """Problem 1 starts at B2, problem 2 at A1: the same port changes category."""
    return {
        1: {"m1": {"ses-1": session([({"B1": 3, "A1": 1}, "B1")] * 2, initiation="B2")}},
        2: {"m1": {"ses-1": session([({"B1": 3, "A1": 1}, "A1")] * 2, initiation="A1")}},
    }


def test_problems_pool_counts_per_port_and_per_category_not_ratios():
    counts = count_port_preference(two_problem_mouse())
    ports = pool_problems(counts.by_port, "port")
    b1 = row(ports, mouse="m1", port="B1")
    assert (b1.n_trials, b1.chosen, b1.best) == (4, 2, 4)
    cats = pool_problems(counts.by_category, "category")
    # B1 is plus from B2 and from A1; A1 is X from B2 and the initiation port in problem 2
    plus = row(cats, mouse="m1", category="plus")
    assert (plus.n_trials, plus.chosen, plus.best) == (4, 2, 4)
    x = row(cats, mouse="m1", category="X")
    assert (x.n_trials, x.chosen, x.best) == (2, 0, 0)
    initiation = row(cats, mouse="m1", category="initiation")
    assert (initiation.n_trials, initiation.chosen) == (2, 2)


def test_pooled_ratio_differs_from_the_mean_of_problem_ratios():
    # Problem 1: 1 of 1 trials chosen, best 1 of 1 -> ratio 1. Problem 2: chosen 0 of 3,
    # best 3 of 3 -> ratio 0. Pooled counts: chosen 1/4 over best 4/4 = 0.25, not 0.5.
    problems = {
        1: {"m1": {"ses-1": session([({"B1": 3, "A1": 1}, "B1")])}},
        2: {"m1": {"ses-1": session([({"B1": 3, "A1": 1}, "A1")] * 3)}},
    }
    pooled = pool_problems(count_port_preference(problems).by_port, "port")
    assert row(add_ratio(pooled), port="B1").ratio == pytest.approx(0.25)


def test_summary_is_mean_and_se_over_mice_ignoring_never_best_mice():
    per_mouse = pd.DataFrame(
        {
            "mouse": ["a", "b", "c", "a", "b", "c"],
            "category": ["X", "X", "X", "plus", "plus", "plus"],
            "ratio": [1.0, 3.0, np.nan, 2.0, 2.0, 2.0],
        }
    )
    summary = category_summary(per_mouse)
    x = summary.loc["X"]
    assert x["mean"] == pytest.approx(2.0)
    assert x["se"] == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
    assert x["n_mice"] == 2
    plus = summary.loc["plus"]
    assert (plus["mean"], plus["se"], plus["n_mice"]) == (2.0, 0.0, 3)


def test_nan_cases_are_counted_for_ports_and_categories():
    mice = {
        1: {
            "m1": {"ses-1": session(TRIALS)},
            "m2": {"ses-1": session([({"A1": 3, "B1": 1, "C3": 0}, "B1")] * 3)},
        }
    }
    counts = count_port_preference(mice)
    ports = add_ratio(pool_problems(counts.by_port, "port"))
    cats = add_ratio(pool_problems(counts.by_category, "category"))
    nan = count_nan_ratios(ports, cats)
    # offered but never best: m1 C3; m2 B1 and C3 (A1 is best every trial)
    assert nan == {"ports": 3, "categories": 1}  # m2 never has plus best
