"""Which drifting problems enter the rank-signature tests, and what the cutoff drops."""

import pandas as pd
import pytest

from src.behavior_analysis import problem_selection as ps


def table(rows: list[tuple]) -> pd.DataFrame:
    """A minimal trial table from ``(mouse, (cohort, n), n_trials)``."""
    records = [
        dict(mouse=mouse, problem=problem, trial=t)
        for mouse, problem, n_trials in rows
        for t in range(n_trials)
    ]
    return pd.DataFrame(records)


TRIALS = table(
    [
        ("a", (1, 1), 3),
        ("a", (1, 7), 2),
        ("a", (1, 8), 4),
        ("a", (1, 9), 1),
        ("b", (1, 7), 5),
        ("c", (3, 1), 2),
        ("c", (3, 12), 2),
    ]  # fmt: skip
)
CUTOFF = {1: 7, 3: None}


def test_a_cohort_cutoff_keeps_problems_up_to_and_including_it():
    kept = ps.select_problems(TRIALS, CUTOFF)
    assert sorted(set(kept.loc[kept["mouse"] == "a", "problem"])) == [(1, 1), (1, 7)]


def test_a_cohort_without_a_cutoff_keeps_every_drifting_problem():
    kept = ps.select_problems(TRIALS, CUTOFF)
    assert sorted(set(kept.loc[kept["mouse"] == "c", "problem"])) == [(3, 1), (3, 12)]


def test_selection_changes_no_row_it_keeps():
    kept = ps.select_problems(TRIALS, CUTOFF)
    expected = TRIALS[TRIALS["problem"].map(lambda p: p[0] == 3 or p[1] <= 7)]
    pd.testing.assert_frame_equal(kept, expected.reset_index(drop=True))


def test_changing_the_cutoff_is_one_edit():
    kept = ps.select_problems(TRIALS, {1: 8, 3: None})
    assert (1, 8) in set(kept["problem"]) and (1, 9) not in set(kept["problem"])


def test_a_cohort_missing_from_the_cutoffs_raises():
    with pytest.raises(ValueError, match=r"cohort 3"):
        ps.select_problems(TRIALS, {1: 7})


def test_a_problem_key_without_a_cohort_raises():
    bad = TRIALS.assign(problem=TRIALS["trial"])
    with pytest.raises(ValueError, match="cohort"):
        ps.select_problems(bad, CUTOFF)


def test_a_cutoff_that_removes_a_whole_mouse_raises():
    with pytest.raises(ValueError, match="mouse b"):
        ps.select_problems(TRIALS, {1: 6, 3: None})


@pytest.mark.parametrize("cutoff", [0, -1, 2.5])
def test_a_cutoff_must_be_a_positive_integer_or_none(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        ps.select_problems(TRIALS, {1: cutoff, 3: None})


def test_the_dropped_table_counts_problems_and_trials_per_mouse():
    kept = ps.select_problems(TRIALS, CUTOFF)
    dropped = ps.dropped_table(TRIALS, kept).set_index("mouse")
    assert dropped.loc["a", "n_problems_dropped"] == 2
    assert dropped.loc["a", "n_trials_dropped"] == 5
    assert dropped.loc["a", "problems_dropped"] == "(1, 8), (1, 9)"
    assert list(dropped.index) == ["a"]  # mice that lose nothing are not listed
