"""Measuring the drifting task's rules from trials: block changes, drift, block lengths."""

import pandas as pd
import pytest

from src.behavior_analysis.drifting_task_rules import (
    block_change_summary,
    block_changes,
    complete_block_lengths,
    drift_counts,
)


def _table(magnitudes, choices, mouse="m", problem=1, trial_in_session=None, after_gap=None):
    n = len(choices)
    return pd.DataFrame(
        dict(
            mouse=mouse,
            problem=problem,
            trial=range(n),
            trial_in_session=range(n) if trial_in_session is None else trial_in_session,
            after_gap=[False] * n if after_gap is None else after_gap,
            magnitudes=magnitudes,
            choice=choices,
        )
    )


def _changes_table():
    a = {"A": 4, "B": 2, "C": 0}
    # block 1: A chosen most; A drops to 0 and C takes A's value 4
    b = {"A": 0, "B": 2, "C": 4}
    # block 2: B chosen most; B drops to 0 but A gets 1 instead of B's value 2
    c = {"A": 1, "B": 0, "C": 4}
    mags = [a] * 4 + [b] * 5 + [c] * 2
    choices = ["A", "A", "A", "B"] + ["B", "B", "B", "C", "A"] + ["A", "A"]
    return _table(mags, choices)


def test_block_changes_rank_the_zeroed_arm_and_check_inheritance():
    changes = block_changes(_changes_table())
    assert changes["trial"].tolist() == [4, 9]
    assert changes["zeroed_arm"].tolist() == ["A", "B"]
    assert changes["previous_zero_arm"].tolist() == ["C", "A"]
    assert changes["zeroed_choice_rank"].tolist() == ["most", "most"]
    assert changes["inherits_value"].tolist() == [True, False]
    assert changes["block_length"].tolist() == [4, 5]
    assert changes["complete_block"].tolist() == [False, True]
    assert complete_block_lengths(changes).tolist() == [5]


def test_least_and_tie_ranks():
    a, b = {"A": 4, "B": 2, "C": 0}, {"A": 4, "B": 0, "C": 2}
    least = block_changes(_table([a, a, a, b], ["A", "A", "B", "A"]))
    assert least["zeroed_choice_rank"].tolist() == ["least"]
    tie = block_changes(_table([a, a, b], ["A", "B", "A"]))
    assert tie["zeroed_choice_rank"].tolist() == ["tie"]


def test_summary_counts_and_fractions():
    summary = block_change_summary(block_changes(_changes_table())).set_index("mouse")
    assert summary.loc["all", "n_block_changes"] == 2
    assert summary.loc["all", "frac_most"] == 1.0
    assert summary.loc["all", "frac_inherits_value"] == 0.5


def test_drift_counts_ignore_block_changes():
    a, a2, b = {"A": 4, "B": 2, "C": 0}, {"A": 3, "B": 2, "C": 0}, {"A": 0, "B": 2, "C": 3}
    counts = drift_counts(_table([a, a2, a2, b], ["A"] * 4))
    # two within-block transitions with two moving arms each; one change (A 4 -> 3)
    assert counts.n_opportunities == 4
    assert counts.n_changes == 1
    assert counts.rate == 0.25


def test_two_zero_arms_raise():
    a, bad = {"A": 4, "B": 2, "C": 0}, {"A": 0, "B": 2, "C": 0}
    with pytest.raises(ValueError, match="exactly one arm at 0"):
        block_changes(_table([a, bad], ["A", "A"]))


def test_drift_counts_skip_session_starts_and_gaps():
    a, a2 = {"A": 4, "B": 2, "C": 0}, {"A": 3, "B": 2, "C": 0}
    new_session = _table([a, a2, a2], ["A"] * 3, trial_in_session=[0, 0, 1])
    assert drift_counts(new_session).n_opportunities == 2
    assert drift_counts(new_session).n_changes == 0
    gap = _table([a, a2, a2], ["A"] * 3, after_gap=[False, True, False])
    assert drift_counts(gap).n_changes == 0
