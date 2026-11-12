"""P(stay) as a function of run length, after a zero and after a non-zero payout."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import stay_by_run_length as rl
from tests.test_null_agents import hand_table


def test_run_length_counts_the_consecutive_trials_at_the_current_port():
    table = hand_table([0, 0, 0, 1, 1, 2, 0, 0], [1] * 8)
    lengths = rl.run_lengths(table)
    # run length before each trial's decision = trials in a row at the previous port; 0 at a start
    assert lengths.tolist() == [0, 1, 2, 3, 1, 2, 1, 1]


def test_run_length_restarts_at_a_session_start_and_after_a_gap():
    table = hand_table([0] * 8, [1] * 8, starts=(3,), gaps=(6,))
    assert rl.run_lengths(table).tolist() == [0, 1, 2, 0, 1, 2, 0, 1]


def test_bins_follow_the_default_edges():
    assert rl.bin_labels(np.array([1, 2, 3, 4, 5, 6, 9, 10, 40])).tolist() == [
        "1", "2", "3", "4-5", "4-5", "6-9", "6-9", "10+", "10+"
    ]  # fmt: skip
    with pytest.raises(ValueError, match="run lengths"):
        rl.bin_labels(np.array([0]))


def test_stay_by_run_length_by_hand():
    # run of 1 at arm 0 (paid 0, then stayed), runs of 2 and 3 after payouts, then a switch
    choices = [0, 0, 0, 0, 1, 1, 2]
    rewards = [0, 2, 1, 0, 1, 0, 1]
    table = hand_table(choices, rewards)
    out = rl.stay_by_run_length(table, min_pairs=1).set_index(["payout", "bin"])
    # decisions: t1 run 1 after payout 0 -> stay; t2 run 2 after 2 -> stay; t3 run 3 after 1 ->
    # stay; t4 run 4 after 0 -> switch; t5 run 1 after 1 -> stay; t6 run 2 after 0 -> switch
    assert out.loc[("zero", "1"), "p_stay"] == 1.0 and out.loc[("zero", "1"), "n"] == 1
    assert out.loc[("zero", "4-5"), "p_stay"] == 0.0
    assert out.loc[("zero", "2"), "p_stay"] == 0.0
    assert out.loc[("non-zero", "2"), "p_stay"] == 1.0
    assert out.loc[("non-zero", "3"), "p_stay"] == 1.0
    assert out.loc[("non-zero", "1"), "p_stay"] == 1.0


def test_a_bin_with_too_few_decisions_is_nan_and_warned():
    table = hand_table([0, 0, 1, 1, 1, 0], [1] * 6)
    with pytest.warns(UserWarning, match="fewer than 5 decisions"):
        out = rl.stay_by_run_length(table, min_pairs=5)
    assert out["p_stay"].isna().all() and out["n"].sum() == 5  # five decisions, all in thin bins


def test_slope_is_the_change_of_p_stay_per_bin_and_needs_three_bins():
    rows = [
        ("m", "zero", b, 100, p)
        for b, p in zip(rl.BIN_NAMES, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7), strict=True)
    ]
    table = pd.DataFrame(rows, columns=["mouse", "payout", "bin", "n", "p_stay"])
    slopes = rl.run_length_slopes(table)
    assert slopes.loc[0, "slope"] == pytest.approx(0.1)
    short = table.iloc[:2]
    assert rl.run_length_slopes(short)["slope"].isna().all()


def test_one_mouse_only():
    two = pd.concat(
        [hand_table([0, 1, 2], [1] * 3, mouse="a"), hand_table([0, 1], [1] * 2, mouse="b")]
    )
    with pytest.raises(ValueError, match="one mouse"):
        rl.stay_by_run_length(two)
