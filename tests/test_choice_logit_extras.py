"""Optional design columns of the choice logit: believed value, never-visited and binary rewards."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_logit as cl
from tests.test_choice_logit import PORTS, trial_frame


@pytest.fixture
def hand_built():
    # Arms in sorted order: A1, B3, C2. Sessions start at trials 0 and 3.
    # Rewards received: t0 B3 3, t1 A1 2, t2 A1 4, t3 C2 2, t4 B3 4.
    return trial_frame(
        ["B3", "A1", "A1", "C2", "B3"],
        [(1, 3, 0), (2, 3, 0), (4, 3, 0), (1, 1, 2), (1, 4, 0)],
        session_starts={0, 3},
    )


def column(design, name):
    return design.X[:, :, list(design.feature_names).index(name)]


def test_believed_value_is_the_last_reward_received_at_each_arm(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BELIEVED,))
    np.testing.assert_array_equal(
        column(design, "believed_value"),
        [[0, 0, 0], [0, 3, 0], [2, 3, 0], [4, 3, 0], [4, 3, 2]],
    )


def test_never_visited_is_one_until_the_arm_is_first_chosen(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BELIEVED,))
    np.testing.assert_array_equal(
        column(design, "believed_unvisited"),
        [[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]],
    )


def test_believed_value_carries_across_sessions_and_gaps(hand_built):
    hand_built.loc[2, "after_gap"] = True  # a dropped trial breaks choice history, not this
    design = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BELIEVED,))
    values = column(design, "believed_value")
    np.testing.assert_array_equal(values[2], [2, 3, 0])
    np.testing.assert_array_equal(values[3], [4, 3, 0])  # trial 3 starts a session


def test_believed_value_resets_at_the_start_of_a_problem(hand_built):
    second = trial_frame(
        ["A1", "A1", "B3"], [(2, 0, 0), (1, 0, 0), (0, 4, 0)], session_starts={0}, problem=2
    )
    design = cl.design_matrix(
        pd.concat([hand_built, second], ignore_index=True), 2, None, (cl.BELIEVED,)
    )
    values, unvisited = column(design, "believed_value"), column(design, "believed_unvisited")
    np.testing.assert_array_equal(values[5], [0, 0, 0])  # first trial of problem 2
    np.testing.assert_array_equal(unvisited[5], [1, 1, 1])
    np.testing.assert_array_equal(values[6], [2, 0, 0])  # not problem 1's A1 value


def test_believed_value_of_a_visited_arm_that_paid_nothing_is_zero_but_visited():
    trials = trial_frame(["A1", "B3", "A1"], [(0, 1, 0), (1, 1, 1), (1, 1, 1)], session_starts={0})
    design = cl.design_matrix(trials, n_lags=1, extras=(cl.BELIEVED,))
    assert column(design, "believed_value")[1, 0] == 0
    assert column(design, "believed_unvisited")[1, 0] == 0  # visited: not the same as unvisited


def test_design_without_extras_is_unchanged(hand_built):
    plain = cl.design_matrix(hand_built, n_lags=2)
    explicit = cl.design_matrix(hand_built, n_lags=2, extras=())
    np.testing.assert_array_equal(plain.X, explicit.X)
    assert plain.feature_names == explicit.feature_names
    assert not any(n.startswith(("believed_", "binary_")) for n in plain.feature_names)


def test_binary_reward_lags_code_rewarded_or_not(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BINARY,))
    np.testing.assert_array_equal(column(design, "binary_lag1")[1], [0, 1, 0])  # t0 paid 3 pulses
    np.testing.assert_array_equal(column(design, "reward_lag1")[1], [0, 3, 0])


def test_binary_lag_is_zero_for_a_chosen_arm_that_paid_nothing():
    trials = trial_frame(["A1", "B3", "A1"], [(0, 1, 0), (1, 1, 1), (1, 1, 1)], session_starts={0})
    design = cl.design_matrix(trials, n_lags=1, extras=(cl.BINARY,))
    np.testing.assert_array_equal(column(design, "binary_lag1")[1], [0, 0, 0])


def test_binary_equals_pulse_when_every_reward_is_one_pulse():
    choices = ["A1", "B3", "B3", "C2", "A1", "A1", "C2", "B3"]
    magnitudes = [(1, 1, 1)] * len(choices)
    trials = trial_frame(choices, magnitudes, session_starts={0, 4})
    design = cl.design_matrix(trials, n_lags=3, tail_decay=0.8, extras=(cl.BINARY,))
    for k in (1, 2, 3):
        np.testing.assert_array_equal(
            column(design, f"binary_lag{k}"), column(design, f"reward_lag{k}")
        )
    np.testing.assert_allclose(column(design, "binary_tail"), column(design, "reward_tail"))


def test_binary_tail_is_present_only_with_a_tail(hand_built):
    with_tail = cl.design_matrix(hand_built, n_lags=2, tail_decay=0.9, extras=(cl.BINARY,))
    without = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BINARY,))
    assert "binary_tail" in with_tail.feature_names
    assert "binary_tail" not in without.feature_names


def test_extras_follow_the_standard_columns(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BINARY, cl.BELIEVED))
    standard = cl.design_matrix(hand_built, n_lags=2)
    n = len(standard.feature_names)
    assert design.feature_names[:n] == standard.feature_names
    np.testing.assert_array_equal(design.X[:, :, :n], standard.X)
    assert design.feature_names[n:] == (
        "binary_lag1", "binary_lag2", "believed_value", "believed_unvisited",
    )  # fmt: skip


def test_select_features_picks_the_extra_groups(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2, extras=(cl.BINARY, cl.BELIEVED))
    believed = cl.select_features(design, (cl.BELIEVED,))
    assert believed.feature_names == ("believed_value", "believed_unvisited")
    binary = cl.select_features(design, (cl.BINARY, cl.CHOICE))
    # columns keep the design's order, whatever the order of the groups asked for
    assert binary.feature_names == ("choice_lag1", "choice_lag2", "binary_lag1", "binary_lag2")
    # the reward group is the pulse columns only, so adding the extras changes no model value
    assert cl.select_features(design, (cl.REWARD,)).feature_names == ("reward_lag1", "reward_lag2")


def test_weights_table_names_the_group_of_extra_features():
    table = cl.weights_table(
        "m1",
        ("binary_lag1", "binary_tail", "believed_value", "believed_unvisited"),
        np.array([0.1, 0.2, 0.3, 0.4]),
    )
    assert list(table["group"]) == [cl.BINARY, cl.BINARY, cl.BELIEVED, cl.BELIEVED]
    assert table["lag"].iloc[0] == 1.0
    assert table["lag"].iloc[1:].isna().all()


def test_unknown_extra_is_rejected(hand_built):
    with pytest.raises(ValueError, match="unknown extra"):
        cl.design_matrix(hand_built, n_lags=2, extras=("magic",))


def test_ports_fixture_matches_sorted_arm_order():
    assert PORTS == sorted(PORTS)  # the expected arrays above assume A1, B3, C2


def test_duplicate_extras_are_rejected(hand_built):
    with pytest.raises(ValueError, match="duplicate"):
        cl.design_matrix(hand_built, n_lags=2, extras=(cl.BELIEVED, cl.BELIEVED))


def test_binary_tail_has_its_own_decay(hand_built):
    design = cl.design_matrix(
        hand_built, n_lags=1, tail_decay=0.9, extras=(cl.BINARY,), binary_tail_decay=0.5
    )
    # rewarded-or-not per trial for (A1, B3, C2): t0 B3, t1 A1, t2 A1, t3 C2, t4 B3 all paid
    # tail(t) = 0.5 * tail(t-1) + rewarded(t-2); lag 1 is separate
    expected_b3 = [0, 0, 1, 0.5, 0.25]  # B3 rewarded at t0 and t4
    np.testing.assert_allclose(column(design, "binary_tail")[:, 1], expected_b3)
    # the pulse tail keeps its own decay
    pulse = cl.design_matrix(hand_built, n_lags=1, tail_decay=0.9)
    np.testing.assert_allclose(column(design, "reward_tail"), column(pulse, "reward_tail"))


def test_binary_tail_decay_defaults_to_the_pulse_decay(hand_built):
    same = cl.design_matrix(hand_built, n_lags=1, tail_decay=0.9, extras=(cl.BINARY,))
    explicit = cl.design_matrix(
        hand_built, n_lags=1, tail_decay=0.9, extras=(cl.BINARY,), binary_tail_decay=0.9
    )
    np.testing.assert_array_equal(same.X, explicit.X)


def test_binary_tail_decay_needs_a_tail_and_a_valid_value(hand_built):
    with pytest.raises(ValueError, match="binary_tail_decay"):
        cl.design_matrix(hand_built, n_lags=1, extras=(cl.BINARY,), binary_tail_decay=0.5)
    with pytest.raises(ValueError, match="binary_tail_decay"):
        cl.design_matrix(
            hand_built, n_lags=1, tail_decay=0.9, extras=(cl.BINARY,), binary_tail_decay=1.5
        )
