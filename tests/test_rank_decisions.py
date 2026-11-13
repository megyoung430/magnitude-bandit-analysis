"""The decision table on the hand-built mouse of test_rank_state."""

import numpy as np
import pytest

from src.agents import kernel_q_agent as kq
from src.behavior_analysis.rank_tests import rank_decisions as rd
from src.behavior_analysis.rank_tests import rank_state as rs
from tests.test_rank_state import hand_table


@pytest.fixture(scope="module")
def state():
    return rs.mouse_state(hand_table())


@pytest.fixture(scope="module")
def table(state):
    return rd.decision_table(state)


def test_decisions_are_the_defined_states_whose_next_trial_is_in_the_segment(table):
    # t2, t3 (problem 1, session 1) and t6 (after the dropped trial); t4, t5, t7 cross a segment
    assert table["t"].tolist() == [2, 3, 6]


def test_a_decision_row_holds_its_state_and_outcome(table):
    row = table[table["t"] == 3].iloc[0]  # chose A paying 3; believed A3 B2 C0; next trial A
    assert (row.own, row.next, row.leave) == (0, 0, 0)
    assert (row.payout, row.own_zero, row.own_value) == (3, 0, 3)
    assert (row.v_best, row.v_worst, row.d) == (2, 0, 1)
    assert (row.own_first, row.own_second) == (1, 0)
    assert row.run_log == 0


def test_leaving_is_choosing_another_arm_next(table):
    row = table[table["t"] == 2].iloc[0]  # chose C paying 0, then A
    assert (row.own, row.next, row.leave, row.own_zero) == (2, 0, 1, 1)
    assert (row.v_best, row.v_worst, row.d) == (4, 2, -4)
    assert (row.own_first, row.own_second) == (0, 0)  # the lowest of the three


def test_history_covariates_are_zero_at_a_segment_start(table):
    row = table[table["t"] == 6].iloc[0]  # first trial after a dropped trial
    assert (row.lag2, row.lag3, row.lag4, row.trace) == (0, 0, 0, 0)


def test_lags_flag_the_own_arm_on_earlier_trials_of_the_segment(table):
    # t3 chose A; the trials before it chose C (t2), B (t1) and A (t0): only three trials back
    lags = table.loc[table["t"] == 3, ["lag2", "lag3", "lag4"]].to_numpy()[0]
    assert lags.tolist() == [0, 0, 1]


def test_values_replace_the_believed_values_but_not_the_decisions(state):
    values = np.tile([1.0, 5.0, 3.0], (len(state.chosen), 1))
    table = rd.decision_table(state, values)
    assert table["t"].tolist() == [2, 3, 6]
    row = table[table["t"] == 3].iloc[0]  # own A has value 1 and the alternatives 5 and 3
    assert (row.own_value, row.v_best, row.v_worst, row.d) == (1, 5, 3, -4)


def test_values_of_the_wrong_shape_raise(state):
    with pytest.raises(ValueError, match="values must be"):
        rd.decision_table(state, np.zeros((3, 3)))


def test_kernel_q_values_follow_the_model_update_within_a_problem(state):
    params = (0.5, 0.5, 0.8, 0.2, 2.0, 1.0, 0.0)
    q = rd.kernel_q_values(state, params)
    expected, k = np.zeros(3), np.zeros(3)
    for t in range(8):
        expected, k = kq.update_state(
            expected, k, int(state.chosen[t]), float(state.payout[t]), params
        )
    np.testing.assert_allclose(q[7], expected)
    # problem 2 starts again from zero: after C pays 0 nothing has been learned
    np.testing.assert_allclose(q[8], np.zeros(3))
