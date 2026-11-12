"""Believed values, ranks and segment bookkeeping on a hand-built mouse."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import rank_state as rs
from tests.switch_regression_fixtures import table_from_rows

NAN = np.nan
PORTS = ("A1", "B3", "C2")

# (problem key, session, trial in session, after_gap, choice arm, magnitudes of A1, B3, C2)
HAND = (
    ((1, 1), 0, 0, False, 0, (4, 2, 0)),  # t0 A pays 4
    ((1, 1), 0, 1, False, 1, (4, 2, 0)),  # t1 B pays 2
    ((1, 1), 0, 2, False, 2, (4, 2, 0)),  # t2 C pays 0: every arm visited
    ((1, 1), 0, 3, False, 0, (3, 2, 0)),  # t3 A pays 3
    ((1, 1), 0, 4, False, 0, (2, 2, 0)),  # t4 A pays 2: ties B
    ((1, 1), 1, 0, False, 0, (2, 2, 0)),  # t5 new session: segment start, A pays 2
    ((1, 1), 1, 1, True, 1, (2, 4, 0)),  # t6 after a dropped trial: segment start, B pays 4
    ((1, 1), 1, 2, False, 1, (2, 4, 0)),  # t7 B again
    ((1, 2), 0, 0, False, 2, (1, 3, 0)),  # t8 new problem: believed values restart
    ((1, 2), 0, 1, False, 2, (1, 3, 0)),  # t9 C again
)


def hand_table(mouse: str = "m1") -> pd.DataFrame:
    rows = [
        (
            problem,
            session,
            in_session,
            gap,
            PORTS[arm],
            mags[arm],
            dict(zip(PORTS, mags, strict=True)),
        )
        for problem, session, in_session, gap, arm, mags in HAND
    ]
    return table_from_rows(rows, mouse)


@pytest.fixture(scope="module")
def state():
    return rs.mouse_state(hand_table())


def test_believed_value_is_the_last_payout_of_each_arm_in_the_problem(state):
    expected = np.array(
        [
            [4, NAN, NAN], [4, 2, NAN], [4, 2, 0], [3, 2, 0], [2, 2, 0],
            [2, 2, 0], [2, 4, 0], [2, 4, 0],
            [NAN, NAN, 0], [NAN, NAN, 0],
        ]
    )  # fmt: skip
    np.testing.assert_array_equal(state.believed, expected)


def test_believed_values_cross_sessions_but_not_problems(state):
    assert state.believed[5].tolist() == [2, 2, 0]  # session 2 keeps session 1's values
    assert np.isnan(state.believed[8, 0])  # problem 2 starts over


def test_a_state_is_defined_once_every_arm_of_the_problem_has_been_visited(state):
    assert state.defined.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_the_next_trial_counts_only_inside_a_segment(state):
    # t4 -> t5 crosses a session start, t5 -> t6 an after-gap start, t7 -> t8 a problem change
    assert state.next_ok.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_segment_starts_are_problem_starts_session_starts_and_dropped_trials(state):
    assert np.flatnonzero(state.start).tolist() == [0, 5, 6, 8]


def test_run_length_counts_consecutive_choices_of_the_chosen_arm_in_the_segment(state):
    assert state.run_length.tolist() == [1, 1, 1, 1, 2, 1, 1, 2, 1, 2]


def test_problem_index_is_the_number_in_the_key(state):
    assert state.problem_index.tolist() == [1] * 8 + [2] * 2


def test_an_integer_problem_key_is_its_own_index():
    table = hand_table().assign(problem=lambda t: t["problem"].map(lambda k: k[1]))
    assert rs.mouse_state(table).problem_index.tolist() == [1] * 8 + [2] * 2


def test_average_ranks_share_ties():
    values = np.array([[3.0, 2, 0], [2, 2, 0], [0, 2, 2], [1, 1, 1], [2, 0, 2]])
    expected = [[1, 2, 3], [1.5, 1.5, 3], [3, 1.5, 1.5], [2, 2, 2], [1.5, 3, 1.5]]
    np.testing.assert_allclose(rs.average_rank(values), expected)


def test_average_rank_of_an_undefined_row_is_nan():
    assert np.isnan(rs.average_rank(np.array([[1.0, np.nan, 0]]))).all()


def test_rank_weights_interpolate_between_ranks_and_sum_to_one():
    ranks = np.array([[1.0, 2, 3], [1.5, 1.5, 3], [2, 2, 2]])
    weights = rs.rank_weights(ranks)  # (n, 3 arms, first/second/third)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)
    np.testing.assert_allclose(weights[0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(weights[1, 0], [0.5, 0.5, 0])  # tied first and second
    np.testing.assert_allclose(weights[2, 0], [0, 1, 0])  # a three-way tie is everyone's second


def test_alternatives_are_split_into_best_and_worst_by_believed_value(state):
    best, worst = rs.alternative_values(state)
    # t2: chose C (0); alternatives A (4) and B (2); t3: chose A; alternatives B (2), C (0)
    assert best[2] == 4 and worst[2] == 2
    assert best[3] == 2 and worst[3] == 0
    assert np.isnan(best[1])  # undefined state
    assert best[4] == 2 and worst[4] == 0  # A chosen; B (2) and C (0)


def test_a_state_needs_one_mouse():
    with pytest.raises(ValueError, match="one mouse"):
        rs.mouse_state(pd.concat([hand_table("a"), hand_table("b")], ignore_index=True))
