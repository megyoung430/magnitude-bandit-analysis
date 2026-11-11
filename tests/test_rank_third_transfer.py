"""The third-option, transfer and block-inference tests on toys with known answers."""

import numpy as np
import pytest

from src.behavior_analysis import rank_decisions as rd
from src.behavior_analysis import rank_logit as rl
from src.behavior_analysis import rank_state as rs
from src.behavior_analysis import rank_third as r3
from src.behavior_analysis import rank_transfer as rt
from tests.rank_fixtures import rank_mouse, softmax_mouse
from tests.switch_regression_fixtures import table_from_rows

PORTS = ("A1", "B3", "C2")
SIZE = dict(n_problems=8, sessions=3, trials=200)


def parts(trials):
    state = rs.mouse_state(trials)
    return state, rd.decision_table(state)


@pytest.fixture(scope="module")
def rank():
    return parts(rank_mouse(seed=1, **SIZE))


@pytest.fixture(scope="module")
def softmax():
    return parts(softmax_mouse(seed=1, **SIZE))


# --- the third option ---------------------------------------------------------------------------


def test_the_worst_arm_is_chosen_less_the_further_it_lies_below_only_for_the_softmax_learner(
    rank, softmax
):
    r = r3.worst_choice(*rank).readouts
    s = r3.worst_choice(*softmax).readouts
    assert s["worst_slope"] < -3 * s["se_worst_slope"]
    assert r["worst_slope"] > s["worst_slope"] + 0.5  # flat at a floor, not falling


def test_the_curve_gives_the_share_choosing_the_worst_by_gap(softmax):
    curve = r3.worst_choice(*softmax).curves
    assert curve["x"].min() >= 1  # a unique worst arm lies at least one pulse below
    assert curve["n"].sum() == r3.worst_choice(*softmax).readouts["n_worst"]
    assert curve["y"].between(0, 1).all()


def test_the_worst_arms_value_does_not_decide_best_against_second_for_either_agent(rank, softmax):
    for source in (rank, softmax):
        r = r3.top_two_choice(*source).readouts
        assert abs(r["top2_w_worst"]) < 3 * r["se_top2_worst"] + 0.1
        assert r["n_top2"] > 100


def test_a_worst_arm_that_is_never_chosen_raises(rank):
    state, table = rank
    arms, _ = r3.sorted_arms(state.believed[table["t"].to_numpy()])
    with pytest.raises(rl.DegenerateOutcomeError):
        r3.worst_choice(state, table.assign(next=arms[:, 0]))  # always the best arm


# --- transfer across problems -------------------------------------------------------------------


def learning_mouse(p_best_by_problem, trials=150, seed=0):
    """Problems whose best arm is A (4 pulses, B 2, C 0), chosen with a rising probability."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, p_best in enumerate(p_best_by_problem, start=1):
        for t in range(trials):
            arm = 0 if rng.random() < p_best else int(rng.integers(1, 3))
            mags = (4, 2, 0)
            rows.append(
                ((1, k), 0, t, False, PORTS[arm], mags[arm], dict(zip(PORTS, mags, strict=True)))
            )
    return table_from_rows(rows)


def test_a_mouse_that_improves_across_problems_has_a_positive_slope_and_faster_thresholds():
    state = rs.mouse_state(learning_mouse([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]))
    r = rt.transfer(state).readouts
    assert r["slope_p100"] > 0.05 and r["slope_p50"] > 0.05
    assert r["slope_threshold"] < 0 and r["half_p100"] > 0.3
    assert r["n_problems"] == 8


def test_a_mouse_that_does_not_improve_has_no_slope():
    state = rs.mouse_state(learning_mouse([0.6] * 8))
    r = rt.transfer(state).readouts
    assert abs(r["slope_p100"]) < 0.03 and abs(r["half_p100"]) < 0.1


def test_the_threshold_is_censored_at_the_problems_length_when_never_reached():
    state = rs.mouse_state(learning_mouse([0.2, 0.2, 0.2, 0.9], trials=60))
    table = rt.problem_metrics(state)
    assert table["trials_to_threshold"].iloc[0] == 60
    assert table["trials_to_threshold"].iloc[3] < 40


def test_short_problems_have_no_p100_and_too_few_problems_no_slope():
    state = rs.mouse_state(learning_mouse([0.5, 0.9], trials=80))
    assert rt.problem_metrics(state)["p100"].isna().all()
    r = rt.transfer(state).readouts
    assert np.isnan(r["slope_p100"]) and np.isnan(r["slope_threshold"])


def test_transfer_curves_are_drawn_by_problem_number():
    curves = rt.transfer(rs.mouse_state(learning_mouse([0.3, 0.6, 0.9]))).curves
    assert set(curves["curve"]) == {"transfer_p100", "transfer_threshold"}
    assert sorted(set(curves["x"])) == [1, 2, 3]


# --- block changes ------------------------------------------------------------------------------

# t0-t2 visit A (4 pulses), B (2) and C (0); A is chosen on to t4. At t5 the task zeroes A (the
# arm chosen most) and gives its 4 to C, which was at zero: magnitudes (0, 2, 4). The mouse stays
# on A for the 0 at t5 and t6, then leaves at t7 for C, the old zero arm.
BLOCK = (
    (0, (4, 2, 0)), (1, (4, 2, 0)), (2, (4, 2, 0)), (0, (4, 2, 0)), (0, (4, 2, 0)),
    (0, (0, 2, 4)), (0, (0, 2, 4)), (2, (0, 2, 4)), (2, (0, 2, 4)),
)  # fmt: skip


def hand(rows):
    return table_from_rows(
        [
            ((1, 1), 0, t, False, PORTS[arm], mags[arm], dict(zip(PORTS, mags, strict=True)))
            for t, (arm, mags) in enumerate(rows)
        ]
    )


def test_a_block_change_event_records_where_the_first_leave_goes():
    events = rt.block_events(rs.mouse_state(hand(BLOCK)))
    row = events.iloc[0]
    assert len(events) == 1
    assert (row.t, row.leave, row.zeroed, row.inherit, row.target) == (5, 7, 0, 2, 2)
    assert (row.v_inherit, row.v_other, row.to_inherit) == (0, 2, 1)  # last payouts say go to B


def test_a_change_the_mouse_does_not_sit_through_is_not_an_event():
    rows = list(BLOCK)
    rows[5] = (1, (0, 2, 4))  # the mouse is on B when the zero moves
    assert rt.block_events(rs.mouse_state(hand(rows))).empty


def test_block_inference_needs_enough_events(monkeypatch):
    state = rs.mouse_state(hand(BLOCK))
    r = rt.block_inference(state).readouts
    assert r["n_block_events"] == 1 and np.isnan(r["block_inherit"])
    monkeypatch.setattr(rt, "MIN_BLOCK_EVENTS", 1)
    r = rt.block_inference(state).readouts
    assert r["block_inherit"] == 1 and r["block_inherit_below"] == 1


def test_a_value_learner_goes_to_the_third_arm_after_a_block_change_more_often_than_chance_says(
    softmax,
):
    state, _ = softmax
    events = rt.block_events(state)
    assert len(events) >= 10
    below = events[events["v_inherit"] < events["v_other"]]
    assert below["to_inherit"].mean() < 0.4  # it follows the last payouts: away from the old zero
