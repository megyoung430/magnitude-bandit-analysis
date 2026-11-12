"""Stay models on a toy rank agent and a softmax learner, whose answers are known."""

import numpy as np
import pytest

from src.behavior_analysis import rank_decisions as rd
from src.behavior_analysis import rank_state as rs
from src.behavior_analysis import rank_stay as st
from tests.rank_fixtures import rank_mouse, softmax_mouse


def decisions(trials):
    return rd.decision_table(rs.mouse_state(trials))


@pytest.fixture(scope="module")
def rank():
    return decisions(rank_mouse(seed=1))


@pytest.fixture(scope="module")
def softmax():
    return decisions(softmax_mouse(seed=1))


def test_the_rank_agent_stays_by_the_best_alternative_and_ignores_the_worst(rank):
    r = st.stay_weights(rank).readouts
    assert r["w_best"] < -0.5
    assert abs(r["w_worst"]) < 2 * r["se_worst"]
    assert r["w_diff"] > 0.5  # worst minus best


def test_the_softmax_learner_stays_less_by_both_alternatives(softmax):
    r = st.stay_weights(softmax).readouts
    assert r["w_best"] < -0.5 and r["w_worst"] < -0.3
    assert 0.3 < r["w_ratio"] < 1.5  # the worst counts, about as much as the best


def test_the_contrast_error_covers_the_difference_of_the_weights(rank):
    r = st.stay_weights(rank).readouts
    assert r["se_diff"] > 0 and r["w_diff"] == pytest.approx(r["w_worst"] - r["w_best"])


def test_a_prefix_renames_every_readout(rank):
    names = st.stay_weights(rank, prefix="q_").readouts
    assert all(name.startswith("q_") for name in names)


def test_the_rank_agent_is_better_predicted_by_ranks_and_the_softmax_by_values(rank, softmax):
    r = st.value_or_rank(rank).readouts
    s = st.value_or_rank(softmax).readouts
    assert r["rank_minus_value"] > 0.005
    assert s["rank_minus_value"] < -0.01
    assert r["rank_over_value"] > 0 and s["value_over_rank"] > 0.01


def test_the_gains_are_differences_of_the_held_out_nats(softmax):
    r = st.value_or_rank(softmax).readouts
    assert r["rank_over_value"] == pytest.approx(r["nats_value"] - r["nats_both"])
    assert r["rank_minus_value"] == pytest.approx(r["nats_value"] - r["nats_rank"])


def test_staying_at_a_given_rank_is_invariant_to_scale_for_the_rank_agent_only(rank, softmax):
    r = st.scale_invariance(rank).readouts
    s = st.scale_invariance(softmax).readouts
    assert abs(r["scale_slope_first"]) < 2.5 * r["se_scale_first"]
    assert s["scale_slope_first"] > 5 * s["se_scale_first"]


def test_the_scale_curve_lists_each_rank_and_payout(softmax):
    curve = st.scale_invariance(softmax).curves
    assert set(curve["series"]) <= {"first", "second"}
    assert curve["y"].between(0, 1).all() and (curve["n"] > 0).all()


def test_a_mouse_that_never_leaves_raises():
    table = decisions(rank_mouse(seed=1)).assign(leave=0.0)
    with pytest.raises(ValueError, match="one class"):
        st.stay_weights(table)


def test_one_problem_cannot_be_cross_validated(rank):
    with pytest.raises(ValueError, match="two groups"):
        st.value_or_rank(rank.assign(problem=0))


def test_stay_outcome_is_one_where_the_mouse_stays(rank):
    assert np.array_equal(st.stay_outcome(rank), 1 - rank["leave"].to_numpy())
