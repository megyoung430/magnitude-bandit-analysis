"""Step against graded leaving, and choices around believed and true crossings."""

import numpy as np
import pytest

from src.behavior_analysis import rank_crossings as rc
from src.behavior_analysis import rank_decisions as rd
from src.behavior_analysis import rank_state as rs
from tests.rank_fixtures import rank_mouse, softmax_mouse
from tests.switch_regression_fixtures import table_from_rows

PORTS = ("A1", "B3", "C2")
SIZE = dict(n_problems=8, sessions=3, trials=200)


def hand(rows):
    """``rows``: (choice arm, magnitudes of A1, B3, C2); one problem, one session."""
    return table_from_rows(
        [
            ((1, 1), 0, t, False, PORTS[arm], mags[arm], dict(zip(PORTS, mags, strict=True)))
            for t, (arm, mags) in enumerate(rows)
        ]
    )


# A pays 4 and is the arm in use while B (2) and C (1) are the alternatives; at t5 A's payout
# falls to 1, below B's believed 2: a believed crossing to B. At t7 B pays 0, below A's believed 1
# and C's 1 (a tie for the best alternative), which is not an event.
BELIEVED_TOY = (
    (0, (4, 2, 1)), (1, (4, 2, 1)), (2, (4, 2, 1)),  # visit every arm
    (0, (4, 2, 1)), (0, (4, 2, 1)),  # A pays 4, 4
    (0, (1, 2, 1)),  # t5: A pays 1 < B's 2
    (1, (1, 2, 1)),  # t6: moved to B, pays 2
    (1, (1, 0, 1)),  # t7: B pays 0; A and C both believed 1: tied alternatives
    (1, (1, 0, 1)),
)  # fmt: skip


def test_a_believed_crossing_is_a_payout_dropping_below_the_best_alternative():
    state = rs.mouse_state(hand(BELIEVED_TOY))
    events = rc.believed_crossings(state)
    assert events.to_dict("list") == {"t": [5], "own": [0], "new_best": [1]}


def test_true_best_change_names_the_previous_and_new_best():
    rows = [(0, (4, 2, 1))] * 4 + [(0, (1, 4, 2))] * 3  # best A, then B at t4
    events = rc.true_crossings(rs.mouse_state(hand(rows)), 1)
    assert events.to_dict("list") == {"t": [4], "prev": [0], "next": [1]}


def test_a_second_best_change_needs_an_unchanged_best():
    rows = [(0, (4, 2, 1))] * 3 + [(0, (4, 1, 2))] * 2 + [(0, (1, 4, 2))] * 2
    state = rs.mouse_state(hand(rows))
    second = rc.true_crossings(state, 2)
    assert second.to_dict("list") == {"t": [3], "prev": [1], "next": [2]}
    assert rc.true_crossings(state, 1)["t"].tolist() == [5]  # the best changes at t5, not counted


def test_an_unknown_rank_raises():
    with pytest.raises(ValueError, match="rank must be"):
        rc.true_crossings(rs.mouse_state(hand(BELIEVED_TOY)), 3)


def test_aligned_choices_give_the_share_choosing_each_role_around_the_event():
    state = rs.mouse_state(hand(BELIEVED_TOY))
    profile = rc.aligned_choices(state, np.array([5]), {"own": np.array([0]), "new": np.array([1])})
    at = profile.set_index(["role", "k"])
    assert at.loc[("own", 0), "p"] == 1 and at.loc[("new", 1), "p"] == 1  # t5 on A, t6 on B
    assert at.loc[("new", 3), "n"] == 1 and at.loc[("new", 4), "n"] == 0  # the problem ends at t8
    assert np.isnan(at.loc[("new", 4), "p"])
    assert at.loc[("own", -5), "n"] == 1  # t0, the problem's first trial


def test_the_rank_agent_leaves_abruptly_at_a_crossing_and_the_softmax_learner_does_not():
    rank = rd.decision_table(rs.mouse_state(rank_mouse(seed=1, **SIZE)))
    soft = rd.decision_table(rs.mouse_state(softmax_mouse(seed=1, **SIZE)))
    r, s = rc.crossing_step(rank).readouts, rc.crossing_step(soft).readouts
    assert r["step_weight"] > 4 * r["se_step"] > 0
    assert abs(s["step_weight"]) < 2 * s["se_step"]
    assert r["gain_step_vs_smooth"] > 0.01 > -0.01 > s["gain_step_vs_smooth"]
    assert r["gain_indicator"] > 0.01 > s["gain_indicator"]


def test_the_leaving_curve_bins_the_difference_and_pools_beyond_the_clip():
    rank = rd.decision_table(rs.mouse_state(rank_mouse(seed=1, **SIZE)))
    curve = rc.crossing_step(rank).curves
    assert curve["x"].min() == -rc.D_CLIP and curve["x"].max() == rc.D_CLIP
    assert curve["n"].sum() == len(rank)
    below = curve.loc[curve["x"] < 0, "y"]
    assert (below > 0.8).all() and (curve.loc[curve["x"] > 0, "y"] < 0.2).all()


def test_the_rank_agent_adopts_the_new_best_at_once_after_a_believed_crossing():
    state = rs.mouse_state(rank_mouse(seed=1, **SIZE))
    r = rc.alignment_readouts(state).readouts
    assert r["cross_leave_1"] > 0.85 and r["cross_adopt_1"] > 0.85 and r["cross_t50"] == 1
    assert r["n_cross_believed"] >= rc.MIN_EVENTS


def test_the_softmax_learner_leaves_less_often_at_the_first_trial_after_a_crossing():
    rank = rc.alignment_readouts(rs.mouse_state(rank_mouse(seed=1, **SIZE))).readouts
    soft = rc.alignment_readouts(rs.mouse_state(softmax_mouse(seed=1, **SIZE))).readouts
    assert soft["cross_leave_1"] < rank["cross_leave_1"] - 0.05


def test_too_few_events_give_nan_summaries_and_still_count_the_events():
    r = rc.alignment_readouts(rs.mouse_state(hand(BELIEVED_TOY))).readouts
    assert r["n_cross_believed"] == 1 and np.isnan(r["cross_leave_1"])
    assert np.isnan(r["cross_t50"]) and np.isnan(r["true_best_t50"])


def test_curves_name_the_roles_around_each_kind_of_event():
    curves = rc.alignment_readouts(rs.mouse_state(rank_mouse(seed=1, **SIZE))).curves
    series = curves.groupby("curve")["series"].agg(lambda s: sorted(set(s)))
    assert series["cross_believed"] == ["new_best", "other", "own"]
    assert series["cross_true_best"] == ["next", "other", "prev"]
    assert curves["x"].min() == rc.WINDOW[0] and curves["x"].max() == rc.WINDOW[1]
