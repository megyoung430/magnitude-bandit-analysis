"""Heuristic+: WSLS (Magnitude) staying, switches on recency, believed value and never visited."""

import warnings

import numpy as np
import pytest
from scipy.special import expit, log_expit

from src.agents import drifting_simulation as sim
from src.agents import heuristic_agent as ha
from src.agents import null_agents as na
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from tests.switch_regression_fixtures import table_from_rows

PORTS = ("A1", "B3", "C2")
RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
LAYOUT = sim.MouseLayout(list(PORTS), "B2", {"A1": 4, "B3": 2, "C2": 0}, [300] * 8)
PARAMS = ha.HeuristicPlusParams(a_win=0.4, a_lose=-0.3, b=0.2, w=0.7, v=0.5, u=-0.9)


def simulated(params, seed=0, mouse="m", layout=LAYOUT):
    sessions = sim.play_problem(
        ha.HeuristicPolicy(params), layout, RULES, np.random.default_rng(seed)
    )
    problems = {1: {mouse: sim.simulated_sessions(mouse, 1, layout, sessions)}}
    return drifting_trial_table(problems)


def toy_table():
    """One session: B (start), A, A, C. Arms sort as A1 < B3 < C2; magnitudes (A, B, C)."""
    mags = {0: (1, 3, 0), 1: (2, 3, 0), 2: (2, 3, 0), 3: (2, 3, 0)}
    choices = {0: 1, 1: 0, 2: 0, 3: 2}
    rows = []
    for t in range(4):
        row_mags = dict(zip(PORTS, mags[t], strict=True))
        rows.append((1, 0, t, False, PORTS[choices[t]], mags[t][choices[t]], row_mags))
    return table_from_rows(rows)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_non_finite_parameters_and_a_decay_outside_zero_one_are_rejected():
    with pytest.raises(ValueError, match="w must be a finite number"):
        ha.HeuristicPlusParams(0.4, -0.3, 0.2, float("nan"), 0.5, 0.0)
    with pytest.raises(ValueError, match="decay must be in"):
        ha.HeuristicPlusParams(0.4, -0.3, 0.2, 0.7, 0.5, 0.0, decay=1.0)


def test_the_stay_rule_is_the_one_of_wsls_magnitude():
    wsls = na.WslsMagnitudeParams(PARAMS.a_win, PARAMS.a_lose, PARAMS.b, PARAMS.w)
    magnitudes = np.array([0, 1, 2, 3, 4])
    assert np.allclose(PARAMS.stay_logit(magnitudes), wsls.stay_logit(magnitudes))


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def test_likelihood_equals_a_hand_computation_on_a_toy_sequence():
    # r0 start: log(1/3).
    # r1 B -> A, prev paid 3: log(1 - s(0.4 + 0.2 * 2)); alternatives A, C both unvisited and
    #    untraced, so the switch logit is 0: log(1/2).
    # r2 A -> A, prev paid 2: log s(0.4 + 0.2 * 1).
    # r3 A -> C, prev paid 2: log(1 - s(0.6)); alternatives B (first) and C (second): trace
    #    h_B = 0.9**2 (chosen 3 trials ago), h_C = 0; believed B = 3 (paid at r0), C = 0; B is
    #    visited and C is not (never-visited difference -1). C is the second alternative:
    #    log s(-(0.7 * 0.81 + 0.5 * 3 + (-0.9) * (-1))).
    pairs = (
        log_expit(-(0.4 + 0.2 * 2)) + np.log(0.5)
        + log_expit(0.4 + 0.2 * 1)
        + log_expit(-(0.4 + 0.2 * 1)) + log_expit(-(0.7 * 0.81 + 0.5 * 3 + 0.9))
    )  # fmt: skip
    assert ha.log_likelihood(toy_table(), PARAMS) == pytest.approx(np.log(1 / 3) + pairs)
    assert ha.log_likelihood(toy_table(), PARAMS, include_starts=False) == pytest.approx(pairs)


def test_with_no_believed_or_never_visited_weight_it_is_wsls_magnitude():
    trials = simulated(PARAMS, seed=3)
    wsls = na.WslsMagnitudeParams(PARAMS.a_win, PARAMS.a_lose, PARAMS.b, PARAMS.w)
    reduced = ha.HeuristicPlusParams(PARAMS.a_win, PARAMS.a_lose, PARAMS.b, PARAMS.w, 0.0, 0.0)
    assert ha.log_likelihood(trials, reduced) == pytest.approx(na.log_likelihood(trials, wsls))


def test_the_policy_assigns_the_observed_choices_the_likelihood_of_the_table():
    trials = simulated(PARAMS, seed=5)
    policy, total = ha.HeuristicPolicy(PARAMS), 0.0
    for row in trials.sort_values("trial").itertuples():
        if row.trial_in_session == 0:
            policy.start_session()
        arm = PORTS.index(row.choice)
        total += np.log(policy.probabilities()[arm])
        policy.update(arm, row.reward)
    assert total == pytest.approx(ha.log_likelihood(trials, PARAMS))


def test_choice_probabilities_sum_to_one_and_follow_the_switch_logit():
    h, believed, visited = np.array([2.0, 0.5, 1.0]), np.array([1.0, 4.0, 2.0]), np.ones(3, bool)
    p = ha.choice_probabilities(PARAMS, 0, 3.0, h, believed, visited)
    assert p.sum() == pytest.approx(1.0)
    stay = expit(PARAMS.stay_logit(3.0))
    to_first = expit(0.7 * (0.5 - 1.0) + 0.5 * (4.0 - 2.0))  # arms 1 and 2 follow arm 0
    assert p[0] == pytest.approx(stay)
    assert p[1] == pytest.approx((1 - stay) * to_first)
    p_start = ha.choice_probabilities(PARAMS, None, 0.0, h, believed, visited)
    assert np.allclose(p_start, 1 / 3)


def test_the_policy_keeps_believed_values_across_sessions_and_clears_the_trace():
    policy = ha.HeuristicPolicy(PARAMS)
    policy.update(1, 3.0)
    policy.start_session()
    assert np.array_equal(policy.believed, [0.0, 3.0, 0.0])
    assert np.array_equal(policy.visited, [False, True, False])
    assert policy.previous is None and not policy.trace.any()


# ---------------------------------------------------------------------------
# Fit and recovery
# ---------------------------------------------------------------------------


def test_the_fit_recovers_the_believed_value_weight_it_was_simulated_with():
    layout = sim.MouseLayout(list(PORTS), "B2", {"A1": 4, "B3": 2, "C2": 0}, [400] * 10)
    estimates = []
    for seed in range(4):
        trials = simulated(PARAMS, seed=seed, layout=layout)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimates.append(ha.fit(trials))
    for name, tolerance in (("v", 0.15), ("w", 0.15), ("b", 0.15), ("a_win", 0.25)):
        mean = np.mean([getattr(e, name) for e in estimates])
        assert mean == pytest.approx(getattr(PARAMS, name), abs=tolerance), name


def test_a_mouse_that_ignores_believed_value_gets_a_believed_weight_near_zero():
    blind = ha.HeuristicPlusParams(0.4, -0.3, 0.2, 0.7, 0.0, 0.0)
    layout = sim.MouseLayout(list(PORTS), "B2", {"A1": 4, "B3": 2, "C2": 0}, [400] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ha.fit(simulated(blind, seed=1, layout=layout))
    assert abs(fitted.v) < 0.15


def test_the_fit_names_the_mouse_when_there_are_too_few_switches():
    stays = table_from_rows(
        [(1, 0, t, False, "A1", 2, {"A1": 2, "B3": 1, "C2": 0}) for t in range(60)], mouse="mm"
    )
    with pytest.raises(ValueError, match="mouse mm"):
        ha.fit(stays)
