"""The generating utilities of a simulated value learner and their best linear-kernel summary."""

import numpy as np
import pytest
from scipy.special import logsumexp

from src.agents import choice_logit_truth as truth
from src.agents import drifting_simulation as sim
from src.agents import value_agent
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis.drifting_trial_table import drifting_trial_table

PARAMETERS = (0.3, 1.2, 0.8, 0.5)  # learning rate, beta, kappa, plus bias


def simulated_table(seed=0):
    rng = np.random.default_rng(seed)
    rules = sim.TaskRules(0.15, np.arange(30, 36))
    problems = {}
    for k, ports in enumerate((["A1", "B3", "C2"], ["A3", "B2", "C1"]), start=1):
        layout = sim.MouseLayout(ports, "B2", dict(zip(ports, (4, 2, 0), strict=True)), [120, 90])
        played = sim.simulate_problem(PARAMETERS, "magnitude", layout, rules, rng)
        problems[k] = {"q": sim.simulated_sessions("q", k, layout, played)}
    return drifting_trial_table(problems)


@pytest.mark.filterwarnings("ignore:the problems")
def test_true_utilities_reproduce_the_value_agent_likelihood():
    table = simulated_table()
    utilities, _ = truth.true_utilities(table, PARAMETERS)
    design = cl.design_matrix(table, n_lags=2)
    log_p = utilities - logsumexp(utilities, axis=1, keepdims=True)
    total = log_p[np.arange(len(design.chosen)), design.chosen].sum()
    sequences = sim.problem_sequences(table, "magnitude")
    assert total == pytest.approx(value_agent.log_likelihood(PARAMETERS, sequences))


def test_true_q_values_start_at_zero_and_follow_the_delta_rule():
    table = simulated_table()
    _, q = truth.true_utilities(table, PARAMETERS)
    ordered = table[table["problem"] == 1].sort_values("trial")
    ports = sorted(ordered["magnitudes"].iloc[0])
    arm = ports.index(ordered["choice"].iloc[0])
    assert (q[0] == 0).all()
    assert q[1, arm] == pytest.approx(PARAMETERS[0] * ordered["reward"].iloc[0])


def test_generating_weights_recover_an_exactly_linear_utility():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3, 4))
    beta = np.array([0.5, -0.2, 1.0, 0.0])
    reference = truth.generating_weights(X @ beta, X, tuple("abcd"))
    np.testing.assert_allclose(reference, beta, atol=1e-3)


def test_generating_weights_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="utilities"):
        truth.generating_weights(np.zeros((5, 3)), np.zeros((6, 3, 2)), ("a", "b"))


def test_q_kernel_bounds_are_the_geometric_and_the_flat_weight():
    bounds = truth.q_kernel_bounds((0.25, 2.0, 0.0, 0.0), n_lags=3)
    assert bounds["lag"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(bounds["upper"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(bounds["lower"], [0.5, 0.375, 0.28125])
