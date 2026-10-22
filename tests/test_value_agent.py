"""The four-parameter value agent: likelihood, exact gradient, and parameter recovery."""

import numpy as np
import pytest
from scipy.optimize import check_grad

from src.agents import value_agent

PLUS = np.array([0.0, 1.0, 0.0])


def _simulate(params, n_problems=3, n_trials=600, session_length=60, seed=0):
    """Choices of the agent on a bandit whose arm rewards switch every 40 trials."""
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(n_problems):
        q, previous = np.zeros(3), None
        actions, rewards, starts = [], [], []
        for t in range(n_trials):
            if t % session_length == 0:
                previous = None
            payoff = np.roll([4.0, 1.0, 0.0], (t // 40) % 3)
            a = rng.choice(3, p=value_agent.choice_probabilities(q, previous, PLUS, params))
            q[a] += params[0] * (payoff[a] - q[a])
            actions.append(a)
            rewards.append(payoff[a])
            starts.append(t % session_length == 0)
            previous = a
        problems.append(dict(actions=actions, rewards=rewards, session_start=starts, plus=PLUS))
    return value_agent.pad_sequences(problems)


def test_likelihood_of_two_trials_by_hand():
    seqs = value_agent.pad_sequences(
        [dict(actions=[0, 0], rewards=[2.0, 0.0], session_start=[True, False], plus=PLUS)]
    )
    lr, beta, kappa, bias = 0.5, 1.0, 0.3, 0.2
    first = value_agent.choice_probabilities(np.zeros(3), None, PLUS, (lr, beta, kappa, bias))
    second = value_agent.choice_probabilities(
        np.array([1.0, 0, 0]), 0, PLUS, (lr, beta, kappa, bias)
    )
    expected = np.log(first[0]) + np.log(second[0])
    assert value_agent.log_likelihood((lr, beta, kappa, bias), seqs) == pytest.approx(expected)


def test_padding_does_not_change_the_likelihood():
    short = dict(actions=[1, 2], rewards=[1.0, 0.0], session_start=[True, False], plus=PLUS)
    long = dict(actions=[0, 1, 1, 2], rewards=[4.0, 1, 1, 0], session_start=[True, 0, 0, 0],
                plus=PLUS)  # fmt: skip
    params = (0.4, 1.2, 0.5, 0.1)
    together = value_agent.log_likelihood(params, value_agent.pad_sequences([short, long]))
    apart = sum(value_agent.log_likelihood(params, value_agent.pad_sequences([p]))
                for p in (short, long))  # fmt: skip
    assert together == pytest.approx(apart)


def test_gradient_is_exact():
    seqs = _simulate((0.4, 1.0, 1.0, 0.5), n_problems=2, n_trials=200)
    params = np.array([0.3, 0.8, 0.6, 0.2])
    error = check_grad(
        lambda p: value_agent.log_likelihood(p, seqs),
        lambda p: value_agent.log_likelihood(p, seqs, return_gradient=True)[1],
        params,
    )
    gradient = value_agent.log_likelihood(params, seqs, return_gradient=True)[1]
    assert error / np.linalg.norm(gradient) < 1e-5


def test_fit_recovers_parameters():
    true = (0.4, 1.0, 1.2, 0.6)
    fit = value_agent.fit(_simulate(true, seed=3), n_starts=3, seed=0)
    assert fit.n_trials == 1800
    assert fit.parameters[0] == pytest.approx(true[0], abs=0.15)
    assert fit.parameters[1] == pytest.approx(true[1], abs=0.3)
    assert fit.parameters[2] == pytest.approx(true[2], abs=0.3)
    assert fit.parameters[3] == pytest.approx(true[3], abs=0.3)


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="no trials"):
        value_agent.pad_sequences([])
    bad = dict(actions=[3], rewards=[0.0], session_start=[True], plus=PLUS)
    with pytest.raises(ValueError, match="port indices"):
        value_agent.pad_sequences([bad])
    with pytest.raises(ValueError, match="n_starts"):
        value_agent.fit(_simulate((0.4, 1.0, 1.0, 0.5), n_problems=1, n_trials=10), n_starts=0)
