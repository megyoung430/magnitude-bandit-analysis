"""A four-parameter value learner for the drifting task, with its likelihood and gradient.

On each trial the agent chooses port ``j`` with probability
``softmax(beta * Q + kappa * s + plus_bias * c)_j``, where ``Q`` are its learned values, ``s`` is
1 for the port chosen on the previous trial of the session (0 at a session start) and ``c`` is 1
for a port in the same row or column as the initiation port (a "plus" port, see
:func:`src.behavior_analysis.port_preference.port_category`). After the choice, the chosen
port's value moves toward the reward: ``Q_a <- Q_a + learning_rate * (r - Q_a)``. Values start
at 0 at the start of each problem and carry over between its sessions.

The parameters (``PARAMETER_NAMES``) are the learning rate, the inverse temperature ``beta``
on values, the stickiness ``kappa`` and the spatial bias ``plus_bias``, the last two in logit
units and directly comparable to the ``stay`` and ``plus_port`` weights of
:mod:`src.behavior_analysis.choice_regression`. The spatial bias is needed: without it the
fitted learning rate tends toward 0, because slowly accumulated values are then the only way to
express a fixed preference between ports. The likelihood can still have a slow (learning rate
near 0) and a fast mode for some mice, so fit with several starts.

The likelihood is computed for all of a mouse's problems at once: problems are padded to the
same length and stepped together, which keeps the per-trial Python loop short. The gradient is
exact (forward-mode: the derivative of every value with respect to the learning rate is carried
along).
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Bump when a change alters what this model computes (likelihood, bounds, fitting); comments and
# refactors don't need a bump. The cached agent fits are keyed on it.
MODEL_VERSION = 1
PARAMETER_NAMES = ("learning_rate", "beta", "kappa", "plus_bias")
BOUNDS = ((1e-3, 1.0), (0.0, 30.0), (-5.0, 5.0), (-5.0, 5.0))
# A fitted parameter this close to a bound is reported as at the bound.
AT_BOUND_ATOL = 1e-6


class Sequences(NamedTuple):
    """A mouse's problems as padded arrays, one row per problem.

    Attributes:
        actions (np.ndarray): ``(n_problems, n_steps)`` chosen port index (0-2); 0 where padded.
        rewards (np.ndarray): ``(n_problems, n_steps)`` reward the agent learns from.
        session_start (np.ndarray): ``(n_problems, n_steps)`` True where a session (or a run of
            trials after a dropped trial) starts, so the previous choice is forgotten.
        mask (np.ndarray): ``(n_problems, n_steps)`` True for real trials.
        plus (np.ndarray): ``(n_problems, 3)`` 1.0 for the plus ports of each problem.
    """

    actions: np.ndarray
    rewards: np.ndarray
    session_start: np.ndarray
    mask: np.ndarray
    plus: np.ndarray


def pad_sequences(problems: list[dict]) -> Sequences:
    """Stack per-problem arrays into padded :class:`Sequences`.

    Args:
        problems (list[dict]): One dict per problem with equal-length 1-D arrays ``actions``
            (int), ``rewards`` (float) and ``session_start`` (bool), and ``plus``, a length-3
            array marking the problem's plus ports.

    Returns:
        Sequences: The padded arrays.

    Raises:
        ValueError: If there are no trials, or an action is not 0, 1 or 2.
    """
    n_steps = max((len(p["actions"]) for p in problems), default=0)
    if n_steps == 0:
        raise ValueError("no trials")
    shape = (len(problems), n_steps)
    actions, rewards = np.zeros(shape, dtype=int), np.zeros(shape)
    starts, mask = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for i, p in enumerate(problems):
        n = len(p["actions"])
        actions[i, :n], rewards[i, :n] = p["actions"], p["rewards"]
        starts[i, :n], mask[i, :n] = p["session_start"], True
    if np.any((actions < 0) | (actions > 2)):
        raise ValueError("actions must be port indices 0, 1 or 2")
    plus = np.array([np.asarray(p["plus"], dtype=float) for p in problems]).reshape(-1, 3)
    return Sequences(actions, rewards, starts, mask, plus)


def log_likelihood(params, seqs: Sequences, return_gradient: bool = False):
    """Summed log probability of the chosen ports, and optionally its gradient.

    Args:
        params (Sequence[float]): ``(learning_rate, beta, kappa, plus_bias)``.
        seqs (Sequences): The mouse's problems.
        return_gradient (bool): Also return the gradient. Default False.

    Returns:
        float | tuple[float, np.ndarray]: The log likelihood, with the ``(4,)`` gradient in
        ``PARAMETER_NAMES`` order when asked.
    """
    learning_rate, beta, kappa, plus_bias = (float(p) for p in params)
    n_problems, n_steps = seqs.actions.shape
    rows = np.arange(n_problems)
    q = np.zeros((n_problems, 3))
    dq = np.zeros((n_problems, 3))  # d q / d learning_rate
    previous = np.full(n_problems, -1)
    total, gradient = 0.0, np.zeros(4)
    for t in range(n_steps):
        live = seqs.mask[:, t]
        action, reward = seqs.actions[:, t], seqs.rewards[:, t]
        previous = np.where(seqs.session_start[:, t], -1, previous)
        stay = (np.arange(3)[None, :] == previous[:, None]).astype(float)
        logits = beta * q + kappa * stay + plus_bias * seqs.plus
        log_p = logits - logsumexp(logits, axis=1, keepdims=True)
        total += log_p[rows, action][live].sum()
        if return_gradient:
            p = np.exp(log_p)
            for k, d_logits in enumerate((beta * dq, q, stay, seqs.plus)):
                per_trial = d_logits[rows, action] - (p * d_logits).sum(axis=1)
                gradient[k] += per_trial[live].sum()
        error = reward - q[rows, action]
        chosen_dq = dq[rows, action]
        q_new = q[rows, action] + learning_rate * error
        dq_new = (1 - learning_rate) * chosen_dq + error
        q[rows[live], action[live]] = q_new[live]
        dq[rows[live], action[live]] = dq_new[live]
        previous = np.where(live, action, previous)
    return (total, gradient) if return_gradient else total


class Fit(NamedTuple):
    """Maximum-likelihood parameters of one mouse.

    Attributes:
        parameters (np.ndarray): ``(4,)`` in ``PARAMETER_NAMES`` order.
        log_likelihood (float): Summed log likelihood at the fit.
        n_trials (int): Trials fitted.
    """

    parameters: np.ndarray
    log_likelihood: float
    n_trials: int


def fit(seqs: Sequences, n_starts: int = 5, seed: int = 0) -> Fit:
    """Fit the agent by multi-start L-BFGS-B within ``BOUNDS``.

    Args:
        seqs (Sequences): The mouse's problems.
        n_starts (int): Random starting points. Default 5.
        seed (int): Seed of the starting points. Default 0.

    Returns:
        Fit: The best start.

    Raises:
        ValueError: If *n_starts* < 1 or no start converged.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    rng = np.random.default_rng(seed)
    n_trials = int(seqs.mask.sum())

    def loss(params):
        value, gradient = log_likelihood(params, seqs, return_gradient=True)
        return -value / n_trials, -gradient / n_trials

    results = []
    for _ in range(n_starts):
        start = [
            rng.uniform(0.05, 0.8),
            rng.uniform(0.2, 3.0),
            rng.uniform(-0.5, 2.0),
            rng.uniform(-0.5, 1.5),
        ]
        result = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=BOUNDS)
        if result.success:
            results.append(result)
    if not results:
        raise ValueError("no start of the value-agent fit converged")
    best = min(results, key=lambda r: r.fun)
    return Fit(best.x, float(-best.fun * n_trials), n_trials)


def parameters_at_bounds(parameters) -> tuple[str, ...]:
    """Names of the parameters within ``AT_BOUND_ATOL`` of a bound in ``BOUNDS``.

    A parameter at a bound means the optimum lies outside the allowed range, so its value (and
    the likelihood comparison) should be read with care.
    """
    return tuple(
        name
        for name, value, (low, high) in zip(PARAMETER_NAMES, parameters, BOUNDS, strict=True)
        if abs(value - low) <= AT_BOUND_ATOL or abs(value - high) <= AT_BOUND_ATOL
    )


def choice_probabilities(q, previous: int | None, plus, params) -> np.ndarray:
    """The agent's choice probabilities for one trial.

    Args:
        q (np.ndarray): ``(3,)`` current values.
        previous (int | None): Port chosen on the previous trial of the session, if any.
        plus (np.ndarray): ``(3,)`` 1.0 for the plus ports.
        params (Sequence[float]): ``(learning_rate, beta, kappa, plus_bias)``.

    Returns:
        np.ndarray: ``(3,)`` probabilities.
    """
    _, beta, kappa, plus_bias = params
    logits = beta * np.asarray(q, dtype=float) + plus_bias * np.asarray(plus, dtype=float)
    if previous is not None:
        logits[previous] += kappa
    return np.exp(logits - logsumexp(logits))
