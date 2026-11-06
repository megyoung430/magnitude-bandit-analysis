"""What a simulated value learner's choice rule looks like to the choice logit.

The value learner of :mod:`src.agents.value_agent` chooses with probability
``softmax(beta * Q + kappa * stay + plus_bias * plus)``. :func:`true_utilities` replays one
simulated animal's trials to recover that utility (and the values ``Q``) on every trial, with
the arms in the order of :func:`src.behavior_analysis.choice_logit.design_matrix`.

The choice logit is linear in lagged rewards, but ``Q`` is not: a reward is forgotten by a
factor ``1 - learning_rate`` at each later *visit* of the same arm, not at each later trial, and
an arm that is not chosen keeps its value. :func:`generating_weights` therefore fits the logit
to the generating choice probabilities themselves, as soft targets (cross-entropy, no sampling
noise): the weights a logit with these features would approach with unlimited data from the
agent. Those are the reference weights against which a fit is judged.

Everything here is pure.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.agents.drifting_simulation import plus_ports
from src.behavior_analysis import choice_logit as cl

N_ARMS = 3
REFERENCE_L2 = 1e-6


def _problem_utilities(rows: pd.DataFrame, parameters) -> tuple[np.ndarray, np.ndarray]:
    """``(utilities, q)`` of one problem; *rows* in trial order."""
    learning_rate, beta, kappa, plus_bias = parameters
    ports = sorted(rows["magnitudes"].iloc[0])
    plus = plus_ports(ports, rows["initiation"].iloc[0])
    q = np.zeros(N_ARMS)
    previous = None
    q_before, utilities = [], []
    starts = (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy()
    for choice, reward, start in zip(rows["choice"], rows["reward"], starts, strict=True):
        if start:
            previous = None
        stay = np.zeros(N_ARMS)
        if previous is not None:
            stay[previous] = 1.0
        q_before.append(q.copy())
        utilities.append(beta * q + kappa * stay + plus_bias * plus)
        arm = ports.index(choice)
        q[arm] += learning_rate * (reward - q[arm])
        previous = arm
    return np.array(utilities), np.array(q_before)


def true_utilities(trials: pd.DataFrame, parameters) -> tuple[np.ndarray, np.ndarray]:
    """The generating utility and values of a magnitude-learning agent on every trial.

    Args:
        trials (pd.DataFrame): One animal's rows of the drifting trial table.
        parameters (Sequence[float]): ``(learning_rate, beta, kappa, plus_bias)``, the order of
            ``value_agent.PARAMETER_NAMES``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(utilities, q)``, each ``(n_trials, 3)``, rows in the
        order of ``choice_logit.design_matrix(trials)``; ``q`` is the value before the choice.
    """
    parts = [
        _problem_utilities(rows.sort_values("trial"), parameters)
        for _, rows in trials.groupby("problem", sort=False)
    ]
    utilities, q = (np.concatenate(part) for part in zip(*parts, strict=True))
    return utilities, q


def generating_weights(
    utilities: np.ndarray, X: np.ndarray, feature_names: tuple, l2: float = REFERENCE_L2
) -> np.ndarray:
    """The logit weights closest to the generating choice probabilities, in cross-entropy.

    Args:
        utilities (np.ndarray): ``(n_trials, 3)`` generating utility.
        X (np.ndarray): ``(n_trials, 3, n_features)`` design of the same trials.
        feature_names (tuple[str, ...]): One name per feature.
        l2 (float): Penalty; tiny by default, only to keep the fit finite.

    Returns:
        np.ndarray: ``(n_features,)`` reference weights.

    Raises:
        ValueError: If the shapes disagree.
        RuntimeError: If the fit does not converge.
    """
    if utilities.shape != X.shape[:2]:
        raise ValueError(f"utilities {utilities.shape} do not match the design {X.shape}")
    targets = np.exp(utilities - logsumexp(utilities, axis=1, keepdims=True))
    return cl.fit_soft_targets(X, targets, feature_names, l2).beta


def q_kernel_bounds(parameters, n_lags: int) -> pd.DataFrame:
    """Bounds of the reward weight a value learner implies at each lag.

    A reward followed by ``m`` further visits to the same arm contributes
    ``beta * alpha * (1 - alpha)**m`` to that arm's utility, and ``m`` lies between 0 (the arm
    was not chosen again) and ``k - 1`` (it was chosen on every trial since), for a reward
    ``k`` trials back. The weight a lag-``k`` regressor can take is therefore between
    ``beta * alpha * (1 - alpha)**(k - 1)`` and ``beta * alpha``.

    Args:
        parameters (Sequence[float]): ``(learning_rate, beta, kappa, plus_bias)``.
        n_lags (int): Lags ``1..n_lags``.

    Returns:
        pd.DataFrame: ``lag, lower, upper``.
    """
    learning_rate, beta, *_ = (float(x) for x in parameters)
    lags = np.arange(1, n_lags + 1)
    return pd.DataFrame(
        dict(
            lag=lags,
            lower=beta * learning_rate * (1 - learning_rate) ** (lags - 1),
            upper=np.full(n_lags, beta * learning_rate),
        )
    )
