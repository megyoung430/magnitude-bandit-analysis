"""A Q-learner with a choice kernel, forgetting and asymmetric learning rates (drifting task).

The model follows Aloor, Sit, Geuter, Tutt & Duan (bioRxiv 2026.10.31.763226, Methods "Q-learning
model", eqs 7-14), adapted to three arms with graded rewards. On each trial the agent chooses arm
``j`` with probability ``softmax(beta * Q + beta_k * K + plus_bias * c)_j``, where ``Q`` are its
learned values, ``K`` its choice kernel and ``c`` is 1 for a port in the same row or column as the
initiation port (a "plus" port, see :func:`src.behavior_analysis.port_preference.port_category`).
After the choice, with ``delta = r - Q_chosen`` (``r`` is the pulses paid):

- the chosen value moves by ``alpha_pos * delta`` if ``delta > 0`` and by ``alpha_neg * delta``
  otherwise (eqs 9-11). With graded rewards ``delta > 0`` means *better than expected*, not
  *rewarded*: a 1-pulse reward from an arm expected to pay 3 lowers the value;
- every unchosen value keeps the fraction ``zeta`` of itself (eq 12), so values of arms that are
  not visited fade toward 0;
- the chosen arm's kernel moves toward 1 and every other kernel toward 0 at rate ``alpha_k``
  (eqs 13-14). It is reward independent: a trace of recent choices. The WSLS choice trace
  ``h <- 0.9 h + chosen`` of :mod:`src.agents.null_agents` is this kernel with ``alpha_k = 0.1``,
  scaled by 10.

**Differences from the paper.** (1) The paper has two arms and one left-right bias ``b``; here the
spatial term is the same plus-port bias as :mod:`src.agents.value_agent`, so the two Q-learners
differ only in the learning rule and the kernel. (2) The paper resets Q and K at every session
start; here values start at 0 for a problem and carry over its sessions (as the value agent's) and
the kernel is cleared at each session start, or after a dropped trial, as the value agent's
previous choice is. (3) The unit-interval rates are bounded at 1e-3 from 0 where the paper uses 0
(an exact 0 makes a value or kernel that never moves); ``zeta`` may reach 0 and 1.

**Special cases.** ``alpha_pos = alpha_neg``, ``zeta = 1`` and ``alpha_k = 1`` (a one-trial
kernel, the previous choice) is the value agent, with ``beta_k`` as its stickiness ``kappa``
(:func:`value_agent_parameters`). The ablations in ``VARIANTS`` remove one piece of the full model
each and are fitted by the same code, holding or tying the removed parameters.

The likelihood is computed for all of a mouse's problems at once, as in the value agent, with the
exact forward-mode gradient: the derivative of every value with respect to the three parameters
that move values (``alpha_pos``, ``alpha_neg``, ``zeta``) and of every kernel with respect to
``alpha_k`` is carried along. Fit with several starts: the likelihood has local optima.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

PARAMETER_NAMES = ("alpha_pos", "alpha_neg", "zeta", "alpha_k", "beta", "beta_k", "plus_bias")
INDEX = {name: i for i, name in enumerate(PARAMETER_NAMES)}
BOUNDS_BY_NAME = {
    "alpha_pos": (1e-3, 1.0),
    "alpha_neg": (1e-3, 1.0),
    "zeta": (0.0, 1.0),
    "alpha_k": (1e-3, 1.0),
    "beta": (1e-3, 20.0),
    "beta_k": (-20.0, 20.0),
    "plus_bias": (-5.0, 5.0),
}
BOUNDS = tuple(BOUNDS_BY_NAME[name] for name in PARAMETER_NAMES)
# Random starts are drawn uniformly from these ranges (inside the bounds, on the scale mice fit).
START_RANGES = {
    "alpha_pos": (0.05, 0.8),
    "alpha_neg": (0.05, 0.8),
    "zeta": (0.3, 1.0),
    "alpha_k": (0.05, 0.8),
    "beta": (0.2, 3.0),
    "beta_k": (-0.5, 3.0),
    "plus_bias": (-0.5, 1.5),
}
START_CENTERS = {name: float(np.mean(r)) for name, r in START_RANGES.items()}
# A fitted parameter this close to a bound is reported as at the bound.
AT_BOUND_ATOL = 1e-6
# The parameters that are rates or retention fractions in [0, 1].
_UNIT_INTERVAL = ("alpha_pos", "alpha_neg", "zeta", "alpha_k")


class Variant(NamedTuple):
    """The full model or an ablation of it: which parameters are free, held, or tied.

    Attributes:
        name (str): Key in ``VARIANTS``.
        label (str): Display name.
        free (tuple[str, ...]): Fitted parameters, in ``PARAMETER_NAMES`` order.
        fixed (dict[str, float]): Parameters held at a value.
        tied (dict[str, str]): ``{parameter: parameter it copies}``.
    """

    name: str
    label: str
    free: tuple[str, ...]
    fixed: dict
    tied: dict


def _variant(name: str, label: str, fixed: dict | None = None, tied: dict | None = None) -> Variant:
    fixed, tied = fixed or {}, tied or {}
    free = tuple(n for n in PARAMETER_NAMES if n not in fixed and n not in tied)
    return Variant(name, label, free, fixed, tied)


# With beta_k = 0 the kernel never enters the choice, so alpha_k is unidentified and held at an
# arbitrary value inside its bounds.
VARIANTS = {
    v.name: v
    for v in (
        _variant("full", "Q-Learner (Kernel + Forgetting)"),
        _variant("no_asymmetry", "No Asymmetry", tied={"alpha_neg": "alpha_pos"}),
        _variant("no_forgetting", "No Forgetting", fixed={"zeta": 1.0}),
        _variant("no_kernel", "No Kernel", fixed={"beta_k": 0.0, "alpha_k": 0.5}),
        _variant(
            "kernel_only", "Kernel Only", fixed={"zeta": 1.0}, tied={"alpha_neg": "alpha_pos"}
        ),
    )
}


def expand(variant: Variant, free: Sequence[float]) -> tuple[float, ...]:
    """All seven parameters of *variant* from its free values.

    Args:
        variant (Variant): The model variant.
        free (Sequence[float]): Values of ``variant.free``, in that order.

    Returns:
        tuple[float, ...]: ``PARAMETER_NAMES`` order, with fixed values set and tied ones copied.

    Raises:
        ValueError: If *free* has the wrong length.
    """
    if len(free) != len(variant.free):
        raise ValueError(
            f"{variant.name} has {len(variant.free)} free parameters {variant.free}, "
            f"got {len(free)} values"
        )
    values = {**dict(zip(variant.free, free, strict=True)), **variant.fixed}
    values.update({name: values[source] for name, source in variant.tied.items()})
    return tuple(float(values[name]) for name in PARAMETER_NAMES)


def value_agent_parameters(
    learning_rate: float, beta: float, kappa: float, plus_bias: float
) -> tuple[float, ...]:
    """The parameters of this model that make it the value agent with the given parameters."""
    return (learning_rate, learning_rate, 1.0, 1.0, beta, kappa, plus_bias)


def _check_parameters(params) -> np.ndarray:
    values = np.asarray(params, dtype=float)
    if values.shape != (len(PARAMETER_NAMES),):
        raise ValueError(f"expected seven parameters {PARAMETER_NAMES}, got {values.size} values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must be finite, got {values.tolist()}")
    for name in _UNIT_INTERVAL:
        if not 0.0 <= values[INDEX[name]] <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {values[INDEX[name]]}")
    return values


def choice_probabilities(q, k, plus, params) -> np.ndarray:
    """The agent's choice probabilities for one trial.

    Args:
        q (np.ndarray): ``(3,)`` current values.
        k (np.ndarray): ``(3,)`` current choice kernel.
        plus (np.ndarray): ``(3,)`` 1.0 for the plus ports.
        params (Sequence[float]): ``PARAMETER_NAMES`` order.

    Returns:
        np.ndarray: ``(3,)`` probabilities.
    """
    _, _, _, _, beta, beta_k, plus_bias = params
    logits = (
        beta * np.asarray(q, dtype=float)
        + beta_k * np.asarray(k, dtype=float)
        + plus_bias * np.asarray(plus, dtype=float)
    )
    return np.exp(logits - logsumexp(logits))


def update_state(q, k, action: int, reward: float, params) -> tuple[np.ndarray, np.ndarray]:
    """Values and kernel after choosing *action* and receiving *reward* (eqs 9-14).

    Args:
        q (np.ndarray): ``(3,)`` values before the choice.
        k (np.ndarray): ``(3,)`` kernel before the choice.
        action (int): The arm chosen (0-2).
        reward (float): What it paid.
        params (Sequence[float]): ``PARAMETER_NAMES`` order.

    Returns:
        tuple[np.ndarray, np.ndarray]: New ``(q, k)``; the inputs are not modified.
    """
    alpha_pos, alpha_neg, zeta, alpha_k = params[:4]
    chosen = np.arange(len(q)) == action
    delta = reward - q[action]
    rate = alpha_pos if delta > 0 else alpha_neg
    new_q = np.where(chosen, q + rate * delta, zeta * q)
    new_k = k + alpha_k * (chosen - k)
    return new_q, new_k


def _softmax(logits) -> tuple[np.ndarray, np.ndarray]:
    """``(log p, p)`` of ``(n, 3)`` logits; inline because scipy logsumexp dominated the fit."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return log_p, np.exp(log_p)


def _step_gradient(beta, beta_k, q, k, dq, dk, plus, p, action) -> np.ndarray:
    """``(n, 7)`` d log p(chosen) / d parameter for every problem, in ``PARAMETER_NAMES`` order.

    *d_logits* holds the derivative of every arm's logit with respect to each parameter; the
    derivative of the log softmax at the chosen arm is its own minus the probability-weighted one.
    """
    d_logits = np.empty((len(action), 3, len(PARAMETER_NAMES)))
    d_logits[:, :, :3] = beta * dq
    d_logits[:, :, INDEX["alpha_k"]] = beta_k * dk
    d_logits[:, :, INDEX["beta"]] = q
    d_logits[:, :, INDEX["beta_k"]] = k
    d_logits[:, :, INDEX["plus_bias"]] = plus
    return d_logits[np.arange(len(action)), action] - (p[:, :, None] * d_logits).sum(axis=1)


def _forward(params, seqs, return_gradient: bool):
    """``(per-trial log probabilities, gradient of their sum)``; see :func:`log_likelihood`.

    The first is ``(n_problems, n_steps)`` with NaN at padded steps; the second is ``None``
    unless *return_gradient*.
    """
    alpha_pos, alpha_neg, zeta, alpha_k, beta, beta_k, plus_bias = _check_parameters(params)
    n_problems, n_steps = seqs.actions.shape
    rows, arms = np.arange(n_problems), np.arange(3)
    q, k = np.zeros((n_problems, 3)), np.zeros((n_problems, 3))
    dq = np.zeros((n_problems, 3, 3))  # d q / d (alpha_pos, alpha_neg, zeta)
    dk = np.zeros((n_problems, 3))  # d k / d alpha_k
    per_trial = np.full((n_problems, n_steps), np.nan)
    gradient = np.zeros(len(PARAMETER_NAMES))
    for t in range(n_steps):
        live = seqs.mask[:, t]
        action, reward = seqs.actions[:, t], seqs.rewards[:, t]
        cleared = seqs.session_start[:, t, None]
        k, dk = np.where(cleared, 0.0, k), np.where(cleared, 0.0, dk)
        logits = beta * q + beta_k * k + plus_bias * seqs.plus
        log_p, p = _softmax(logits)
        per_trial[:, t] = np.where(live, log_p[rows, action], np.nan)
        if return_gradient:
            per_step = _step_gradient(beta, beta_k, q, k, dq, dk, seqs.plus, p, action)
            gradient += per_step[live].sum(axis=0)
        chosen = arms[None, :] == action[:, None]
        delta = reward - q[rows, action]
        better = delta > 0
        rate = np.where(better, alpha_pos, alpha_neg)
        q_new = np.where(chosen, q + (rate * delta)[:, None], zeta * q)
        k_new = k + alpha_k * (chosen - k)
        if return_gradient:
            dq_new = np.where(chosen[:, :, None], (1 - rate)[:, None, None] * dq, zeta * dq)
            dq_new[:, :, 0] += chosen * (better * delta)[:, None]  # the chosen value's own rate
            dq_new[:, :, 1] += chosen * (~better * delta)[:, None]
            dq_new[:, :, 2] += ~chosen * q  # the unchosen values' retention
            dk_new = (1 - alpha_k) * dk + (chosen - k)
            dq, dk = np.where(live[:, None, None], dq_new, dq), np.where(live[:, None], dk_new, dk)
        q, k = np.where(live[:, None], q_new, q), np.where(live[:, None], k_new, k)
    return per_trial, (gradient if return_gradient else None)


def log_likelihood(params, seqs, return_gradient: bool = False):
    """Summed log probability of the chosen ports, and optionally its gradient.

    Args:
        params (Sequence[float]): ``PARAMETER_NAMES`` order.
        seqs (value_agent.Sequences): The mouse's problems.
        return_gradient (bool): Also return the gradient. Default False.

    Returns:
        float | tuple[float, np.ndarray]: The log likelihood, with the ``(7,)`` gradient in
        ``PARAMETER_NAMES`` order when asked.

    Raises:
        ValueError: For other than seven finite parameters, or a rate or ``zeta`` outside [0, 1].
    """
    per_trial, gradient = _forward(params, seqs, return_gradient)
    total = float(np.nansum(per_trial))
    return (total, gradient) if return_gradient else total


def trial_log_probabilities(params, seqs) -> np.ndarray:
    """Log probability of the chosen port on every trial, to score any subset of trials.

    The agent learns from every trial of a problem, so the trials a score leaves out still shape
    the values: select them in the returned array rather than removing them from *seqs*.

    Args:
        params (Sequence[float]): ``PARAMETER_NAMES`` order.
        seqs (value_agent.Sequences): The problems to score.

    Returns:
        np.ndarray: ``(n_problems, n_steps)``, NaN at padded steps.
    """
    return _forward(params, seqs, False)[0]


def variant_log_likelihood(variant: Variant, free: Sequence[float], seqs, return_gradient=False):
    """:func:`log_likelihood` as a function of the free parameters of *variant*.

    The gradient is with respect to the free parameters: a tied parameter's derivative is added
    to the one it copies.
    """
    params = expand(variant, free)
    if not return_gradient:
        return log_likelihood(params, seqs)
    total, full = log_likelihood(params, seqs, return_gradient=True)
    gradient = np.array(
        [
            full[INDEX[name]]
            + sum(full[INDEX[t]] for t, source in variant.tied.items() if source == name)
            for name in variant.free
        ]
    )
    return total, gradient


class Fit(NamedTuple):
    """Maximum-likelihood parameters of one mouse.

    Attributes:
        parameters (np.ndarray): ``(7,)`` in ``PARAMETER_NAMES`` order (fixed and tied values
            filled in for an ablation).
        log_likelihood (float): Summed log likelihood at the fit.
        n_trials (int): Trials fitted.
    """

    parameters: np.ndarray
    log_likelihood: float
    n_trials: int


def fit(seqs, n_starts: int = 5, seed: int = 0, variant: Variant = VARIANTS["full"]) -> Fit:
    """Fit *variant* by multi-start L-BFGS-B within ``BOUNDS``.

    Args:
        seqs (value_agent.Sequences): The mouse's problems.
        n_starts (int): Random starting points. Default 5.
        seed (int): Seed of the starting points. Default 0.
        variant (Variant): The full model or an ablation. Default the full model.

    Returns:
        Fit: The best start.

    Raises:
        ValueError: If *n_starts* < 1.
        RuntimeError: If no start converged.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    rng = np.random.default_rng(seed)
    n_trials = int(seqs.mask.sum())
    bounds = [BOUNDS_BY_NAME[name] for name in variant.free]

    def loss(free):
        value, gradient = variant_log_likelihood(variant, free, seqs, return_gradient=True)
        return -value / n_trials, -gradient / n_trials

    results = []
    for _ in range(n_starts):
        start = [rng.uniform(*START_RANGES[name]) for name in variant.free]
        result = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=bounds)
        if result.success:
            results.append(result)
    if not results:
        raise RuntimeError(f"no start of the {variant.name} fit converged")
    best = min(results, key=lambda r: r.fun)
    return Fit(np.array(expand(variant, best.x)), float(-best.fun * n_trials), n_trials)


def parameters_at_bounds(parameters, variant: Variant = VARIANTS["full"]) -> tuple[str, ...]:
    """Names of the free parameters of *variant* within ``AT_BOUND_ATOL`` of a bound.

    A parameter at a bound means the optimum lies outside the allowed range, so its value (and
    the likelihood comparison) should be read with care. Held and tied parameters are not
    reported: they are not fitted.
    """
    return tuple(
        name
        for name in variant.free
        if any(
            abs(parameters[INDEX[name]] - bound) <= AT_BOUND_ATOL for bound in BOUNDS_BY_NAME[name]
        )
    )
