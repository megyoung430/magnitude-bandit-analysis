"""A GLM-HMM for three-alternative choices: hidden strategy states with their own weights.

The model follows Ashwood, Roy, Stone et al. (2022, Nat Neurosci, "Mice alternate between
discrete strategies during perceptual decision-making") and the mixture-of-agents HMM of
Venditto, Miller, Brody & Daw (2024, eLife). A session is an independent sequence of trials. In
each trial ``t`` the mouse is in a hidden state ``z_t`` and chooses arm ``j`` with probability

    p(y_t = j | z_t = k) = softmax_j( sum_f w[k, f] * x[t, j, f] )

(a conditional logit with state-specific weights ``w[k]`` on per-arm features ``x``: the
"agent signals" of :mod:`src.behavior_analysis.glm_hmm.glm_hmm_features`). The states follow a
Markov
chain with transition matrix ``A`` and start each session from the initial distribution ``pi``,
so a state never carries over a session boundary.

**Fitting** is expectation-maximization on the penalized log likelihood (the log posterior):

- the E-step is forward-backward in the scaled probability domain, batched over sessions;
- the M-step sets ``pi`` and ``A`` in closed form (a Dirichlet MAP, see :class:`Prior`) and takes a
  few L-BFGS steps on each state's weighted ridge conditional logit from its current weights. A
  partial M-step still raises the log posterior (generalized EM), which every iteration checks.

The likelihood has local optima, so ``fit_restarts`` starts from several initializations (a split
of the one-state fit, a k-means of the sessions' own fitted weights, and random perturbations) and
keeps the best converged one. States are sorted by their weight on one feature, so the labels
mean the same thing from fit to fit.

**Held-out score.** ``predictive_log_probs`` is the log probability of each choice given the
*earlier* choices of the same held-out session (the forward filter), so a held-out session's own
future never informs the state used to predict a choice. With one state it is the plain
conditional logit.

Everything here is pure: arrays in, arrays out.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize

N_ARMS = 3
# Scale of the random starts of the state weights (logits per unit of a feature).
INIT_SCALE = 1.0
# Ridge penalty of the per-session fits behind the k-means start: a session has about 60 trials,
# so its weights need more shrinkage than the whole mouse's.
SESSION_L2 = 5.0
# L-BFGS iterations of one M-step (a generalized EM step) and of the convex one-state fit.
M_STEP_ITERATIONS = 25
SINGLE_STATE_ITERATIONS = 500
# A line-search stop with a gradient (max norm) above this is a failure, not a converged fit.
STUCK_GRADIENT = 1e-3
# Initial transition matrix of every start: stay with this probability.
INITIAL_STAY = 0.95
# EM stops when the log posterior improves by less than this many nats in total.
DEFAULT_TOLERANCE = 1e-2
DEFAULT_MAX_ITERATIONS = 200
# A step that lowers the log posterior by more than this (relative) is a bug, not noise.
MONOTONE_RTOL = 1e-9
MONOTONE_ATOL = 1e-6


class Sequences(NamedTuple):
    """Sessions as padded arrays.

    Attributes:
        X (np.ndarray): ``(sessions, steps, 3, features)`` per-arm features.
        chosen (np.ndarray): ``(sessions, steps)`` index of the chosen arm.
        mask (np.ndarray): ``(sessions, steps)`` bool; live trials. Padding is trailing.
        scored (np.ndarray): ``(sessions, steps)`` bool; the live trials whose log probability a
            held-out score counts (a subset of ``mask``).
    """

    X: np.ndarray
    chosen: np.ndarray
    mask: np.ndarray
    scored: np.ndarray


@dataclass(frozen=True)
class Prior:
    """The penalties of the log posterior.

    Attributes:
        l2 (float): Ridge penalty on every state weight, added to the summed negative log
            likelihood (a Gaussian prior with sd ``1/sqrt(l2)``; as the choice logit's).
        transition_alpha (float): Dirichlet concentration of every transition probability.
        transition_kappa (float): Extra concentration on each diagonal entry (the "sticky"
            prior of Ashwood et al.): the MAP transitions are ``n_ij + alpha - 1 + kappa * [i = j]``
            normalized over ``j``. Default 20 adds the weight of 20 self-transitions, a
            few seconds of the 1 to 4 s trial rate: weak against thousands of trials.
        initial_alpha (float): Dirichlet concentration of the initial-state distribution.
    """

    l2: float = 1.0
    transition_alpha: float = 2.0
    transition_kappa: float = 20.0
    initial_alpha: float = 2.0

    def __post_init__(self):
        if self.l2 < 0:
            raise ValueError(f"l2 must be non-negative, got {self.l2}")
        if self.transition_alpha < 1 or self.initial_alpha < 1:
            raise ValueError("Dirichlet concentrations must be at least 1 (the MAP needs it)")
        if self.transition_kappa < 0:
            raise ValueError(f"transition_kappa must be non-negative, got {self.transition_kappa}")


@dataclass(frozen=True)
class Params:
    """The parameters of a GLM-HMM.

    Attributes:
        weights (np.ndarray): ``(states, features)``.
        initial (np.ndarray): ``(states,)`` distribution of a session's first state.
        transition (np.ndarray): ``(states, states)``; row ``i`` is the distribution of the next
            state given state ``i``.
    """

    weights: np.ndarray
    initial: np.ndarray
    transition: np.ndarray

    def __post_init__(self):
        k = len(self.initial)
        if self.weights.ndim != 2 or self.weights.shape[0] != k:
            raise ValueError(f"weights must be ({k}, features), got {self.weights.shape}")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        if self.transition.shape != (k, k):
            raise ValueError(f"transition must be ({k}, {k}), got {self.transition.shape}")
        for name, probs in (("initial", self.initial), ("transition", self.transition)):
            sums = probs.sum(axis=-1)
            if np.any(probs < 0) or not np.allclose(sums, 1.0, atol=1e-6):
                raise ValueError(f"{name} must hold non-negative rows that sum to 1")

    @property
    def n_states(self) -> int:
        return len(self.initial)


class ForwardBackward(NamedTuple):
    """The E-step.

    Attributes:
        gamma (np.ndarray): ``(sessions, steps, states)`` posterior state probabilities (0 at
            padding).
        xi (np.ndarray): ``(states, states)`` expected transition counts summed over sessions.
        initial_gamma (np.ndarray): ``(states,)`` summed posterior of every session's first state.
        log_likelihood (np.ndarray): ``(sessions,)`` marginal log likelihood of each session.
        log_predictive (np.ndarray): ``(sessions, steps)`` log p(choice_t | choices before t)
            (0 at padding).
    """

    gamma: np.ndarray
    xi: np.ndarray
    initial_gamma: np.ndarray
    log_likelihood: np.ndarray
    log_predictive: np.ndarray


class EMFit(NamedTuple):
    """One EM run.

    Attributes:
        params (Params): Parameters at the last E-step.
        objective (float): Log posterior at *params*.
        log_likelihood (float): Summed log likelihood at *params* (without the penalties).
        objective_trace (np.ndarray): Log posterior at every E-step; never decreases.
        n_iterations (int): M-steps taken.
        converged (bool): The last improvement was below the tolerance.
    """

    params: Params
    objective: float
    log_likelihood: float
    objective_trace: np.ndarray
    n_iterations: int
    converged: bool


class HmmFit(NamedTuple):
    """The best of several EM runs.

    Attributes:
        params (Params): Parameters of the best converged run, states sorted.
        objective (float): Its log posterior.
        log_likelihood (float): Its summed log likelihood.
        converged (bool): Always True (a fit with no converged run raises).
        n_restarts (int): Starts run.
        n_converged (int): Starts that converged.
        objectives (np.ndarray): Final log posterior of every start (``-inf`` if not converged).
    """

    params: Params
    objective: float
    log_likelihood: float
    converged: bool
    n_restarts: int
    n_converged: int
    objectives: np.ndarray


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_sequences(seqs: Sequences) -> None:
    """Raise ``ValueError`` unless *seqs* hold well-formed sessions.

    Raises:
        ValueError: For wrong shapes, non-finite features on live trials, a chosen arm outside
            0-2, padding that is not trailing, an empty session, or ``scored`` outside ``mask``.
    """
    X, chosen, mask, scored = seqs
    if X.ndim != 4 or X.shape[2] != N_ARMS:
        raise ValueError(f"X must be (sessions, steps, 3, features), got {X.shape}")
    if not (chosen.shape == mask.shape == scored.shape == X.shape[:2]):
        raise ValueError("chosen, mask and scored must be (sessions, steps)")
    if mask.dtype != bool or scored.dtype != bool:
        raise ValueError("mask and scored must be boolean")
    if not np.all(np.isfinite(X[mask])):
        raise ValueError("features must be finite on live trials")
    if chosen[mask].min(initial=0) < 0 or chosen[mask].max(initial=0) >= N_ARMS:
        raise ValueError("chosen must be an arm index 0-2 on live trials")
    if np.any(np.diff(mask.astype(int), axis=1) > 0):
        raise ValueError("padding must be trailing: a live trial follows a padded one")
    if not mask[:, 0].all():
        raise ValueError("every session needs at least one live trial")
    if np.any(scored & ~mask):
        raise ValueError("scored trials must be live trials")


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------


def emission_log_probs(weights: np.ndarray, X: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """``(sessions, steps, states)`` log probability of the chosen arm under each state.

    Args:
        weights (np.ndarray): ``(states, features)``.
        X (np.ndarray): ``(sessions, steps, 3, features)``.
        chosen (np.ndarray): ``(sessions, steps)``.
    """
    logits = np.einsum("stjf,kf->stkj", X, weights)
    logits = logits - logits.max(axis=-1, keepdims=True)
    log_p = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    index = np.broadcast_to(chosen[:, :, None, None], (*chosen.shape, log_p.shape[2], 1))
    return np.take_along_axis(log_p, index, axis=-1)[..., 0]


def forward_backward(log_emit: np.ndarray, params: Params, mask: np.ndarray) -> ForwardBackward:
    """Scaled forward-backward over independent sessions.

    A session starts from ``params.initial``; padded steps leave the filter untouched.

    Args:
        log_emit (np.ndarray): ``(sessions, steps, states)`` from :func:`emission_log_probs`.
        params (Params): The model.
        mask (np.ndarray): ``(sessions, steps)`` live trials, padding trailing.

    Returns:
        ForwardBackward: Posteriors, expected transitions, and the likelihoods.
    """
    n_sessions, n_steps, k = log_emit.shape
    A = params.transition
    emit = np.where(mask[:, :, None], np.exp(log_emit), 1.0)
    alpha, scale = np.empty((n_sessions, n_steps, k)), np.ones((n_sessions, n_steps))
    first = params.initial[None, :] * emit[:, 0]
    scale[:, 0] = first.sum(axis=1)
    alpha[:, 0] = first / scale[:, 0, None]
    for t in range(1, n_steps):
        live = mask[:, t, None]
        step = np.where(live, alpha[:, t - 1] @ A, alpha[:, t - 1]) * emit[:, t]
        scale[:, t] = step.sum(axis=1)
        alpha[:, t] = step / scale[:, t, None]
    beta = np.ones((n_sessions, n_steps, k))
    for t in range(n_steps - 2, -1, -1):
        nxt = emit[:, t + 1] * beta[:, t + 1] / scale[:, t + 1, None]
        beta[:, t] = np.where(mask[:, t + 1, None], nxt @ A.T, 1.0)
    gamma = np.where(mask[:, :, None], alpha * beta, 0.0)
    xi = np.zeros((k, k))
    for t in range(n_steps - 1):
        live = mask[:, t + 1, None]
        weight = np.where(live, emit[:, t + 1] * beta[:, t + 1] / scale[:, t + 1, None], 0.0)
        xi += (alpha[:, t].T @ weight) * A
    log_scale = np.log(scale)
    return ForwardBackward(gamma, xi, gamma[:, 0].sum(axis=0), log_scale.sum(axis=1), log_scale)


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------


def _weighted_logit(w, X, chosen, resp, l2):
    """Penalized weighted negative log likelihood of a conditional logit, and its gradient."""
    u = X @ w
    u = u - u.max(axis=1, keepdims=True)
    log_p = u - np.log(np.exp(u).sum(axis=1, keepdims=True))
    p = np.exp(log_p)
    rows = np.arange(len(chosen))
    value = -(resp * log_p[rows, chosen]).sum() + 0.5 * l2 * w @ w
    # d/dw of -log p(chosen) is -(x_chosen - sum_j p_j x_j)
    expected = np.einsum("nj,njf->nf", p, X)
    gradient = -(resp[:, None] * (X[rows, chosen] - expected)).sum(axis=0) + l2 * w
    return value, gradient


def fit_state_weights(
    X: np.ndarray,
    chosen: np.ndarray,
    resp: np.ndarray,
    start: np.ndarray,
    l2: float,
    max_iterations: int,
    tight: bool = False,
) -> np.ndarray:
    """Weights of one weighted ridge conditional logit by L-BFGS from *start*.

    Args:
        X (np.ndarray): ``(trials, 3, features)``.
        chosen (np.ndarray): ``(trials,)``.
        resp (np.ndarray): ``(trials,)`` non-negative weight of each trial.
        start (np.ndarray): ``(features,)`` initial weights; the result is never worse.
        l2 (float): Ridge penalty.
        max_iterations (int): L-BFGS iteration limit (hitting it is a partial step, not an error).
        tight (bool): Tighter tolerances, for a fit run to convergence. Default False.

    Returns:
        np.ndarray: ``(features,)`` weights.

    Raises:
        RuntimeError: If the optimizer stops abnormally.
    """
    options = {"maxiter": max_iterations}
    if tight:
        options |= {"ftol": 1e-13, "gtol": 1e-8}
    result = minimize(
        _weighted_logit, start, args=(X, chosen, resp, l2), jac=True, method="L-BFGS-B",
        options=options,
    )  # fmt: skip
    # L-BFGS reports ABNORMAL when its line search cannot improve on a point that is already
    # at the optimum to machine precision; that is a convergence, a large gradient is not
    stuck = "ABNORMAL" in str(result.message) and np.abs(result.jac).max() > STUCK_GRADIENT
    if not np.all(np.isfinite(result.x)) or stuck:
        raise RuntimeError(f"the state-weight fit stopped abnormally: {result.message}")
    return result.x


def transition_from_counts(counts: np.ndarray, prior: Prior) -> np.ndarray:
    """MAP transition matrix from expected counts under the sticky Dirichlet prior.

    Raises:
        ValueError: If a row has no counts and no prior mass.
    """
    k = len(counts)
    mass = counts + (prior.transition_alpha - 1.0) + prior.transition_kappa * np.eye(k)
    rows = mass.sum(axis=1, keepdims=True)
    if np.any(rows <= 0):
        raise ValueError("a state has no transitions and the prior adds none")
    return mass / rows


def _initial_from_counts(counts: np.ndarray, prior: Prior) -> np.ndarray:
    mass = counts + (prior.initial_alpha - 1.0)
    if mass.sum() <= 0:
        raise ValueError("no sessions and the prior adds no mass")
    return mass / mass.sum()


def _log_prior(params: Params, prior: Prior) -> float:
    k = params.n_states
    concentration = (prior.transition_alpha - 1.0) + prior.transition_kappa * np.eye(k)
    with np.errstate(divide="ignore"):
        log_a, log_pi = np.log(params.transition), np.log(params.initial)
    transition = np.where(concentration > 0, concentration * log_a, 0.0).sum()
    initial = np.where(prior.initial_alpha > 1, (prior.initial_alpha - 1.0) * log_pi, 0.0).sum()
    return float(transition + initial - 0.5 * prior.l2 * (params.weights**2).sum())


def _flat_live(seqs: Sequences):
    return seqs.X[seqs.mask], seqs.chosen[seqs.mask]


def _m_step(seqs, flat, fb, params, prior, iterations) -> Params:
    X, chosen = flat
    resp = fb.gamma[seqs.mask]
    weights = np.stack(
        [
            fit_state_weights(X, chosen, resp[:, k], params.weights[k], prior.l2, iterations)
            for k in range(params.n_states)
        ]
    )
    return Params(
        weights,
        _initial_from_counts(fb.initial_gamma, prior),
        transition_from_counts(fb.xi, prior),
    )


def _e_step(seqs: Sequences, params: Params) -> ForwardBackward:
    return forward_backward(
        emission_log_probs(params.weights, seqs.X, seqs.chosen), params, seqs.mask
    )


def _single_state(seqs: Sequences, init: Params, prior: Prior) -> EMFit:
    """The one-state model is a convex ridge conditional logit: one tight fit."""
    X, chosen = _flat_live(seqs)
    w = fit_state_weights(
        X, chosen, np.ones(len(chosen)), init.weights[0], prior.l2, SINGLE_STATE_ITERATIONS, True
    )
    params = Params(w[None, :], np.ones(1), np.ones((1, 1)))
    fb = _e_step(seqs, params)
    log_likelihood = float(fb.log_likelihood.sum())
    objective = log_likelihood + _log_prior(params, prior)
    return EMFit(params, objective, log_likelihood, np.array([objective]), 1, True)


def fit_em(
    seqs: Sequences,
    init: Params,
    prior: Prior,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> EMFit:
    """EM from *init* until the log posterior improves by less than *tolerance* nats.

    Args:
        seqs (Sequences): The sessions.
        init (Params): Starting parameters.
        prior (Prior): The penalties.
        max_iterations (int): M-steps allowed. Default ``DEFAULT_MAX_ITERATIONS``.
        tolerance (float): Convergence threshold on the log posterior (nats). Default
            ``DEFAULT_TOLERANCE``; 0 never converges unless a step changes nothing.

    Returns:
        EMFit: The fit and its objective trace.

    Raises:
        ValueError: For malformed *seqs*.
        RuntimeError: If an iteration lowers the log posterior (a bug in a step).
    """
    check_sequences(seqs)
    if init.n_states == 1:
        return _single_state(seqs, init, prior)
    flat = _flat_live(seqs)
    params, trace, converged = init, [], False
    for iteration in range(max_iterations + 1):
        fb = _e_step(seqs, params)
        log_likelihood = float(fb.log_likelihood.sum())
        trace.append(log_likelihood + _log_prior(params, prior))
        if iteration:
            drop = trace[-2] - trace[-1]
            if drop > MONOTONE_ATOL + MONOTONE_RTOL * abs(trace[-1]):
                raise RuntimeError(f"EM lowered the log posterior by {drop:.3g} nats")
            if trace[-1] - trace[-2] <= tolerance:
                converged = True
                break
        if iteration == max_iterations:
            break
        params = _m_step(seqs, flat, fb, params, prior, M_STEP_ITERATIONS)
    return EMFit(params, trace[-1], log_likelihood, np.array(trace), len(trace) - 1, converged)


# ---------------------------------------------------------------------------
# Initializations and restarts
# ---------------------------------------------------------------------------


def session_weights(seqs: Sequences, l2: float = SESSION_L2) -> np.ndarray:
    """``(sessions, features)`` ridge conditional-logit weights fitted to each session alone."""
    n_features = seqs.X.shape[-1]
    start = np.zeros(n_features)
    out = []
    for s in range(len(seqs.chosen)):
        live = seqs.mask[s]
        out.append(
            fit_state_weights(
                seqs.X[s][live], seqs.chosen[s][live], np.ones(live.sum()), start, l2,
                M_STEP_ITERATIONS, True,
            )
        )  # fmt: skip
    return np.array(out)


def _sticky_start(weights: np.ndarray) -> Params:
    k = len(weights)
    transition = np.full((k, k), (1 - INITIAL_STAY) / max(k - 1, 1))
    np.fill_diagonal(transition, INITIAL_STAY)
    return Params(weights, np.full(k, 1 / k), transition)


def initial_params(
    n_states: int, one_state: np.ndarray, per_session: np.ndarray | None, index: int,
    rng: np.random.Generator,
) -> Params:  # fmt: skip
    """Starting parameters of restart *index*.

    Restart 0 splits the one-state weights along a random direction; restart 1 (with at least as
    many sessions as states) clusters the sessions' own weights with k-means; the others perturb
    the one-state weights at random.

    Args:
        n_states (int): States.
        one_state (np.ndarray): ``(features,)`` the one-state fit.
        per_session (np.ndarray | None): ``(sessions, features)`` from :func:`session_weights`.
        index (int): Restart number.
        rng (np.random.Generator): Source of randomness.
    """
    n_features = len(one_state)
    if index == 0:
        direction = rng.normal(size=n_features)
        direction *= INIT_SCALE / np.linalg.norm(direction)
        offsets = (np.arange(n_states) - (n_states - 1) / 2)[:, None] * direction
        return _sticky_start(one_state + offsets)
    if index == 1 and per_session is not None and len(per_session) >= n_states:
        centers, _ = kmeans2(per_session, n_states, minit="++", seed=rng)
        if np.all(np.isfinite(centers)) and len(np.unique(centers.round(6), axis=0)) == n_states:
            return _sticky_start(centers)
    return _sticky_start(one_state + rng.normal(scale=INIT_SCALE, size=(n_states, n_features)))


def sort_states(params: Params, feature: int) -> Params:
    """*params* with the states in descending order of their weight on *feature*."""
    order = np.argsort(-params.weights[:, feature], kind="stable")
    return Params(
        params.weights[order], params.initial[order], params.transition[np.ix_(order, order)]
    )


def fit_restarts(
    seqs: Sequences,
    n_states: int,
    prior: Prior,
    n_restarts: int,
    seed: int,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    sort_feature: int = 0,
    screen_iterations: int | None = None,
    n_keep: int = 3,
) -> HmmFit:
    """The best converged EM run of several starts.

    One state is convex, so it runs once. With *screen_iterations*, every start runs that many
    iterations first and only the *n_keep* best continue to convergence, which saves most of the
    time of starts that were never going to win.

    Args:
        seqs (Sequences): The sessions.
        n_states (int): Number of states.
        prior (Prior): The penalties.
        n_restarts (int): Starts (at least 1).
        seed (int): Seed of the starts.
        max_iterations (int): EM iterations allowed per start. Default ``DEFAULT_MAX_ITERATIONS``.
        tolerance (float): Convergence threshold (nats). Default ``DEFAULT_TOLERANCE``.
        sort_feature (int): Feature whose weight orders the states, descending. Default 0.
        screen_iterations (int | None): Iterations of the screening stage. Default ``None``: none.
        n_keep (int): Starts that continue after screening. Default 3.

    Returns:
        HmmFit: The best converged start, states sorted.

    Raises:
        ValueError: For ``n_states < 1``, ``n_restarts < 1`` or malformed *seqs*.
        RuntimeError: If no start converges.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be at least 1, got {n_states}")
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be at least 1, got {n_restarts}")
    check_sequences(seqs)
    rng = np.random.default_rng(seed)
    n_features = seqs.X.shape[-1]
    X, chosen = _flat_live(seqs)
    one = fit_state_weights(
        X,
        chosen,
        np.ones(len(chosen)),
        np.zeros(n_features),
        prior.l2,
        SINGLE_STATE_ITERATIONS,
        True,
    )
    if n_states == 1:
        run = _single_state(seqs, Params(one[None, :], np.ones(1), np.ones((1, 1))), prior)
        runs = [run]
    else:
        per_session = session_weights(seqs) if n_restarts > 1 else None
        starts = [initial_params(n_states, one, per_session, i, rng) for i in range(n_restarts)]
        stage = screen_iterations or max_iterations
        runs = [fit_em(seqs, s, prior, stage, tolerance) for s in starts]
        if screen_iterations:
            best = sorted(range(len(runs)), key=lambda i: -runs[i].objective)[:n_keep]
            for i in best:
                runs[i] = fit_em(seqs, runs[i].params, prior, max_iterations, tolerance)
            runs = [r if i in best else r._replace(converged=False) for i, r in enumerate(runs)]
    converged = [r for r in runs if r.converged]
    if not converged:
        raise RuntimeError(
            f"none of {len(runs)} EM starts converged in {max_iterations} iterations "
            f"(tolerance {tolerance} nats, {n_states} states)"
        )
    best = max(converged, key=lambda r: r.objective)
    return HmmFit(
        sort_states(best.params, sort_feature),
        best.objective,
        best.log_likelihood,
        True,
        len(runs),
        len(converged),
        np.array([r.objective if r.converged else -np.inf for r in runs]),
    )


# ---------------------------------------------------------------------------
# Using a fit
# ---------------------------------------------------------------------------


def posteriors(params: Params, seqs: Sequences) -> np.ndarray:
    """``(sessions, steps, states)`` posterior state probabilities given the whole session."""
    check_sequences(seqs)
    return _e_step(seqs, params).gamma


def predictive_log_probs(params: Params, seqs: Sequences) -> np.ndarray:
    """``(sessions, steps)`` log p(choice_t | the session's earlier choices), 0 at padding."""
    check_sequences(seqs)
    return _e_step(seqs, params).log_predictive


def heldout_log_likelihood(params: Params, seqs: Sequences) -> float:
    """Summed predictive log probability of the scored trials of *seqs*.

    Raises:
        ValueError: For malformed *seqs* (``scored`` must lie within ``mask``).
    """
    check_sequences(seqs)
    return float(_e_step(seqs, params).log_predictive[seqs.scored].sum())


def dwell_times(transition: np.ndarray) -> np.ndarray:
    """Mean dwell time of each state in trials, ``1 / (1 - A_ii)``."""
    return 1.0 / (1.0 - np.diag(transition))


def occupancy(gamma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(states,)`` mean posterior probability of each state over the live trials."""
    return gamma[mask].mean(axis=0)
