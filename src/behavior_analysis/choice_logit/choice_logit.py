"""A conditional logit of the chosen port on reward history, choice history and tower bias.

On every trial the mouse chooses among the three offered ports. Port ``a`` gets the utility

    U_a(t) = bias[tower a] + sum_k wr_k * r_a(t-k) + wt * tail_a(t) + sum_k wc_k * c_a(t-k)

and is chosen with probability ``softmax_a(U)``, where

- ``r_a(t-k)`` is the reward magnitude (pulses) received at port ``a`` on trial ``t-k`` if the
  mouse chose it then, else 0 (``reward_lag<k>`` features);
- optionally one more reward regressor per port, the *tail* (``reward_tail``): the exponentially
  weighted reward received there more than ``K`` trials ago, ``sum_{j>K} g**(j-K-1) r_a(t-j)``
  with decay ``g``. Without it a port not chosen in the last ``K`` trials has no value at all,
  and the last lag absorbs all older history;
- ``c_a(t-k)`` is 1 if port ``a`` was chosen on trial ``t-k``, else 0 (``choice_lag<k>``);
- ``bias`` is one weight per tower, shared by all problems of the mouse (``tower_<name>``). The
  alphabetically first tower is the reference with bias 0: adding the same constant to every
  tower's bias leaves the softmax unchanged, so one tower is pinned.

Optional columns
----------------
``design_matrix(..., extras=...)`` appends columns that the standard model does not have, so the
standard columns and their order never change:

- ``BINARY``: reward lags (and tail) that code *whether* the arm paid (1) rather than how many
  pulses, ``binary_lag<k>`` and ``binary_tail``. Next to the pulse columns they test whether the
  mouse's choices are sensitive to magnitude beyond rewarded versus not;
- ``BELIEVED``: the believed value of each arm, the magnitude it last paid in the current problem
  (``believed_value``, 0 before the arm's first visit), and ``believed_unvisited``, 1 before the
  first visit. The indicator lets an unvisited arm differ from an arm that paid 0 instead of
  being treated as worth 0 or dropped. Believed values carry across sessions and dropped trials
  like the reward history and start empty at a problem's start;
- ``CHOICE_TAIL``: one more choice regressor per port, ``choice_tail``: the exponentially weighted
  choice indicator older than the ``n_lags`` choice lags,
  ``sum_{j>K} g**(j-K-1) c_a(t-j)`` with decay ``g`` (``choice_tail_decay``), like the reward
  tail and cleared at a session start like the choice lags. It lets a slow build-up of
  perseveration (a habit) be one weight instead of many collinear lags that the L2 penalty
  shares between. The decay is a parameter; a decay of 0.9 (half-life about 7 trials, the
  default of the reward tail) puts its weight on the 6-30 trials before a choice, beyond the five
  standard lags.

History
-------
Reward history carries over between the sessions of a problem and starts empty at the problem's
start, like the values of the value learner in :mod:`src.agents.value_agent`. Choice history is
reset at a session start and after a dropped trial, as that learner's stickiness is: the mouse
begins a session at the initiation port, so the last choice of the previous session is not a
choice to repeat. A lag that reaches back before the start of the history is zero-padded.

The model value
---------------
``V_a(t) = sum_k wr_k * r_a(t-k) + wt * tail_a(t)`` (:func:`model_values`) is the reward part
alone. The choice
and tower terms are nuisance: they explain perseveration and port preference, and leaving them
out of the value keeps them out of the value itself. They do *not* keep them out of a
psychometric drawn against it: a value built from recent rewards is non-zero mainly for recently
chosen arms, and a perseverative mouse chooses those arms. Compare every such psychometric with
the prediction of the nuisance-only model
(:mod:`src.behavior_analysis.choice_logit.choice_logit_cv`).

Everything here is pure: a trial table in, arrays out.
"""

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import logsumexp

N_ARMS = 3
REWARD, CHOICE, TOWER = "reward", "choice", "tower"
# Groups of the optional columns (see the module docstring).
BINARY, BELIEVED, CHOICE_TAIL = "binary", "believed", "choice_tail"
EXTRAS = (BINARY, BELIEVED)
# ``CHOICE_TAIL`` is an extra too, but it needs a decay, so ``EXTRAS`` (the groups that take no
# argument) leaves it out.
ALL_EXTRAS = (*EXTRAS, CHOICE_TAIL)
# Feature-name prefix of each feature group.
GROUP_PREFIX = {
    REWARD: "reward_",
    CHOICE: "choice_lag",
    TOWER: "tower_",
    BINARY: "binary_",
    BELIEVED: "believed_",
    CHOICE_TAIL: "choice_tail",
}
TAIL_FEATURE = "reward_tail"
BINARY_TAIL_FEATURE = "binary_tail"
BELIEVED_FEATURES = ("believed_value", "believed_unvisited")
DEFAULT_LAGS = 5
# Decay per trial of the reward tail; the pipeline chooses it from a grid by held-out likelihood.
DEFAULT_TAIL_DECAY = 0.9
# L2 penalty on every weight, added to the summed (not mean) negative log likelihood. It is a
# Gaussian prior with SD 1/sqrt(l2) = 1 logit per pulse (or per unit of the 0/1 features): weak
# next to thousands of trials, but it keeps the fit finite when a feature hardly varies and
# holds the bias of a tower the training problems never offered at exactly 0.
DEFAULT_L2 = 1.0
MAX_ITERATIONS = 500


class LogitDesign(NamedTuple):
    """Arm-level features of one mouse's trials.

    Attributes:
        X (np.ndarray): ``(n_trials, 3, n_features)``; ports are in sorted order within a trial.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen port along axis 1.
        feature_names (tuple[str, ...]): Names of the last axis of ``X``.
        meta (pd.DataFrame): One row per trial, in the order of ``X``: the columns of the trial
            table that later steps need (``mouse``, ``problem``, ``session``, ``trial``,
            ``trial_in_session``, ``after_gap``, ``reward``, ``magnitudes``).
        ports (np.ndarray | None): ``(n_trials, 3)`` port names behind axis 1 of ``X``.
    """

    X: np.ndarray
    chosen: np.ndarray
    feature_names: tuple
    meta: pd.DataFrame
    ports: np.ndarray | None


class LogitFit(NamedTuple):
    """Maximum-penalised-likelihood weights of one fit.

    Attributes:
        beta (np.ndarray): ``(n_features,)`` weights in ``feature_names`` order.
        feature_names (tuple[str, ...]): Name of each weight.
        nll (float): Summed negative log likelihood of the fitted trials (without the penalty).
        n_trials (int): Trials fitted.
        l2 (float): The penalty used.
    """

    beta: np.ndarray
    feature_names: tuple
    nll: float
    n_trials: int
    l2: float


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

META_COLUMNS = (
    "mouse",
    "problem",
    "session",
    "trial",
    "trial_in_session",
    "after_gap",
    "reward",
    "magnitudes",
)


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    """*values* shifted down by *lag* rows, zero-padded at the top."""
    out = np.zeros_like(values)
    out[lag:] = values[:-lag]
    return out


def _problem_ports(rows: pd.DataFrame) -> list[str]:
    """The three ports offered throughout one problem, sorted.

    Raises:
        ValueError: If a trial does not offer three ports, or the offered ports change.
    """
    port_sets = {tuple(sorted(m)) for m in rows["magnitudes"]}
    problem = rows["problem"].iloc[0]
    if len(port_sets) != 1:
        raise ValueError(f"problem {problem!r}: offered ports change within the problem")
    ports = list(port_sets.pop())
    if len(ports) != N_ARMS:
        raise ValueError(f"problem {problem!r}: expected three offered ports, got {ports}")
    return ports


def _tail(older: np.ndarray, decay: float) -> np.ndarray:
    """``tail(t) = decay * tail(t-1) + older(t)``: the recursion of the decayed sum."""
    return lfilter([1.0], [1.0, -decay], older, axis=0)


def _segment_tail(older: np.ndarray, segment: np.ndarray, decay: float) -> np.ndarray:
    """:func:`_tail` restarted at every segment, so nothing carries over a segment start."""
    out = np.zeros_like(older)
    for label in np.unique(segment):
        rows = segment == label
        out[rows] = _tail(older[rows], decay)
    return out


def _believed_features(received: np.ndarray, picked: np.ndarray) -> list[np.ndarray]:
    """``[value, unvisited]``, each ``(n, 3)``: what each arm last paid before each trial.

    *received* holds a trial's reward in the chosen arm's column and 0 elsewhere, *picked* the
    chosen arm's one-hot choice. Before an arm's first visit the value is 0 and *unvisited* 1.
    """
    trial = np.arange(len(received))[:, None]
    last_visit = np.maximum.accumulate(np.where(picked == 1, trial, -1), axis=0)
    before = np.vstack([np.full((1, N_ARMS), -1), last_visit[:-1]])  # visits strictly before t
    seen = before >= 0
    paid = received.sum(axis=1)  # the reward of the arm chosen on each trial
    value = np.where(seen, paid[np.clip(before, 0, None)], 0.0)
    return [value, (~seen).astype(float)]


def _problem_features(
    rows: pd.DataFrame,
    n_lags: int,
    towers: list[str],
    tail_decay,
    extras: tuple[str, ...],
    binary_tail_decay,
    choice_tail_decay=None,
):
    """``(X, chosen, ports)`` of one problem; *rows* in trial order."""
    ports = _problem_ports(rows)
    n = len(rows)
    chosen = np.array([ports.index(c) for c in rows["choice"]], dtype=int)
    received = np.zeros((n, N_ARMS))
    received[np.arange(n), chosen] = rows["reward"].to_numpy(dtype=float)
    picked = np.zeros((n, N_ARMS))
    picked[np.arange(n), chosen] = 1.0
    # a new segment starts at each session start and after each dropped trial
    segment = np.cumsum((rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy())
    reward_lags = [_lagged(received, k) for k in range(1, n_lags + 1)]
    if tail_decay is not None:
        # the tail sums the rewards received more than n_lags trials ago
        reward_lags.append(_tail(_lagged(received, n_lags + 1), tail_decay))
    choice_lags = [
        _lagged(picked, k) * _same_segment(segment, k)[:, None] for k in range(1, n_lags + 1)
    ]
    tower_columns = [np.tile(np.array(ports) == tower, (n, 1)).astype(float) for tower in towers]
    extra_columns = []
    for extra in extras:
        if extra == BINARY:
            rewarded = (received > 0).astype(float)
            extra_columns += [_lagged(rewarded, k) for k in range(1, n_lags + 1)]
            if tail_decay is not None:
                extra_columns.append(_tail(_lagged(rewarded, n_lags + 1), binary_tail_decay))
        elif extra == CHOICE_TAIL:
            older = _lagged(picked, n_lags + 1) * _same_segment(segment, n_lags + 1)[:, None]
            extra_columns.append(_segment_tail(older, segment, choice_tail_decay))
        else:
            extra_columns += _believed_features(received, picked)
    X = np.stack(reward_lags + choice_lags + tower_columns + extra_columns, axis=2)
    return X, chosen, np.tile(np.array(ports, dtype=object), (n, 1))


def _same_segment(segment: np.ndarray, lag: int) -> np.ndarray:
    """1.0 where trial ``t-lag`` lies in the same session segment as trial ``t``, else 0."""
    same = np.zeros(len(segment))
    same[lag:] = segment[lag:] == segment[:-lag]
    return same


def feature_names(
    n_lags: int, towers: list[str], tail: bool = False, extras: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Design column names: reward lags, the tail if any, choice lags, the towers, then *extras*."""
    names = (
        tuple(f"reward_lag{k}" for k in range(1, n_lags + 1))
        + ((TAIL_FEATURE,) if tail else ())
        + tuple(f"{GROUP_PREFIX[CHOICE]}{k}" for k in range(1, n_lags + 1))
        + tuple(f"{GROUP_PREFIX[TOWER]}{tower}" for tower in towers)
    )
    for extra in extras:
        if extra == BINARY:
            names += tuple(f"binary_lag{k}" for k in range(1, n_lags + 1))
            names += (BINARY_TAIL_FEATURE,) if tail else ()
        elif extra == CHOICE_TAIL:
            names += (CHOICE_TAIL,)
        else:
            names += BELIEVED_FEATURES
    return names


def _warn_if_towers_disconnected(trials: pd.DataFrame) -> None:
    """Warn if the problems' tower sets fall into groups that share no tower.

    Tower biases are only comparable within a group of problems linked by shared towers; across
    groups the common offset of the biases is set by the penalty alone (each group's mean is
    shrunk to zero).
    """
    groups: list[set] = []
    for _, rows in trials.groupby("problem", sort=False):
        towers = {port for magnitudes in rows["magnitudes"] for port in magnitudes}
        linked = [g for g in groups if g & towers]
        merged = towers.union(*linked)
        groups = [g for g in groups if not (g & towers)] + [merged]
    if len(groups) > 1:
        warnings.warn(
            f"the problems of {trials['mouse'].iloc[0]} fall into {len(groups)} groups that share "
            "no tower; tower biases of different groups differ by an offset that only the L2 "
            "penalty fixes",
            stacklevel=3,
        )


def design_matrix(
    trials: pd.DataFrame,
    n_lags: int = DEFAULT_LAGS,
    tail_decay: float | None = None,
    extras: tuple[str, ...] = (),
    binary_tail_decay: float | None = None,
    choice_tail_decay: float | None = None,
) -> LogitDesign:
    """The logit's features for one mouse's trials.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table
            (:func:`src.behavior_analysis.data.drifting_trial_table.drifting_trial_table`); rows are
            put in trial order within each problem.
        n_lags (int): Number of reward lags and of choice lags. Default ``DEFAULT_LAGS``.
        tail_decay (float | None): Decay per trial of the reward tail, in (0, 1); ``None``
            (default) adds no tail feature.
        extras (tuple[str, ...]): Optional column groups to append, some of ``ALL_EXTRAS``
            (``BINARY``, ``BELIEVED``, ``CHOICE_TAIL``; see the module docstring), each at most
            once. Default none.
        binary_tail_decay (float | None): Decay of the ``binary_tail`` column, in (0, 1), so the
            binary coding can have the decay that suits it rather than the pulse tail's. Default
            ``None``: the same as *tail_decay*. Needs *tail_decay*.
        choice_tail_decay (float | None): Decay per trial of the ``choice_tail`` column, in
            (0, 1). Required with ``CHOICE_TAIL`` in *extras*, and only then.

    Returns:
        LogitDesign: Features (see the module docstring), chosen port and trial metadata. Rows
        follow the problems in order of first appearance.

    Raises:
        ValueError: If ``n_lags < 1``, *tail_decay* or *binary_tail_decay* is not in (0, 1) (or
            the latter is given without a tail), *choice_tail_decay* is not in (0, 1) or does not
            match ``CHOICE_TAIL`` in *extras*, *extras* names an unknown or a repeated group,
            *trials* hold more than one mouse or none, or a problem does not offer the
            same three ports on every trial.

    Warns:
        UserWarning: If the problems' tower sets form groups that share no tower.
    """
    if n_lags < 1:
        raise ValueError(f"n_lags must be at least 1, got {n_lags}")
    if tail_decay is not None and not 0 < tail_decay < 1:
        raise ValueError(f"tail_decay must be in (0, 1), got {tail_decay}")
    unknown = [e for e in extras if e not in ALL_EXTRAS]
    if unknown:
        raise ValueError(
            f"unknown extra column group {unknown}; expected some of {list(ALL_EXTRAS)}"
        )
    if len(set(extras)) != len(extras):
        raise ValueError(f"duplicate extra column groups in {list(extras)}")
    if binary_tail_decay is not None and (tail_decay is None or not 0 < binary_tail_decay < 1):
        raise ValueError(
            "binary_tail_decay must be in (0, 1) and needs a tail_decay, got "
            f"{binary_tail_decay} with tail_decay {tail_decay}"
        )
    if binary_tail_decay is None:
        binary_tail_decay = tail_decay
    if (CHOICE_TAIL in extras) != (choice_tail_decay is not None) or (
        choice_tail_decay is not None and not 0 < choice_tail_decay < 1
    ):
        raise ValueError(
            "choice_tail_decay must be in (0, 1) and given exactly with CHOICE_TAIL in extras, "
            f"got {choice_tail_decay} with extras {list(extras)}"
        )
    if trials["mouse"].nunique() != 1:
        raise ValueError(
            f"design_matrix takes the trials of one mouse, got {trials['mouse'].nunique()} mice"
        )
    _warn_if_towers_disconnected(trials)
    towers = sorted({port for magnitudes in trials["magnitudes"] for port in magnitudes})
    non_reference = towers[1:]
    parts, metas = [], []
    for _, rows in trials.groupby("problem", sort=False):
        rows = rows.sort_values("trial")
        parts.append(
            _problem_features(
                rows,
                n_lags,
                non_reference,
                tail_decay,
                extras,
                binary_tail_decay,
                choice_tail_decay,
            )
        )
        metas.append(rows[list(META_COLUMNS)])
    X, chosen, ports = (np.concatenate(part) for part in zip(*parts, strict=True))
    return LogitDesign(
        X,
        chosen,
        feature_names(n_lags, non_reference, tail_decay is not None, extras),
        pd.concat(metas, ignore_index=True),
        ports,
    )


def select_features(design: LogitDesign, groups: tuple[str, ...]) -> LogitDesign:
    """*design* with only the columns of the feature *groups*.

    The groups are ``REWARD``, ``CHOICE``, ``TOWER`` and, if the design has them, ``BINARY``,
    ``BELIEVED`` and ``CHOICE_TAIL``; a group the design lacks selects nothing.

    Raises:
        ValueError: For an unknown group.
    """
    unknown = [g for g in groups if g not in GROUP_PREFIX]
    if unknown:
        raise ValueError(f"unknown feature group {unknown}; expected some of {list(GROUP_PREFIX)}")
    prefixes = tuple(GROUP_PREFIX[g] for g in groups)
    keep = [i for i, name in enumerate(design.feature_names) if name.startswith(prefixes)]
    return design._replace(
        X=design.X[:, :, keep], feature_names=tuple(design.feature_names[i] for i in keep)
    )


# ---------------------------------------------------------------------------
# Likelihood and fit
# ---------------------------------------------------------------------------


def negative_log_likelihood(beta: np.ndarray, X: np.ndarray, chosen: np.ndarray, l2: float):
    """Penalised negative log likelihood of the choices and its gradient.

    The value is ``sum_t [logsumexp_a U_a(t) - U_chosen(t)] + l2/2 * |beta|^2``; the gradient
    of a softmax term is the model's expected features minus the chosen port's features.

    Args:
        beta (np.ndarray): ``(n_features,)`` weights.
        X (np.ndarray): ``(n_trials, 3, n_features)`` features.
        chosen (np.ndarray): ``(n_trials,)`` chosen port index.
        l2 (float): Penalty weight.

    Returns:
        tuple[float, np.ndarray]: The value and the ``(n_features,)`` gradient.
    """
    utility = X @ beta
    log_norm = logsumexp(utility, axis=1)
    rows = np.arange(len(chosen))
    value = float((log_norm - utility[rows, chosen]).sum() + 0.5 * l2 * beta @ beta)
    probability = np.exp(utility - log_norm[:, None])
    gradient = np.einsum("na,naf->f", probability, X) - X[rows, chosen].sum(axis=0) + l2 * beta
    return value, gradient


def soft_negative_log_likelihood(
    beta: np.ndarray, X: np.ndarray, targets: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    """Cross-entropy of the model's choice probabilities against *targets*, and its gradient.

    The generalisation of :func:`negative_log_likelihood` to a probability for each arm instead
    of a chosen arm (the one-hot case): ``sum_t [logsumexp_a U_a - sum_a q_a U_a] + l2/2 |beta|^2``.
    """
    utility = X @ beta
    log_norm = logsumexp(utility, axis=1)
    value = float((log_norm - (targets * utility).sum(axis=1)).sum() + 0.5 * l2 * beta @ beta)
    probability = np.exp(utility - log_norm[:, None])
    gradient = np.einsum("na,naf->f", probability - targets, X) + l2 * beta
    return value, gradient


def fit_soft_targets(
    X: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple,
    l2: float = 0.0,
    max_iterations: int = MAX_ITERATIONS,
) -> LogitFit:
    """The weights whose choice probabilities are closest (cross-entropy) to *targets*.

    With the generating probabilities of an agent as targets this is the logit's limit with
    unlimited data from that agent: the weights a fit to its choices would approach.

    Args:
        X (np.ndarray): ``(n_trials, 3, n_features)`` features.
        targets (np.ndarray): ``(n_trials, 3)`` target probabilities, rows summing to 1.
        feature_names (tuple[str, ...]): One name per feature.
        l2 (float): Penalty weight. Default 0.
        max_iterations (int): Iteration limit. Default ``MAX_ITERATIONS``.

    Returns:
        LogitFit: Its ``nll`` is the cross-entropy at the optimum.

    Raises:
        ValueError: If the shapes disagree or the rows of *targets* do not sum to 1.
        RuntimeError: If the optimiser does not converge.
    """
    if targets.shape != X.shape[:2] or X.shape[2] != len(feature_names):
        raise ValueError(f"targets {targets.shape} and X {X.shape} do not match")
    if not np.allclose(targets.sum(axis=1), 1.0):
        raise ValueError("each row of targets must sum to 1")
    result = minimize(
        soft_negative_log_likelihood,
        np.zeros(X.shape[2]),
        args=(X, targets, l2),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(f"the soft-target logit did not converge: {result.message}")
    return LogitFit(result.x, tuple(feature_names), result.fun, len(X), l2)


def log_likelihood(X: np.ndarray, chosen: np.ndarray, beta: np.ndarray) -> float:
    """Summed log probability of the chosen ports under weights *beta* (no penalty)."""
    utility = X @ beta
    return float((utility[np.arange(len(chosen)), chosen] - logsumexp(utility, axis=1)).sum())


def fit_logit(
    X: np.ndarray,
    chosen: np.ndarray,
    feature_names: tuple,
    l2: float = DEFAULT_L2,
    max_iterations: int = MAX_ITERATIONS,
) -> LogitFit:
    """Fit the weights by L-BFGS-B on the penalised negative log likelihood.

    The objective is convex, so one start from zero finds the optimum.

    Args:
        X (np.ndarray): ``(n_trials, 3, n_features)`` features.
        chosen (np.ndarray): ``(n_trials,)`` chosen port index, 0-2.
        feature_names (tuple[str, ...]): One name per feature.
        l2 (float): Penalty weight, ``>= 0``. Default ``DEFAULT_L2``.
        max_iterations (int): Iteration limit of the optimiser. Default ``MAX_ITERATIONS``.

    Returns:
        LogitFit: The fitted weights.

    Raises:
        ValueError: If the shapes disagree, a chosen index is not 0-2, or ``l2 < 0``.
        RuntimeError: If the optimiser does not converge.
    """
    if X.ndim != 3 or X.shape[1] != N_ARMS or X.shape[2] != len(feature_names):
        raise ValueError(f"X must be (trials, 3, {len(feature_names)}), got {X.shape}")
    if (
        chosen.shape != (X.shape[0],)
        or chosen.min(initial=0) < 0
        or chosen.max(initial=0) >= N_ARMS
    ):
        raise ValueError("chosen must hold one port index (0-2) per trial")
    if l2 < 0:
        raise ValueError(f"l2 must be non-negative, got {l2}")
    result = minimize(
        negative_log_likelihood,
        np.zeros(X.shape[2]),
        args=(X, chosen, l2),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(f"the choice logit did not converge: {result.message}")
    return LogitFit(
        result.x, tuple(feature_names), -log_likelihood(X, chosen, result.x), len(chosen), l2
    )


# ---------------------------------------------------------------------------
# Model value and weights
# ---------------------------------------------------------------------------


def model_values(X: np.ndarray, fit: LogitFit) -> np.ndarray:
    """The reward part of the utility, lags and tail, for each trial and port.

    Args:
        X (np.ndarray): ``(n_trials, 3, n_features)`` features with the columns of *fit*.
        fit (LogitFit): Weights; only the reward ones (``reward_lag<k>``, ``reward_tail``) are used.

    Returns:
        np.ndarray: ``(n_trials, 3)`` model values in logit units.

    Raises:
        ValueError: If *fit* has no reward weights or the feature count differs.
    """
    if X.shape[2] != len(fit.beta):
        raise ValueError(f"X has {X.shape[2]} features, the fit {len(fit.beta)}")
    reward = [i for i, n in enumerate(fit.feature_names) if n.startswith(GROUP_PREFIX[REWARD])]
    if not reward:
        raise ValueError("the fit has no reward weights, so there is no model value")
    return X[:, :, reward] @ fit.beta[reward]


def weights_table(mouse: str, feature_names: tuple, beta: np.ndarray) -> pd.DataFrame:
    """One row per weight: ``mouse, feature, group, lag`` (NaN for towers), ``weight``."""
    rows = []
    for name, weight in zip(feature_names, beta, strict=True):
        group = next(g for g, prefix in GROUP_PREFIX.items() if name.startswith(prefix))
        has_lag = name.startswith(("reward_lag", "choice_lag", "binary_lag"))
        lag = float(name.rsplit("lag", 1)[1]) if has_lag else np.nan
        rows.append(dict(mouse=mouse, feature=name, group=group, lag=lag, weight=float(weight)))
    return pd.DataFrame(rows, columns=["mouse", "feature", "group", "lag", "weight"])
