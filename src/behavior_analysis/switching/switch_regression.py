"""Where does a mouse go when it switches? A two-alternative logit on switch trials.

On a **switch trial** the mouse leaves the arm it chose on the previous trial. The pair of
trials is a stay/switch decision of :func:`src.agents.null_agents.decisions`: consecutive trials
of one segment, where a segment starts at a problem's start, a session's start and after a
dropped trial (``trial_in_session == 0`` or ``after_gap``). The **choice set** is the two arms
other than the previous choice. A problem always offers the same three arms
(:func:`src.behavior_analysis.choice_logit.choice_logit.design_matrix` raises otherwise), and an arm
paying 0
is still available, so the choice set never needs to drop an arm.

The utility of alternative ``a`` is

    U(a) = b_V V(a) + sum_{k=2..K} b_k c_a(t-k) + b_tr h_a(t) + b_rec rec_a(t) + b_nev nev_a(t)
           + bias[tower a]

and the first alternative (the lower arm in sorted port order) is chosen with probability
``sigmoid(U(first) - U(second))``. It is the conditional logit of
:mod:`src.behavior_analysis.choice_logit.choice_logit` on two alternatives: a logistic regression on
the
difference of the two alternatives' features with **no intercept**. An intercept would be a bias
towards the lower-sorted arm, which is an artifact of the ordering and not a property of a port;
the port biases are the tower terms. Without it, swapping the two labels flips the sign of every
difference and of the outcome together, so weights and likelihood do not depend on the ordering.

History features (computed on all trials, then read on the switch trials):

- ``choice_lag<k>``, k = 2..K: 1 if ``a`` was chosen on trial ``t-k`` of the same segment (the
  logit's choice lags; lag 1 is the previous choice, which is never in the choice set, so it is
  0 for both alternatives and left out);
- ``trace``: ``h_a(t) = sum_{k>=1} decay**(k-1) [a chosen on t-k]`` within the segment, the
  win-stay/lose-shift agents' history trace (:mod:`src.agents.null_agents`). It reaches beyond the
  K lags, which the logit's choice baseline does not: a switch rule driven by a slow trace is
  nested in this model. The decay is chosen per mouse from a grid by held-out likelihood;
- ``recency``: ``log(1 + min(trials since a was last chosen, cap))``, counted within the problem
  across sessions (trials, not time), so it keeps a memory the segment-bound trace drops; capped
  so that very old visits all look alike;
- ``never_chosen``: 1 if ``a`` has not been chosen yet in the problem (its recency is then the
  capped value). It is the same indicator as the believed value's never-visited column;
- ``tower_<name>``: one bias per tower, the logit's (alphabetically first tower as reference).

Value ``V(a)``, one variant per fit, each added to the history features as the column ``value``:

- ``true``: the magnitude arm ``a`` offers on the switch trial, the convention of
  :func:`src.pipelines.choice_logit.choice_logit_values.task_values` (pulses);
- ``believed``: the magnitude ``a`` paid when last chosen in the problem, 0 before its first visit
  (the ``believed_value`` column of the logit, pulses); its never-visited indicator is
  ``never_chosen``, already in the history model;
- ``model``: the held-out reward part of the choice logit (logits). It is built from the same
  reward history as the believed value, so it is a consistency check, not an independent test.

Everything here is pure: trial tables and arrays in, arrays and tables out.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit

from src.agents import null_agents as na
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import choice_logit_extensions as ext

AFTER_ZERO, AFTER_NONZERO = "after_zero", "after_nonzero"
CONDITIONS = (AFTER_ZERO, AFTER_NONZERO)
TRUE, BELIEVED, MODEL = "true", "believed", "model"
VALUES = (TRUE, BELIEVED, MODEL)
HISTORY = "history"  # the model without a value column
VALUE_FEATURE = "value"
TRACE, RECENCY, NEVER = "trace", "recency", "never_chosen"
# Decays of the choice trace tried per mouse; 0.9 is the null agents' decay, 0.95 a longer one.
TRACE_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
# Trials since an arm was last chosen are capped here: about three blocks (30-35 trials each),
# beyond which an old visit says little more about the arm than "long ago".
RECENCY_CAP = 100
# Switches a mouse needs in each condition. The model has about 15 weights (4 lags, 3 recency
# terms, up to 8 towers, the value); with fewer switches they rest mostly on the L2 prior.
MIN_SWITCHES = 100
# Firth's bias reduction in every fit (see fit_switch_logit).
FIRTH = False


class TooFewSwitchesError(ValueError):
    """A mouse has too few switches (in a condition, or in too few problems) to be analyzed."""


class ModelValues(NamedTuple):
    """Per-trial values of each arm with the keys of the trials they belong to.

    Attributes:
        problem (np.ndarray): ``(n_trials,)`` problem of each row.
        trial (np.ndarray): ``(n_trials,)`` trial (within the problem) of each row.
        values (np.ndarray): ``(n_trials, 3)`` values in the design's port order, e.g. the
            held-out model values of ``choice_logit_cv.MouseCV.values``.
    """

    problem: np.ndarray
    trial: np.ndarray
    values: np.ndarray


class SwitchData(NamedTuple):
    """The switch trials of one mouse, as two-alternative choices.

    Attributes:
        X (np.ndarray): ``(n_switches, 2, n_features)`` history features of the two alternatives.
        chosen (np.ndarray): ``(n_switches,)`` 0 if the first alternative was chosen, else 1.
        feature_names (tuple[str, ...]): Names of the last axis of ``X``.
        values (dict[str, np.ndarray]): ``{variant: (n_switches, 2)}`` values of the alternatives.
        after_zero (np.ndarray): ``(n_switches,)`` the previous choice paid 0 pulses.
        problem (np.ndarray): ``(n_switches,)`` problem of each switch.
        alternatives (np.ndarray): ``(n_switches, 2)`` arm index (0-2, sorted port order) of the
            two alternatives, the lower first.
        rows (np.ndarray): ``(n_switches,)`` row of each switch in the mouse's design matrix.
        mouse (str): The mouse.
    """

    X: np.ndarray
    chosen: np.ndarray
    feature_names: tuple
    values: dict
    after_zero: np.ndarray
    problem: np.ndarray
    alternatives: np.ndarray
    rows: np.ndarray
    mouse: str


class SwitchFit(NamedTuple):
    """Penalized maximum-likelihood weights of one switch logit.

    Attributes:
        beta (np.ndarray): ``(n_features,)`` weights.
        se (np.ndarray): ``(n_features,)`` standard errors: square roots of the diagonal of the
            inverse Hessian of the penalized negative log likelihood (the Laplace approximation
            to the posterior SD under the Gaussian prior of the penalty).
        feature_names (tuple[str, ...]): Name of each weight.
        nll (float): Summed negative log likelihood of the fitted switches (no penalty).
        n_switches (int): Switches fitted.
    """

    beta: np.ndarray
    se: np.ndarray
    feature_names: tuple
    nll: float
    n_switches: int


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _one_hot(chosen: np.ndarray) -> np.ndarray:
    picked = np.zeros((len(chosen), cl.N_ARMS))
    picked[np.arange(len(chosen)), chosen] = 1.0
    return picked


def choice_trace(chosen: np.ndarray, start: np.ndarray, decay: float) -> np.ndarray:
    """``(n, 3)`` decayed count of each arm's past choices before each trial, per segment.

    Args:
        chosen (np.ndarray): ``(n,)`` chosen arm index.
        start (np.ndarray): ``(n,)`` bool, the trial starts a segment (the trace is cleared).
        decay (float): Decay per trial, in (0, 1).

    Returns:
        np.ndarray: ``h[t, a] = sum_{k>=1} decay**(k-1) [chosen[t-k] == a]`` over the segment.
    """
    picked = _one_hot(chosen)
    bounds = [*np.flatnonzero(start), len(chosen)]
    h = np.zeros_like(picked)
    for first, last in zip(bounds[:-1], bounds[1:], strict=True):
        # y[t] = decay * y[t-1] + picked[t-1]
        h[first:last] = lfilter([0.0, 1.0], [1.0, -decay], picked[first:last], axis=0)
    return h


def trials_since_chosen(chosen: np.ndarray, problem_start: np.ndarray) -> np.ndarray:
    """``(n, 3)`` trials since each arm was last chosen in the problem, NaN if never.

    Args:
        chosen (np.ndarray): ``(n,)`` chosen arm index, rows in trial order within problems.
        problem_start (np.ndarray): ``(n,)`` bool, the row is a problem's first trial.

    Returns:
        np.ndarray: ``t - (last t' < t with chosen[t'] == a)`` within the problem.
    """
    position = np.arange(len(chosen))
    last = np.maximum.accumulate(np.where(_one_hot(chosen) == 1, position[:, None], -1), axis=0)
    before = np.vstack([np.full((1, cl.N_ARMS), -1), last[:-1]])  # visits strictly before t
    first_row = np.maximum.accumulate(np.where(problem_start, position, 0))
    seen = before >= first_row[:, None]
    return np.where(seen, position[:, None] - before, np.nan)


def _arm_values(design: cl.LogitDesign) -> dict:
    """``{"true", "believed"}`` values of every trial and arm, ``(n, 3)`` each."""
    true = np.array(
        [[m[p] for p in ports] for m, ports in zip(design.meta["magnitudes"], design.ports,
                                                    strict=True)],
        dtype=float,
    )  # fmt: skip
    believed = design.X[:, :, design.feature_names.index("believed_value")]
    return {TRUE: true, BELIEVED: believed}


def _check_model_values(design: cl.LogitDesign, model_values: ModelValues) -> np.ndarray:
    meta = design.meta
    aligned = (
        len(model_values.trial) == len(meta)
        and np.array_equal(model_values.trial, meta["trial"].to_numpy())
        and list(model_values.problem) == list(meta["problem"])
    )
    if not aligned or model_values.values.shape != (len(meta), cl.N_ARMS):
        raise ValueError(
            "model values are not aligned with the trials of the design (problem and trial "
            "order must match choice_logit.design_matrix of the same trials)"
        )
    return model_values.values


def _arm_features(design: cl.LogitDesign, cap: int):
    """``(names, (n, 3, f))``: lags 2..K, trace (zeros, see ``_with_decay``), recency, never
    chosen and towers of every trial."""
    names = list(design.feature_names)
    lag_names = [
        n for n in names if n.startswith(cl.GROUP_PREFIX[cl.CHOICE]) and n != "choice_lag1"
    ]
    tower_names = [n for n in names if n.startswith(cl.GROUP_PREFIX[cl.TOWER])]
    problem_start = np.r_[True, design.meta["problem"].to_numpy()[1:] != design.meta[
        "problem"].to_numpy()[:-1]]  # fmt: skip
    since = trials_since_chosen(design.chosen, problem_start)
    never = np.isnan(since)
    recency = np.log1p(np.minimum(np.where(never, cap, since), cap))
    columns = (
        [design.X[:, :, names.index(n)] for n in lag_names]
        + [np.zeros_like(recency), recency, never.astype(float)]  # trace: set per decay
        + [design.X[:, :, names.index(n)] for n in tower_names]
    )
    return tuple(lag_names + [TRACE, RECENCY, NEVER] + tower_names), np.stack(columns, axis=2)


class _SwitchParts(NamedTuple):
    """Everything of :func:`switch_data` that does not depend on the trace decay."""

    names: tuple
    features: np.ndarray  # (n_trials, 3, f) per-arm features, the trace column still empty
    chosen: np.ndarray  # (n_trials,) chosen arm of every trial
    start: np.ndarray  # (n_trials,) the trial starts a segment
    rows: np.ndarray
    alternatives: np.ndarray
    arm_values: dict
    data: "SwitchData"  # the decay-independent fields; X is filled in per decay


def _switch_parts(
    trials: pd.DataFrame, n_lags: int, recency_cap: int, model_values: ModelValues | None
) -> _SwitchParts:
    if recency_cap < 1:
        raise ValueError(f"recency_cap must be at least 1, got {recency_cap}")
    design = cl.design_matrix(trials, n_lags, None, extras=(cl.BELIEVED,))
    decisions = na.decisions(trials, 0.5)  # the decay of the agents' trace plays no part here
    if len(decisions.start) != len(design.chosen):
        raise RuntimeError("the stay/switch decisions and the design have different trials")
    mouse = design.meta["mouse"].iloc[0]
    rows = np.flatnonzero(decisions.switched)
    if len(rows) == 0:
        raise TooFewSwitchesError(f"mouse {mouse} never switches")
    names, features = _arm_features(design, recency_cap)
    alternatives = na.OTHERS[design.chosen[rows - 1]]
    arm_values = _arm_values(design)
    if model_values is not None:
        arm_values[MODEL] = _check_model_values(design, model_values)
    data = SwitchData(
        X=np.empty((len(rows), 2, 0)),
        chosen=(design.chosen[rows] == alternatives[:, 1]).astype(int),
        feature_names=names,
        values={kind: np.take_along_axis(v[rows], alternatives, axis=1)
                for kind, v in arm_values.items()},
        after_zero=decisions.prev_magnitude[rows] == 0,
        problem=design.meta["problem"].to_numpy()[rows],
        alternatives=alternatives,
        rows=rows,
        mouse=mouse,
    )  # fmt: skip
    return _SwitchParts(
        names, features, design.chosen, decisions.start, rows, alternatives, arm_values, data
    )


def _with_decay(parts: _SwitchParts, decay: float) -> "SwitchData":
    if not 0 < decay < 1:
        raise ValueError(f"trace_decay must be in (0, 1), got {decay}")
    features = parts.features.copy()
    features[:, :, parts.names.index(TRACE)] = choice_trace(parts.chosen, parts.start, decay)
    X = np.take_along_axis(features[parts.rows], parts.alternatives[:, :, None], axis=1)
    return parts.data._replace(X=X)


def switch_data(
    trials: pd.DataFrame,
    n_lags: int = cl.DEFAULT_LAGS,
    trace_decay: float = 0.9,
    recency_cap: int = RECENCY_CAP,
    model_values: ModelValues | None = None,
) -> SwitchData:
    """The switch trials of one mouse with the history features and values of both alternatives.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        n_lags (int): K, the last choice lag (lags 2..K are features). Default
            ``choice_logit.DEFAULT_LAGS``.
        trace_decay (float): Decay of the choice trace, in (0, 1). Default 0.9.
        recency_cap (int): Cap of the trials since an arm was chosen, at least 1. Default
            ``RECENCY_CAP``.
        model_values (ModelValues | None): Values of the ``model`` variant, aligned with the
            design of *trials*; ``None`` (default) leaves the variant out.

    Returns:
        SwitchData: One row per switch trial, problems in order of first appearance.

    Raises:
        TooFewSwitchesError: If the mouse never switches.
        ValueError: As ``choice_logit.design_matrix`` (one mouse, three offered ports per
            problem), for a decay outside (0, 1) or a cap below 1, or if *model_values* are not
            aligned with the trials.

    Warns:
        UserWarning: If the problems' tower sets form groups that share no tower (from
            ``design_matrix``).
    """
    return switch_data_by_decay(trials, (trace_decay,), n_lags, recency_cap, model_values)[
        trace_decay
    ]


def switch_data_by_decay(
    trials: pd.DataFrame,
    decays: Sequence[float],
    n_lags: int = cl.DEFAULT_LAGS,
    recency_cap: int = RECENCY_CAP,
    model_values: ModelValues | None = None,
) -> dict:
    """``{decay: SwitchData}``: :func:`switch_data` for several trace decays, one design built.

    Only the trace column depends on the decay, so the design matrix, the switch trials and the
    values are computed once. Raises as :func:`switch_data`.
    """
    if len(decays) == 0:
        raise ValueError("the trace decay grid is empty")
    for decay in decays:
        if not 0 < decay < 1:
            raise ValueError(f"trace_decay must be in (0, 1), got {decay}")
    parts = _switch_parts(trials, n_lags, recency_cap, model_values)
    return {decay: _with_decay(parts, decay) for decay in decays}


def select_condition(data: SwitchData, condition: str) -> SwitchData:
    """The switches after zero (``AFTER_ZERO``) or after a reward (``AFTER_NONZERO``).

    Raises:
        ValueError: For an unknown condition.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    keep = data.after_zero if condition == AFTER_ZERO else ~data.after_zero
    return data._replace(
        X=data.X[keep],
        chosen=data.chosen[keep],
        values={k: v[keep] for k, v in data.values.items()},
        after_zero=data.after_zero[keep],
        problem=data.problem[keep],
        alternatives=data.alternatives[keep],
        rows=data.rows[keep],
    )


def switch_counts(data: SwitchData) -> dict:
    """``{condition: number of switches}``."""
    return {AFTER_ZERO: int(data.after_zero.sum()), AFTER_NONZERO: int((~data.after_zero).sum())}


def model_design(data: SwitchData, value: str | None) -> tuple[np.ndarray, tuple]:
    """``(X, feature names)``: the history features, plus the ``value`` column of a variant.

    Args:
        data (SwitchData): Switch trials.
        value (str | None): A key of ``data.values``, or ``None`` for the history-only model.

    Raises:
        ValueError: If *data* has no such value variant.
    """
    if value is None:
        return data.X, data.feature_names
    if value not in data.values:
        raise ValueError(f"no {value!r} values; the data have {sorted(data.values)}")
    X = np.concatenate([data.values[value][:, :, None], data.X], axis=2)
    return X, (VALUE_FEATURE, *data.feature_names)


# ---------------------------------------------------------------------------
# Fit and held-out likelihood
# ---------------------------------------------------------------------------


def _information(beta: np.ndarray, difference: np.ndarray, l2: float) -> np.ndarray:
    """``sum p(1-p) d d' + l2 I``: Hessian of the L2-penalized negative log likelihood."""
    p_first = expit(difference @ beta)
    weight = p_first * (1 - p_first)
    return (difference * weight[:, None]).T @ difference + l2 * np.eye(len(beta))


def _objective(beta, difference, first, l2, firth):
    """Penalized negative log likelihood of the two-alternative logit, and its gradient.

    ``P(first) = sigmoid(d @ beta)`` with ``d`` the first alternative's features minus the
    second's. The L2 term is ``l2/2 |beta|^2``. With *firth* the Jeffreys term ``-1/2 log det
    M`` is added, ``M`` the information of :func:`_information` (the prior included, so a column
    that never varies keeps it invertible); its gradient is ``-sum_i h_i (1/2 - p_i) d_i`` with
    ``h_i = p_i (1 - p_i) d_i' M^-1 d_i`` the leverage of switch ``i``.
    """
    z = difference @ beta
    p_first = expit(z)
    value = float((np.logaddexp(0, z) - first * z).sum() + 0.5 * l2 * beta @ beta)
    gradient = difference.T @ (p_first - first) + l2 * beta
    if firth:
        information = _information(beta, difference, l2)
        _, log_det = np.linalg.slogdet(information)
        solved = np.linalg.solve(information, difference.T).T
        leverage = p_first * (1 - p_first) * (difference * solved).sum(axis=1)
        value -= 0.5 * log_det
        gradient -= difference.T @ (leverage * (0.5 - p_first))
    return value, gradient


def fit_switch_logit(
    X: np.ndarray,
    chosen: np.ndarray,
    feature_names: Sequence[str],
    l2: float = cl.DEFAULT_L2,
    max_iterations: int = cl.MAX_ITERATIONS,
    firth: bool = FIRTH,
) -> SwitchFit:
    """Fit the two-alternative logit by L-BFGS-B on the penalized negative log likelihood.

    It is :func:`choice_logit.negative_log_likelihood` on two alternatives, written on the
    difference of their features. With the L2 penalty alone the objective is convex; Firth's
    Jeffreys-prior term (*firth*) removes the first-order small-sample bias of the estimates.

    Args:
        X (np.ndarray): ``(n, 2, n_features)`` features of the two alternatives.
        chosen (np.ndarray): ``(n,)`` 0 or 1, the chosen alternative.
        feature_names (Sequence[str]): One name per feature.
        l2 (float): Penalty, ``>= 0`` (a Gaussian prior with SD ``1/sqrt(l2)``). Default
            ``choice_logit.DEFAULT_L2``.
        max_iterations (int): Iteration limit. Default ``choice_logit.MAX_ITERATIONS``.
        firth (bool): Add Firth's bias-reducing penalty. Default ``FIRTH``.

    Returns:
        SwitchFit: Weights, standard errors and the unpenalized negative log likelihood.

    Raises:
        ValueError: If the shapes disagree, *chosen* holds other than 0 and 1, or ``l2 < 0``.
        RuntimeError: If the optimizer does not converge.
    """
    if X.ndim != 3 or X.shape[1] != 2 or X.shape[2] != len(feature_names):
        raise ValueError(
            f"X must be (switches, two alternatives, {len(feature_names)}), got {X.shape}"
        )
    chosen = np.asarray(chosen)
    if chosen.shape != (len(X),) or not np.isin(chosen, (0, 1)).all():
        raise ValueError("chosen must hold 0 or 1 (the chosen alternative) for each switch")
    if l2 < 0:
        raise ValueError(f"l2 must be non-negative, got {l2}")
    difference = X[:, 0] - X[:, 1]
    first = (chosen == 0).astype(float)
    result = minimize(
        _objective,
        np.zeros(X.shape[2]),
        args=(difference, first, l2, firth),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(f"the switch logit did not converge: {result.message}")
    se = np.sqrt(np.diag(np.linalg.inv(_information(result.x, difference, l2))))
    return SwitchFit(
        result.x, se, tuple(feature_names), -cl.log_likelihood(X, chosen, result.x), len(chosen)
    )


def _condition_of(data: SwitchData) -> str:
    """The condition of *data*'s switches, or ``"in both conditions"`` if they are mixed."""
    if data.after_zero.all():
        return "after zero"
    return "after non-zero" if not data.after_zero.any() else "in both conditions"


def cross_validate(
    data: SwitchData, value: str | None, l2: float = cl.DEFAULT_L2, firth: bool = FIRTH
) -> pd.DataFrame:
    """Leave-one-problem-out log likelihood of one model on the switches of *data*.

    Each problem with switches is held out once; the model is fitted on the other problems'
    switches. A problem without switches is no fold.

    Args:
        data (SwitchData): Switch trials (usually of one condition).
        value (str | None): Value variant, or ``None`` for the history-only model.
        l2 (float): Penalty. Default ``choice_logit.DEFAULT_L2``.
        firth (bool): Firth's bias reduction in every fit. Default ``FIRTH``.

    Returns:
        pd.DataFrame: ``problem, n_switches, held_out_log_likelihood``, one row per fold.

    Raises:
        TooFewSwitchesError: If the switches come from fewer than two problems.
    """
    codes, problems = pd.factorize(pd.Series(data.problem))
    if len(problems) < 2:
        raise TooFewSwitchesError(
            f"mouse {data.mouse}: switches {_condition_of(data)} in {len(problems)} problem(s); "
            "leave-one-problem-out validation needs at least two"
        )
    X, names = model_design(data, value)
    rows = []
    for code, problem in enumerate(problems):
        test = codes == code
        fit = fit_switch_logit(X[~test], data.chosen[~test], names, l2, firth=firth)
        rows.append(
            (problem, int(test.sum()), cl.log_likelihood(X[test], data.chosen[test], fit.beta))
        )
    return pd.DataFrame(rows, columns=["problem", "n_switches", "held_out_log_likelihood"])


# ---------------------------------------------------------------------------
# One mouse
# ---------------------------------------------------------------------------


class MouseSwitchRegression(NamedTuple):
    """The switch regression of one mouse.

    Attributes:
        scores (pd.DataFrame): ``mouse, condition, model, problem, n_switches,
            held_out_log_likelihood``; ``model`` is ``HISTORY`` or a value variant.
        weights (pd.DataFrame): ``mouse, condition, model, feature, weight, se`` of each model
            fitted to all of the condition's switches.
        counts (pd.DataFrame): ``mouse, condition, n_switches, n_problems``.
        decay_table (pd.DataFrame): ``mouse, trace_decay, nats_per_switch``: held-out nats per
            switch of the history-only models (both conditions pooled) for each candidate decay.
        trace_decay (float): The decay used, the best of *decay_table*.
    """

    scores: pd.DataFrame
    weights: pd.DataFrame
    counts: pd.DataFrame
    decay_table: pd.DataFrame
    trace_decay: float


def _check_counts(data: SwitchData, min_switches: int) -> None:
    for condition, n in switch_counts(data).items():
        if n < min_switches:
            raise TooFewSwitchesError(
                f"mouse {data.mouse}: {n} switches {condition.replace('_', ' ')}; at least "
                f"{min_switches} are needed"
            )


def _history_nats(by_condition: dict, l2: float, firth: bool) -> float:
    """Held-out nats per switch of the history-only models, pooled over the conditions."""
    folds = pd.concat([cross_validate(d, None, l2, firth) for d in by_condition.values()])
    return float(-folds["held_out_log_likelihood"].sum() / folds["n_switches"].sum())


def analyze_mouse(
    trials: pd.DataFrame,
    n_lags: int = cl.DEFAULT_LAGS,
    trace_grid: Sequence[float] = TRACE_GRID,
    l2: float = cl.DEFAULT_L2,
    recency_cap: int = RECENCY_CAP,
    min_switches: int = MIN_SWITCHES,
    model_values: ModelValues | None = None,
    firth: bool = FIRTH,
) -> MouseSwitchRegression:
    """Choose the trace decay, then fit and validate every model in both conditions.

    The decay is the one of *trace_grid* whose history-only models (no value) have the best
    held-out likelihood, both conditions pooled, and every value variant of the mouse uses it.
    Two consequences. The decay is chosen on the same folds that then score the gain, so the
    history-only model is favored and the gain of the value is likely biased down. And for a
    mouse whose switches do follow value, a decay chosen without the value lets the trace take
    up part of what the value explains, which biases a real value weight towards 0. Both make the
    analysis conservative for a real effect; neither protects against a bias of the null (see
    the null calibration in :mod:`src.pipelines.switching.switch_regression`).

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        n_lags (int): Last choice lag K. Default ``choice_logit.DEFAULT_LAGS``.
        trace_grid (Sequence[float]): Candidate trace decays. Default ``TRACE_GRID``.
        l2 (float): Penalty. Default ``choice_logit.DEFAULT_L2``.
        recency_cap (int): Cap of trials since chosen. Default ``RECENCY_CAP``.
        min_switches (int): Switches needed in each condition. Default ``MIN_SWITCHES``.
        model_values (ModelValues | None): Values of the ``model`` variant; ``None`` leaves it out.
        firth (bool): Firth's bias reduction in every fit. Default ``FIRTH``.

    Returns:
        MouseSwitchRegression: Scores, weights, switch counts and the decay selection.

    Raises:
        TooFewSwitchesError: If a condition has fewer than *min_switches* switches or switches
            in fewer than two problems.
        ValueError: If *trace_grid* is empty, or as :func:`switch_data`.
        RuntimeError: If a fit does not converge.
    """
    candidates = switch_data_by_decay(trials, trace_grid, n_lags, recency_cap, model_values)
    _check_counts(next(iter(candidates.values())), min_switches)
    split = {
        d: {c: select_condition(data, c) for c in CONDITIONS} for d, data in candidates.items()
    }
    nats = {d: _history_nats(by_condition, l2, firth) for d, by_condition in split.items()}
    best = min(trace_grid, key=lambda d: nats[d])
    mouse = candidates[best].mouse
    models = (HISTORY, *candidates[best].values)
    scores, weights, counts = [], [], []
    for condition, data in split[best].items():
        counts.append((mouse, condition, len(data.chosen), len(set(data.problem))))
        for model in models:
            value = None if model == HISTORY else model
            folds = cross_validate(data, value, l2, firth)
            scores.append(folds.assign(mouse=mouse, condition=condition, model=model))
            X, names = model_design(data, value)
            fit = fit_switch_logit(X, data.chosen, names, l2, firth=firth)
            weights.append(
                pd.DataFrame(
                    dict(mouse=mouse, condition=condition, model=model, feature=names,
                         weight=fit.beta, se=fit.se)
                )
            )  # fmt: skip
    score_columns = [
        "mouse",
        "condition",
        "model",
        "problem",
        "n_switches",
        "held_out_log_likelihood",
    ]
    return MouseSwitchRegression(
        scores=pd.concat(scores, ignore_index=True)[score_columns],
        weights=pd.concat(weights, ignore_index=True),
        counts=pd.DataFrame(counts, columns=["mouse", "condition", "n_switches", "n_problems"]),
        decay_table=pd.DataFrame(
            dict(
                mouse=mouse,
                trace_decay=list(trace_grid),
                nats_per_switch=[nats[d] for d in trace_grid],
            )
        ),
        trace_decay=best,
    )


# ---------------------------------------------------------------------------
# Tables across mice
# ---------------------------------------------------------------------------


def value_table(scores: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Per mouse, condition and value variant: the value weight and its held-out gain.

    Args:
        scores (pd.DataFrame): ``MouseSwitchRegression.scores`` of one or more mice.
        weights (pd.DataFrame): ``MouseSwitchRegression.weights`` of the same mice.

    Returns:
        pd.DataFrame: ``mouse, condition, value, beta, se`` (the value weight of the fit to all
        switches, per pulse for ``true`` and ``believed``, per logit for ``model``),
        ``nats_history, nats_value`` (held-out nats per switch) and ``gain`` (``nats_history -
        nats_value``: positive when adding the value predicts held-out switches better).
    """
    totals = scores.groupby(["mouse", "condition", "model"], sort=False)[
        ["held_out_log_likelihood", "n_switches"]
    ].sum()
    nats = (-totals["held_out_log_likelihood"] / totals["n_switches"]).rename("nats").reset_index()
    history = (
        nats[nats["model"] == HISTORY]
        .drop(columns="model")
        .rename(columns={"nats": "nats_history"})
    )
    valued = nats[nats["model"] != HISTORY].rename(columns={"model": "value", "nats": "nats_value"})
    beta = weights[(weights["feature"] == VALUE_FEATURE) & (weights["model"] != HISTORY)].rename(
        columns={"model": "value", "weight": "beta"}
    )[["mouse", "condition", "value", "beta", "se"]]
    table = beta.merge(valued, on=["mouse", "condition", "value"]).merge(
        history, on=["mouse", "condition"]
    )
    table["gain"] = table["nats_history"] - table["nats_value"]
    return table[
        ["mouse", "condition", "value", "beta", "se", "nats_history", "nats_value", "gain"]
    ]


def history_weight_table(weights: pd.DataFrame) -> pd.DataFrame:
    """The history-only model's weights without the towers: ``mouse, condition, feature, weight,
    se``."""
    history = weights[(weights["model"] == HISTORY) & ~weights["feature"].str.startswith("tower_")]
    return history.drop(columns="model").reset_index(drop=True)


def random_effects(beta: np.ndarray, se: np.ndarray) -> tuple[float, float, float]:
    """DerSimonian-Laird random-effects mean of per-mouse estimates.

    Each mouse is weighted by ``1 / (se**2 + tau**2)``, where ``tau**2`` is the method-of-moments
    estimate of the variance between mice (0 when the mice agree within their errors, which
    makes it the inverse-variance mean).

    Args:
        beta (np.ndarray): Per-mouse estimates.
        se (np.ndarray): Their standard errors, all positive.

    Returns:
        tuple[float, float, float]: Mean, its standard error and ``tau**2``.

    Raises:
        ValueError: If the lengths differ, there are fewer than two mice, or an error is not
            positive.
    """
    beta, se = np.asarray(beta, dtype=float), np.asarray(se, dtype=float)
    if beta.shape != se.shape or len(beta) < 2 or not np.all(se > 0):
        raise ValueError("random_effects needs two or more estimates with positive errors")
    w = 1 / se**2
    fixed = (w * beta).sum() / w.sum()
    q = (w * (beta - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (len(beta) - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    w_star = 1 / (se**2 + tau2)
    return float((w_star * beta).sum() / w_star.sum()), float(1 / np.sqrt(w_star.sum())), tau2


def group_summary(table: pd.DataFrame, min_mice: int = ext.MIN_TEST_MICE) -> pd.DataFrame:
    """Across mice: mean ± se, Wilcoxon and sign tests of the value weight and the gain.

    Args:
        table (pd.DataFrame): :func:`value_table` of one source.
        min_mice (int): With fewer mice everything but ``n`` and ``n_positive`` is NaN (five
            mice cannot reach p < 0.05 in a two-sided exact test). Default
            ``choice_logit_extensions.MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: ``condition, value, term`` (``beta`` or ``gain``), ``n, mean, se,
        n_positive, p_wilcoxon, p_sign`` (as ``choice_logit_extensions.gain_summary``) and, for
        ``beta``, ``re_mean, re_se, re_tau2``: the :func:`random_effects` summary of the
        per-mouse weights and their errors (NaN for the gain).
    """
    rows = []
    for (condition, value), group in table.groupby(["condition", "value"], sort=False):
        summary = ext.gain_summary(group, ["beta", "gain"], min_mice).rename(
            columns={"gain": "term"}
        )
        enough = len(group) >= min_mice
        re = random_effects(group["beta"], group["se"]) if enough else (np.nan,) * 3
        summary[["re_mean", "re_se", "re_tau2"]] = [list(re), [np.nan] * 3]
        rows.append(summary.assign(condition=condition, value=value))
    columns = ["condition", "value", "term", "n", "mean", "se", "n_positive", "p_wilcoxon",
               "p_sign", "re_mean", "re_se", "re_tau2"]  # fmt: skip
    return pd.concat(rows, ignore_index=True)[columns]


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------

CALIBRATION_KEYS = ["mouse", "condition", "value"]


def null_calibration(observed: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """Each observed value weight and gain against its mouse's own null distribution.

    The null of a mouse is the same analysis of agents that never look at the alternatives'
    values, simulated on that mouse's task (``null`` holds one :func:`value_table` row per
    replicate). Whatever bias the analysis has for such agents at the mouse's switch counts is in
    the null, so the calibrated numbers are free of it.

    Args:
        observed (pd.DataFrame): :func:`value_table` of the mice.
        null (pd.DataFrame): :func:`value_table` rows of the simulated replicates, with a
            ``replicate`` column.

    Returns:
        pd.DataFrame: ``mouse, condition, value, term`` (``beta`` or ``gain``), ``observed,
        n_null, null_mean, null_sd, null_low, null_high`` (2.5th and 97.5th percentiles), ``z``
        (``(observed - null_mean) / null_sd``, NaN for a null without spread), ``percentile``
        (fraction of the null below the observation, ties counted half) and ``p_empirical``
        (two-sided, ``(1 + #{|null - null_mean| >= |observed - null_mean|}) / (n_null + 1)``).

    Raises:
        ValueError: If an observation has no null replicate.
    """
    rows = []
    groups = dict(tuple(null.groupby(CALIBRATION_KEYS, sort=False)))
    for _, obs in observed.iterrows():
        key = tuple(obs[k] for k in CALIBRATION_KEYS)
        if key not in groups:
            raise ValueError(
                f"no null replicates for {dict(zip(CALIBRATION_KEYS, key, strict=True))}"
            )
        mine = groups[key]
        for term in ("beta", "gain"):
            draws = mine[term].to_numpy(dtype=float)
            center, spread = draws.mean(), draws.std(ddof=1)
            value = float(obs[term])
            extreme = np.abs(draws - center) >= abs(value - center)
            rows.append(
                dict(
                    zip(CALIBRATION_KEYS, key, strict=True),
                    term=term,
                    observed=value,
                    n_null=len(draws),
                    null_mean=center,
                    null_sd=spread,
                    null_low=float(np.percentile(draws, 2.5)),
                    null_high=float(np.percentile(draws, 97.5)),
                    # a null that never varies gives no scale for a z (the p still holds)
                    z=(value - center) / spread if spread > 0 else np.nan,
                    percentile=float((draws < value).mean() + 0.5 * (draws == value).mean()),
                    p_empirical=(1 + int(extreme.sum())) / (len(draws) + 1),
                )
            )
    return pd.DataFrame(rows)


def calibrated_summary(
    calibration: pd.DataFrame, min_mice: int = ext.MIN_TEST_MICE
) -> pd.DataFrame:
    """Across mice: mean ± se and two-sided tests of the null-calibrated z.

    Args:
        calibration (pd.DataFrame): :func:`null_calibration` of the mice.
        min_mice (int): Fewer mice give NaN mean, se and tests. Default ``MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: ``condition, value, term, n, mean, se, n_positive, p_wilcoxon, p_sign`` of
        the calibrated z (as ``choice_logit_extensions.gain_summary``).
    """
    rows = [
        ext.gain_summary(group, ["z"], min_mice)
        .drop(columns="gain")
        .assign(condition=condition, value=value, term=term)
        for (condition, value, term), group in calibration.groupby(
            ["condition", "value", "term"], sort=False
        )
    ]
    columns = ["condition", "value", "term", "n", "mean", "se", "n_positive", "p_wilcoxon",
               "p_sign"]  # fmt: skip
    return pd.concat(rows, ignore_index=True)[columns]
