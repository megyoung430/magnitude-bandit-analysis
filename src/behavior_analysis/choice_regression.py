"""Which port a mouse chooses, as a conditional logit over the three offered ports.

On every trial each offered port ``j`` gets a utility ``u_j = x_j . w`` and the port is chosen
with probability ``exp(u_j) / sum_k exp(u_k)``. One weight vector ``w`` is fitted per mouse, and
the mice are the replicates of the group statistics. The features of a port (``FEATURES``):

- ``last_value``: the magnitude the mouse received the last time it chose this port in the
  problem (0 if it never did, or if that visit paid nothing);
- ``last_zero``: 1 if that last visit paid nothing;
- ``unseen``: 1 if the mouse has not chosen this port yet in the problem;
- ``stay``: 1 if this port was chosen on the previous trial of the same session;
- ``plus_port``: 1 if the port is in the same row or column as the initiation port (see
  :func:`src.behavior_analysis.port_preference.port_category`).

Because ``last_zero`` and ``unseen`` take up the empty and never-visited ports, ``last_value``
measures sensitivity to magnitude among ports last seen paying 1-4, separately from avoiding
the empty port. ``stay`` absorbs perseveration, so a positive ``last_value`` is not explained
by the mouse repeating its last choice. ``plus_port`` absorbs the spatial bias.

What the mouse last saw at a port carries over between sessions of one problem and resets at
a new problem, as for the believed values elsewhere in the repo. ``stay`` resets at a session
start and after a dropped trial.

All functions are pure: a trial table in, arrays and DataFrames out.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.behavior_analysis.port_preference import port_category

FEATURES = ("last_value", "last_zero", "unseen", "stay", "plus_port")
# A weak Gaussian prior on every weight (SD in logit units) keeps a mouse's fit finite when a
# feature hardly varies within its trials, e.g. `unseen`, which is 1 only until every port has
# been visited. Its strength does not grow with the number of trials, so with thousands of
# trials it moves the weights negligibly.
PRIOR_SD = 3.0
# Condition number above which the curvature is treated as singular (standard errors NaN).
MAX_CONDITION = 1e12


class DesignArrays(NamedTuple):
    """Arm-level design of one set of trials.

    Attributes:
        X (np.ndarray): ``(n_trials, 3, n_features)`` features of the offered ports, ports in
            sorted order within each trial.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen port along axis 1 of ``X``.
        mouse (np.ndarray): ``(n_trials,)`` mouse of each trial.
    """

    X: np.ndarray
    chosen: np.ndarray
    mouse: np.ndarray


def _problem_design(trials: pd.DataFrame) -> tuple[list, list]:
    """Features and chosen index of one mouse's problem; *trials* in trial order."""
    last_seen: dict[str, float] = {}
    previous_choice = None
    rows, chosen = [], []
    for choice, magnitudes, initiation, trial_in_session, after_gap in zip(
        trials["choice"],
        trials["magnitudes"],
        trials["initiation"],
        trials["trial_in_session"],
        trials["after_gap"],
        strict=True,
    ):
        ports = sorted(magnitudes)
        if len(ports) != 3:
            raise ValueError(f"expected three offered ports, got {ports}")
        if trial_in_session == 0 or after_gap:
            previous_choice = None
        rows.append(
            [
                [
                    last_seen.get(port, 0.0),
                    float(last_seen.get(port) == 0),
                    float(port not in last_seen),
                    float(port == previous_choice),
                    float(port_category(port, initiation) == "plus"),
                ]
                for port in ports
            ]
        )
        chosen.append(ports.index(choice))
        last_seen[choice] = float(magnitudes[choice])
        previous_choice = choice
    return rows, chosen


def choice_design(trial_table: pd.DataFrame) -> DesignArrays:
    """Arm-level features (``FEATURES``) of every trial in *trial_table*.

    Args:
        trial_table (pd.DataFrame): Table of
            :func:`src.behavior_analysis.drifting_trial_table.drifting_trial_table`.

    Returns:
        DesignArrays: Features, chosen port and mouse per trial.

    Raises:
        ValueError: If a trial does not offer exactly three ports.
    """
    X, chosen, mice = [], [], []
    for (mouse, _), trials in trial_table.groupby(["mouse", "problem"], sort=False):
        rows, picks = _problem_design(trials.sort_values("trial"))
        X.extend(rows)
        chosen.extend(picks)
        mice.extend([mouse] * len(picks))
    n_features = len(FEATURES)
    return DesignArrays(
        np.asarray(X, dtype=float).reshape(-1, 3, n_features),
        np.asarray(chosen, dtype=int),
        np.asarray(mice, dtype=object),
    )


def _loss_and_gradient(w: np.ndarray, X: np.ndarray, chosen: np.ndarray, prior_sd: float):
    """Negative log posterior per trial and its gradient (the per-trial scale helps L-BFGS)."""
    n = len(chosen)
    utilities = X @ w
    log_p = utilities - logsumexp(utilities, axis=1, keepdims=True)
    rows = np.arange(n)
    p = np.exp(log_p)
    expected = np.einsum("tj,tjk->tk", p, X)
    prior = 1.0 / (prior_sd**2 * n)
    gradient = -(X[rows, chosen] - expected).mean(axis=0) + prior * w
    return -log_p[rows, chosen].mean() + 0.5 * prior * w @ w, gradient


def _hessian(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Hessian of the summed negative log likelihood (without the prior)."""
    utilities = X @ w
    p = np.exp(utilities - logsumexp(utilities, axis=1, keepdims=True))
    mean_x = np.einsum("tj,tjk->tk", p, X)
    second = np.einsum("tj,tjk,tjl->kl", p, X, X)
    return second - mean_x.T @ mean_x


def _standard_errors(curvature: np.ndarray) -> np.ndarray:
    """Square roots of the diagonal of the inverse curvature; NaN where it is not usable."""
    n = curvature.shape[0]
    if not np.all(np.isfinite(curvature)) or np.linalg.cond(curvature) > MAX_CONDITION:
        return np.full(n, np.nan)
    variances = np.diag(np.linalg.inv(curvature))
    return np.where(variances > 0, np.sqrt(np.abs(variances)), np.nan)


class LogitFit(NamedTuple):
    """One conditional-logit fit.

    Attributes:
        weights (np.ndarray): ``(n_features,)`` fitted weights (posterior mode).
        standard_errors (np.ndarray): ``(n_features,)`` from the inverse curvature of the
            negative log posterior, consistent with the weights; NaN where the curvature is
            singular or a variance is not positive.
        n_trials (int): Trials fitted.
        log_likelihood (float): Summed log likelihood at the fit (without the prior).
    """

    weights: np.ndarray
    standard_errors: np.ndarray
    n_trials: int
    log_likelihood: float


def fit_conditional_logit(
    X: np.ndarray, chosen: np.ndarray, prior_sd: float | None = PRIOR_SD
) -> LogitFit:
    """Weights of the conditional logit under a weak Gaussian prior (posterior mode).

    Args:
        X (np.ndarray): ``(n_trials, n_options, n_features)`` features.
        chosen (np.ndarray): ``(n_trials,)`` index of the chosen option.
        prior_sd (float | None): Prior SD of each weight, in logit units. Default
            ``PRIOR_SD``; ``None`` for plain maximum likelihood.

    Returns:
        LogitFit: Weights, standard errors, number of trials and log likelihood.

    Raises:
        ValueError: If there are no trials, or the optimiser does not converge.
    """
    if len(chosen) == 0:
        raise ValueError("no trials to fit")
    sd = np.inf if prior_sd is None else float(prior_sd)
    w0 = np.zeros(X.shape[2])
    result = minimize(_loss_and_gradient, w0, args=(X, chosen, sd), jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"conditional logit did not converge: {result.message}")
    curvature = _hessian(result.x, X) + np.eye(X.shape[2]) / sd**2
    utilities = X @ result.x
    log_p = utilities - logsumexp(utilities, axis=1, keepdims=True)
    return LogitFit(
        result.x,
        _standard_errors(curvature),
        len(chosen),
        float(log_p[np.arange(len(chosen)), chosen].sum()),
    )


def fit_by_mouse(design: DesignArrays, min_trials: int = 200) -> pd.DataFrame:
    """Fit the conditional logit to each mouse separately.

    Args:
        design (DesignArrays): Output of :func:`choice_design`.
        min_trials (int): Mice with fewer trials are left out (too few for stable weights).
            Default 200.

    Returns:
        pd.DataFrame: One row per mouse and feature with columns ``mouse, feature, weight,
        standard_error, n_trials``.
    """
    rows = []
    for mouse in sorted(set(design.mouse)):
        keep = design.mouse == mouse
        if keep.sum() < min_trials:
            continue
        fit = fit_conditional_logit(design.X[keep], design.chosen[keep])
        rows.extend(
            dict(
                mouse=mouse,
                feature=feature,
                weight=fit.weights[i],
                standard_error=fit.standard_errors[i],
                n_trials=fit.n_trials,
            )
            for i, feature in enumerate(FEATURES)
        )
    return pd.DataFrame(rows, columns=["mouse", "feature", "weight", "standard_error", "n_trials"])


def group_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """Mean weight across mice, its standard error and a one-sample t-test against 0.

    Args:
        weights (pd.DataFrame): Output of :func:`fit_by_mouse`.

    Returns:
        pd.DataFrame: One row per feature (``FEATURES`` order) with columns ``feature, mean,
        standard_error, t, p_value, n_mice``. ``t`` and ``p_value`` are NaN with fewer than two
        mice.
    """
    rows = []
    for feature in FEATURES:
        values = weights.loc[weights["feature"] == feature, "weight"].to_numpy(dtype=float)
        n = len(values)
        if n >= 2:
            test = stats.ttest_1samp(values, 0.0)
            se, t, p = values.std(ddof=1) / np.sqrt(n), float(test.statistic), float(test.pvalue)
        else:
            se = t = p = np.nan
        rows.append(
            dict(
                feature=feature,
                mean=values.mean() if n else np.nan,
                standard_error=se,
                t=t,
                p_value=p,
                n_mice=n,
            )
        )
    return pd.DataFrame(rows)
