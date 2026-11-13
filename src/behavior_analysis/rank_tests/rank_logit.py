"""A small ridge-penalized logistic regression for the rank tests: fit, errors, held-out fit.

The rank tests fit hundreds of thousands of small logistic models (every simulated replay of every
mouse), so this is a plain Newton solver on the design matrix, not a general GLM. The first column
is the intercept and is never penalized; every other weight has a Gaussian prior with SD
``1 / sqrt(l2)``, the lab's penalty (``choice_logit.DEFAULT_L2``), which also keeps a perfectly
separated feature finite.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.special import expit

from src.behavior_analysis.choice_logit import choice_logit as cl

MAX_ITERATIONS = 100
TOLERANCE = 1e-9


class DegenerateOutcomeError(ValueError):
    """The outcome has a single class, so the logistic cannot be fitted."""


class TooFewGroupsError(ValueError):
    """Cross-validation needs at least two groups (problems) and has fewer."""


class LogitFit(NamedTuple):
    """A fitted logistic regression.

    Attributes:
        beta (np.ndarray): ``(k,)`` weights, the intercept first.
        cov (np.ndarray): ``(k, k)`` inverse Hessian of the penalized negative log likelihood
            (the Laplace approximation to the posterior covariance).
        names (tuple[str, ...]): Name of each weight.
        nll (float): Negative log likelihood of the fitted trials (no penalty).
        n (int): Trials fitted.
    """

    beta: np.ndarray
    cov: np.ndarray
    names: tuple
    nll: float
    n: int

    @property
    def se(self) -> np.ndarray:
        """Standard error of each weight."""
        return np.sqrt(np.diag(self.cov))

    def weight(self, name: str) -> float:
        """The weight called *name*."""
        return float(self.beta[self.names.index(name)])

    def weight_se(self, name: str) -> float:
        """The standard error of the weight called *name*."""
        return float(self.se[self.names.index(name)])


class HeldOut(NamedTuple):
    """Leave-one-group-out log likelihood: its sum over the held-out trials and their count."""

    log_likelihood: float
    n: int

    @property
    def nats_per_trial(self) -> float:
        """Negative mean held-out log likelihood."""
        return -self.log_likelihood / self.n


def _check(X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> None:
    if X.ndim != 2 or X.shape[1] != len(names) or y.shape != (len(X),):
        raise ValueError(f"X {X.shape} and y {y.shape} do not match {len(names)} names")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("the design or the outcome holds NaN")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("the outcome must be 0 or 1")
    if y.min() == y.max():
        raise DegenerateOutcomeError("the outcome has only one class")


def log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Summed log likelihood of outcomes *y* under weights *beta*."""
    z = X @ beta
    return float(-(np.logaddexp(0, z) - y * z).sum())


def fit_logit(
    X: np.ndarray, y: np.ndarray, names: Sequence[str], l2: float = cl.DEFAULT_L2
) -> LogitFit:
    """Fit ``P(y = 1) = sigmoid(X @ beta)`` by Newton's method with a ridge on all but the first.

    Args:
        X (np.ndarray): ``(n, k)`` design, the intercept column first.
        y (np.ndarray): ``(n,)`` outcome, 0 or 1, with both classes present.
        names (Sequence[str]): Name of each column.
        l2 (float): Penalty (``>= 0``). Default ``choice_logit.DEFAULT_L2``.

    Returns:
        LogitFit: Weights, Laplace covariance and the unpenalized negative log likelihood.

    Raises:
        ValueError: For mismatched shapes, NaN, a non-binary or one-class outcome, or ``l2 < 0``.
        RuntimeError: If Newton's method does not converge.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    _check(X, y, names)
    if l2 < 0:
        raise ValueError(f"l2 must be non-negative, got {l2}")
    penalty = np.full(X.shape[1], l2)
    penalty[0] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(MAX_ITERATIONS):
        p = expit(X @ beta)
        gradient = X.T @ (p - y) + penalty * beta
        hessian = (X * (p * (1 - p))[:, None]).T @ X + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta = beta - step
        if np.abs(step).max() < TOLERANCE:
            break
    else:
        raise RuntimeError("the logistic fit did not converge")
    p = expit(X @ beta)
    hessian = (X * (p * (1 - p))[:, None]).T @ X + np.diag(penalty)
    return LogitFit(beta, np.linalg.inv(hessian), tuple(names), -log_likelihood(X, y, beta), len(y))


def contrast_se(fit: LogitFit, coefficients: dict[str, float]) -> float:
    """Standard error of ``sum c_name * beta_name`` from the Laplace covariance."""
    c = np.zeros(len(fit.names))
    for name, value in coefficients.items():
        c[fit.names.index(name)] = value
    return float(np.sqrt(c @ fit.cov @ c))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    l2: float = cl.DEFAULT_L2,
) -> HeldOut:
    """Leave-one-group-out held-out log likelihood (the lab's folds are problems).

    A fold whose training trials hold a single class cannot be fitted; it raises (the caller
    decides what a mouse with such a problem means), it is never skipped silently.

    Args:
        X (np.ndarray): ``(n, k)`` design, the intercept first.
        y (np.ndarray): ``(n,)`` outcome, 0 or 1.
        groups (np.ndarray): ``(n,)`` fold label of each trial.
        names (Sequence[str]): Name of each column.
        l2 (float): Penalty. Default ``choice_logit.DEFAULT_L2``.

    Returns:
        HeldOut: Summed held-out log likelihood and the number of trials scored.

    Raises:
        TooFewGroupsError: With fewer than two groups.
        ValueError: As :func:`fit_logit`.
    """
    labels = np.unique(groups)
    if len(labels) < 2:
        raise TooFewGroupsError("cross-validation needs at least two groups")
    total = 0.0
    for label in labels:
        test = groups == label
        fit = fit_logit(X[~test], y[~test], names, l2)
        total += log_likelihood(X[test], y[test], fit.beta)
    return HeldOut(total, len(y))
