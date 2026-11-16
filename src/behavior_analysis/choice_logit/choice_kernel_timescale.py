"""The timescale of a choice kernel: a geometric decay fitted to the choice-lag weights.

A choice kernel ``K <- K + alpha_k (1 - K)`` for the chosen arm and ``K <- (1 - alpha_k) K`` for
the others (:mod:`src.agents.kernel_q_agent`) weights the choice ``k`` trials back by
``beta_k * alpha_k * (1 - alpha_k)**(k - 1)``, a geometric series with decay ``1 - alpha_k``.
The choice-lag weights of the choice logit are the model-free counterpart; fitting
``amplitude * decay**(k - 1)`` to them gives a decay (and timescale ``-1 / ln(decay)`` trials) that
can be set beside the ``alpha_k`` of the agents. A flat kernel has a decay near 1.

Everything here is pure: weights in, tables out.
"""

import numpy as np
import pandas as pd

MIN_LAGS = 4
# Candidate decays; the grid step (0.001) is the resolution of the fitted decay, and the upper
# end caps a flat kernel at a timescale of 1000 trials.
_DECAYS = np.arange(1, 1000) / 1000


def true_kernel(alpha_k: float, beta_k: float, lags) -> np.ndarray:
    """Choice-lag weights implied by a kernel of rate *alpha_k* and weight *beta_k*."""
    lags = np.asarray(lags, dtype=float)
    return beta_k * alpha_k * (1.0 - alpha_k) ** (lags - 1.0)


def _fit_one(lags: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """``(amplitude, decay)`` minimizing the squared error, with a positive amplitude."""
    powers = _DECAYS[:, None] ** (lags[None, :] - 1.0)
    amplitude = powers @ weights / (powers * powers).sum(axis=1)
    sse = ((amplitude[:, None] * powers - weights[None, :]) ** 2).sum(axis=1)
    sse = np.where(amplitude > 0, sse, np.inf)
    best = int(np.argmin(sse))
    if not np.isfinite(sse[best]):
        return np.nan, np.nan
    return float(amplitude[best]), float(_DECAYS[best])


def fit_decay(weights: pd.DataFrame) -> pd.DataFrame:
    """Per mouse, the geometric decay of the choice-lag weights.

    Args:
        weights (pd.DataFrame): ``choice_logit.weights_table`` of one or more mice; only the
            ``choice`` group's lags are used.

    Returns:
        pd.DataFrame: ``mouse, amplitude, decay, tau`` with ``tau = -1 / ln(decay)`` in trials;
        NaN decay and tau for a mouse whose weights have no positive geometric fit.

    Raises:
        ValueError: If a mouse has fewer than ``MIN_LAGS`` choice lags.
    """
    choice = weights[(weights["group"] == "choice") & weights["lag"].notna()]
    rows = []
    for mouse, group in choice.groupby("mouse", sort=False):
        if len(group) < MIN_LAGS:
            raise ValueError(f"mouse {mouse}: {len(group)} choice lags; at least {MIN_LAGS} needed")
        amplitude, decay = _fit_one(group["lag"].to_numpy(float), group["weight"].to_numpy(float))
        tau = -1.0 / np.log(decay) if np.isfinite(decay) else np.nan
        rows.append(dict(mouse=mouse, amplitude=amplitude, decay=decay, tau=tau))
    return pd.DataFrame(rows, columns=["mouse", "amplitude", "decay", "tau"])
