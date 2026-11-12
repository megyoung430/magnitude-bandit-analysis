"""Is the flat choice-history kernel a mixture of states, or does it persist within them?

The mice's choice logit (:mod:`src.behavior_analysis.choice_logit`) has a flat kernel: the weights
of the choices 1 to 5 trials back barely decay. A mixture of strategy states with sharp kernels
(one repeats the last choice, another does not) can look flat when pooled. The test fits the same
logit (five reward lags, five choice lags, tower biases) to the trials a GLM-HMM assigns to each
state (posterior above a threshold) and reads the choice-lag weights.

**Flatness** is the mean weight of lags 3 to 5 over the weight of lag 1: 1 for a flat kernel,
0 for a kernel that only reaches the last choice (NaN when lag 1 is not positive).
"""

import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import glm_hmm_features as feats

DEFAULT_THRESHOLD = 0.8
MIN_TRIALS = 200
FLAT_LAGS = (3, 4, 5)


def state_rows(data: feats.SessionData, gamma: np.ndarray, threshold: float) -> np.ndarray:
    """``(trials, states)`` bool: the posterior of the state on the trial exceeds *threshold*.

    Rows follow ``data.trials``.
    """
    return gamma[data.trials["session"].to_numpy(), data.trials["step"].to_numpy()] > threshold


def flatness(weights: np.ndarray) -> float:
    """Mean of the lag 3-5 weights over the lag 1 weight; NaN unless lag 1 is positive."""
    weights = np.asarray(weights, dtype=float)
    first = weights[0]
    if not first > 0:
        return float("nan")
    return float(weights[[lag - 1 for lag in FLAT_LAGS]].mean() / first)


def _check_alignment(design: cl.LogitDesign, data: feats.SessionData) -> None:
    meta, index = design.meta, data.trials
    same = (
        len(meta) == len(index)
        and np.array_equal(meta["trial"].to_numpy(), index["trial"].to_numpy())
        and np.array_equal(
            meta["problem"].map(str).to_numpy(), index["problem"].map(str).to_numpy()
        )
    )
    if not same:
        raise ValueError("the trial table and the session data do not align row for row")


def _lag_weights(design, rows: np.ndarray, n_lags: int, min_trials: int, l2: float) -> np.ndarray:
    if rows.sum() < min_trials:
        return np.full(n_lags, np.nan)
    fit = cl.fit_logit(design.X[rows], design.chosen[rows], design.feature_names, l2)
    names = [f"choice_lag{lag}" for lag in range(1, n_lags + 1)]
    return np.array([fit.beta[design.feature_names.index(n)] for n in names])


def kernel_by_state(
    trials: pd.DataFrame,
    data: feats.SessionData,
    gamma: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    n_lags: int = cl.DEFAULT_LAGS,
    min_trials: int = MIN_TRIALS,
    l2: float = cl.DEFAULT_L2,
) -> pd.DataFrame:
    """The choice-lag weights of all trials and of the trials of each state.

    Args:
        trials (pd.DataFrame): The mouse's drifting trial table (the one *data* was made from).
        data (feats.SessionData): The mouse's sessions.
        gamma (np.ndarray): ``(sessions, steps, states)`` posteriors of a GLM-HMM on *data*.
        threshold (float): Posterior above which a trial belongs to a state. Default 0.8.
        n_lags (int): Choice lags. Default ``choice_logit.DEFAULT_LAGS``.
        min_trials (int): Fewer trials than this give NaN weights. Default ``MIN_TRIALS``.
        l2 (float): Ridge penalty of the logit. Default ``choice_logit.DEFAULT_L2``.

    Returns:
        pd.DataFrame: ``state`` (``"all"`` or the state number), ``lag``, ``weight`` and
        ``n_trials`` (trials of the group).

    Raises:
        ValueError: If the rows of *trials* and *data* do not line up.
    """
    design = cl.design_matrix(trials, n_lags)
    _check_alignment(design, data)
    groups = {"all": np.ones(len(data.trials), dtype=bool)}
    by_state = state_rows(data, gamma, threshold)
    groups |= {k: by_state[:, k] for k in range(by_state.shape[1])}
    rows = []
    for state, mask in groups.items():
        weights = _lag_weights(design, mask, n_lags, min_trials, l2)
        rows += [
            dict(state=state, lag=lag, weight=w, n_trials=int(mask.sum()))
            for lag, w in enumerate(weights, start=1)
        ]
    return pd.DataFrame(rows)
