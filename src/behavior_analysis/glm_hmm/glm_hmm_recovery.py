"""Recovery of a known GLM-HMM: matched states, state accuracy, weights and dwell times.

Fitted states carry arbitrary labels (they are sorted by one weight, which need not agree with the
truth), so every comparison first matches each true state to the fitted state whose weights are
closest, one to one (Hungarian assignment on the Euclidean distance of the weight vectors). With
fewer fitted than true states some true states stay unmatched, and with more the extra fitted
states are ignored.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.behavior_analysis.glm_hmm import glm_hmm as hmm


def match_states(true_weights: np.ndarray, fitted_weights: np.ndarray) -> dict[int, int]:
    """``{true state: fitted state}`` minimizing the total weight distance.

    Args:
        true_weights (np.ndarray): ``(true states, features)``.
        fitted_weights (np.ndarray): ``(fitted states, features)``.
    """
    rows, cols = linear_sum_assignment(cdist(true_weights, fitted_weights))
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True)}


def state_accuracy(gamma: np.ndarray, true_states: np.ndarray, matching: dict[int, int]) -> float:
    """Share of trials whose most probable fitted state is the matched true state.

    Trials of a true state with no match count as wrong.

    Args:
        gamma (np.ndarray): ``(trials, fitted states)`` posterior probabilities.
        true_states (np.ndarray): ``(trials,)`` the generating state of every trial.
        matching (dict[int, int]): From :func:`match_states`.
    """
    decoded = gamma.argmax(axis=1)
    expected = np.array([matching.get(int(z), -1) for z in true_states])
    return float((decoded == expected).mean())


def weight_recovery(
    true_weights: np.ndarray,
    fitted_weights: np.ndarray,
    matching: dict[int, int],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """``state, fitted_state, feature, true, recovered`` for every matched state and feature."""
    rows = [
        dict(state=t, fitted_state=f, feature=name, true=true_weights[t, i],
             recovered=fitted_weights[f, i])
        for t, f in matching.items()
        for i, name in enumerate(feature_names)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def weight_summary(table: pd.DataFrame) -> dict:
    """Correlation of recovered with true weights and their mean absolute error."""
    correlation = (
        float(np.corrcoef(table["true"], table["recovered"])[0, 1])
        if len(table) > 1 and table["true"].std() > 0 and table["recovered"].std() > 0
        else float("nan")
    )
    return dict(
        correlation=correlation,
        mean_abs_error=float((table["true"] - table["recovered"]).abs().mean()),
    )


def dwell_recovery(
    true_transition: np.ndarray, fitted_transition: np.ndarray, matching: dict[int, int]
) -> pd.DataFrame:
    """``state, true, recovered`` mean dwell times (trials) of every matched state."""
    true_dwell, fitted_dwell = hmm.dwell_times(true_transition), hmm.dwell_times(fitted_transition)
    return pd.DataFrame(
        [dict(state=t, true=true_dwell[t], recovered=fitted_dwell[f]) for t, f in matching.items()]
    )
