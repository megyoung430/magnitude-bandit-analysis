"""How many states? Held-out scores of GLM-HMMs with one to a few states.

Whole sessions are held out (:func:`src.behavior_analysis.glm_hmm_features.assign_folds`): each
fold is scored by models fitted on the other folds, with the forward-filter predictive log
probability of the scored trials (:func:`src.behavior_analysis.glm_hmm.heldout_log_likelihood`).
Nats per trial are the negative summed log probability over the scored trials of all folds
divided by their number; the *gain* of ``K`` states is the nats of one state minus the nats of
``K`` states, positive when states help.

**Selection rule (one standard error).** The state count with the lowest held-out nats is not
always chosen: the smallest count whose nats are within one standard error (of the per-fold paired
difference) of the best is. Extra states that do not clearly help are not credited, which is what
a model that should choose one state on stationary data needs.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.behavior_analysis import glm_hmm as hmm
from src.behavior_analysis import glm_hmm_features as feats

REFERENCE = 1


def _check_folds(seqs: hmm.Sequences, folds: np.ndarray, n_states: Sequence[int]) -> np.ndarray:
    folds = np.asarray(folds)
    if folds.shape != (len(seqs.chosen),):
        raise ValueError(f"folds must hold one entry per session ({len(seqs.chosen)})")
    labels = np.unique(folds)
    if len(labels) < 2:
        raise ValueError("at least two folds are needed")
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError(f"fold numbers must run 0..n-1 without gaps, got {labels.tolist()}")
    if REFERENCE not in n_states:
        raise ValueError("n_states must include one state, the reference of every gain")
    return labels


def cross_validate(
    seqs: hmm.Sequences,
    folds: np.ndarray,
    n_states: Sequence[int],
    prior: hmm.Prior,
    n_restarts: int,
    seed: int,
    **fit_kwargs,
) -> pd.DataFrame:
    """Held-out score of every state count in every fold.

    Args:
        seqs (hmm.Sequences): All of one mouse's sessions.
        folds (np.ndarray): ``(sessions,)`` fold numbers ``0 .. n_folds - 1``.
        n_states (Sequence[int]): State counts to fit; must include 1.
        prior (hmm.Prior): The penalties.
        n_restarts (int): Starts per fit.
        seed (int): Base seed; every (fold, count) has its own stream.
        **fit_kwargs: Passed to :func:`src.behavior_analysis.glm_hmm.fit_restarts`.

    Returns:
        pd.DataFrame: ``fold, n_states, log_likelihood`` (held-out, scored trials), ``n_scored``,
        ``n_restarts, n_converged``.

    Raises:
        ValueError: For malformed folds or *n_states* without 1.
        RuntimeError: If a fit has no converged start.
    """
    labels = _check_folds(seqs, folds, n_states)
    rows = []
    for fold in labels:
        train = feats.subset(seqs, np.flatnonzero(folds != fold))
        test = feats.subset(seqs, np.flatnonzero(folds == fold))
        for k in n_states:
            fit = hmm.fit_restarts(
                train, k, prior, n_restarts, seed * 10_000 + int(fold) * 100 + k, **fit_kwargs
            )
            rows.append(
                dict(
                    fold=int(fold),
                    n_states=k,
                    log_likelihood=hmm.heldout_log_likelihood(fit.params, test),
                    n_scored=int(test.scored.sum()),
                    n_restarts=fit.n_restarts,
                    n_converged=fit.n_converged,
                )
            )
    return pd.DataFrame(rows)


def _fold_nats(scores: pd.DataFrame) -> pd.DataFrame:
    """Nats per scored trial, folds in rows and state counts in columns."""
    per_fold = scores.assign(nats=-scores["log_likelihood"] / scores["n_scored"])
    return per_fold.pivot(index="fold", columns="n_states", values="nats")


def selection_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Pooled held-out nats per trial and the gain over one state, per state count.

    Args:
        scores (pd.DataFrame): :func:`cross_validate` output.

    Returns:
        pd.DataFrame: ``n_states, nats, gain`` (nats of one state minus these) and ``gain_se``
        (standard error of the per-fold gain across folds; 0 for one state).

    Raises:
        ValueError: If one state is missing.
    """
    pooled = scores.groupby("n_states").agg(
        log_likelihood=("log_likelihood", "sum"), n_scored=("n_scored", "sum")
    )
    if REFERENCE not in pooled.index:
        raise ValueError("the scores need one state, the reference of every gain")
    pooled["nats"] = -pooled["log_likelihood"] / pooled["n_scored"]
    fold_nats = _fold_nats(scores)
    gain_by_fold = fold_nats[[REFERENCE]].to_numpy() - fold_nats.to_numpy()
    n_folds = len(fold_nats)
    pooled["gain"] = pooled.loc[REFERENCE, "nats"] - pooled["nats"]
    pooled["gain_se"] = gain_by_fold.std(axis=0, ddof=1) / np.sqrt(n_folds)
    return pooled.reset_index()[["n_states", "nats", "gain", "gain_se"]]


def best_gain(scores: pd.DataFrame) -> float:
    """The largest gain over one state among the counts above one (NaN if there are none)."""
    table = selection_table(scores)
    above = table[table["n_states"] > REFERENCE]
    return float(above["gain"].max()) if len(above) else float("nan")


def select_n_states(scores: pd.DataFrame) -> int:
    """The smallest state count within one standard error of the best (see the module text).

    Args:
        scores (pd.DataFrame): :func:`cross_validate` output.

    Returns:
        int: The selected number of states.
    """
    fold_nats = _fold_nats(scores)
    counts = list(fold_nats.columns)
    best = fold_nats.mean().idxmin()
    for k in counts:
        diff = (fold_nats[k] - fold_nats[best]).to_numpy()
        if k == best:
            return int(k)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        if diff.mean() <= se:
            return int(k)
    return int(best)
