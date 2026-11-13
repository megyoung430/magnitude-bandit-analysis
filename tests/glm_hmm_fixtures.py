"""Synthetic sessions drawn from a known GLM-HMM, for the GLM-HMM tests."""

import numpy as np
from scipy.special import logsumexp

from src.behavior_analysis.glm_hmm import glm_hmm as hmm

TWO_STATES = np.array([[2.5, 0.0], [0.0, 2.5]])


def sticky(k: int, stay: float = 0.95) -> np.ndarray:
    transition = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(transition, stay)
    return transition


def make_sequences(rng, lengths, weights=TWO_STATES, transition=None):
    """Sessions drawn from a known GLM-HMM; each starts in a uniformly drawn state."""
    k, n_features = weights.shape
    transition = sticky(k) if transition is None else transition
    n_sessions, n_steps = len(lengths), max(lengths)
    X = rng.normal(size=(n_sessions, n_steps, 3, n_features))
    chosen = np.zeros((n_sessions, n_steps), dtype=int)
    states = np.full((n_sessions, n_steps), -1)
    mask = np.arange(n_steps)[None, :] < np.asarray(lengths)[:, None]
    for s, length in enumerate(lengths):
        z = rng.integers(k)
        for t in range(length):
            if t:
                z = rng.choice(k, p=transition[z])
            logits = X[s, t] @ weights[z]
            chosen[s, t] = rng.choice(3, p=np.exp(logits - logsumexp(logits)))
            states[s, t] = z
    return hmm.Sequences(X, chosen, mask, mask.copy()), states
