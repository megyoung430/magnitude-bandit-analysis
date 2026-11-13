"""Held-out model selection over the number of states, on synthetic sessions."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.glm_hmm import glm_hmm as hmm
from src.behavior_analysis.glm_hmm import glm_hmm_selection as sel
from tests.glm_hmm_fixtures import make_sequences

FAST = dict(n_restarts=3, seed=0, max_iterations=60)


def folds_of(n_sessions: int, n_folds: int = 5) -> np.ndarray:
    return np.arange(n_sessions) % n_folds


def scores_table(nats_by_k: dict, noise: float = 0.01, n_folds: int = 5) -> pd.DataFrame:
    """Per-fold scores of 100 trials whose nats per trial are *nats_by_k* plus fold noise.

    A fold's difficulty is shared by every model; each model also has its own noise.
    """
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(n_folds):
        shared = rng.normal(scale=0.05)
        for k, nats in nats_by_k.items():
            total = nats + shared + rng.normal(scale=noise)
            rows.append(dict(fold=fold, n_states=k, n_scored=100, log_likelihood=-100 * total))
    return pd.DataFrame(rows)


def test_table_gain_is_the_nats_saved_over_one_state():
    scores = scores_table({1: 0.70, 2: 0.65, 3: 0.66}, noise=0.0)
    table = sel.selection_table(scores).set_index("n_states")
    assert table["nats"].diff().tolist()[1:] == pytest.approx([-0.05, 0.01])
    assert table.loc[1, "gain"] == 0
    assert table.loc[2, "gain"] == pytest.approx(0.05)
    assert table.loc[3, "gain"] == pytest.approx(0.04)
    assert table.loc[2, "gain_se"] == pytest.approx(0.0, abs=1e-12)
    assert sel.best_gain(scores) == pytest.approx(0.05)


def test_one_standard_error_rule_prefers_fewer_states_unless_the_gain_is_clear():
    # three states are no clearer than two (0.002 nats against fold noise), one is far worse
    assert sel.select_n_states(scores_table({1: 0.70, 2: 0.60, 3: 0.598})) == 2
    assert sel.select_n_states(scores_table({1: 0.700, 2: 0.699, 3: 0.701})) == 1
    noisy = scores_table({1: 0.70, 2: 0.65}, noise=0.0)
    noisy.loc[noisy["n_states"] == 2, "log_likelihood"] += np.array([-20, 20, -20, 20, 0])
    assert sel.select_n_states(noisy) == 1  # a gain that swings from fold to fold is not clear
    assert sel.select_n_states(scores_table({1: 0.70, 2: 0.60})) == 2


def test_best_gain_is_negative_when_more_states_always_hurt():
    scores = scores_table({1: 0.70, 2: 0.72, 3: 0.75}, noise=0.0)
    assert sel.best_gain(scores) < 0
    assert sel.select_n_states(scores) == 1


def test_cross_validation_finds_the_states_of_switching_sessions():
    seqs, _ = make_sequences(np.random.default_rng(20), [60] * 20)
    scores = sel.cross_validate(seqs, folds_of(20), (1, 2, 3), hmm.Prior(), **FAST)
    assert set(scores["n_states"]) == {1, 2, 3}
    assert scores.groupby("n_states")["n_scored"].sum().tolist() == [1200] * 3
    assert sel.select_n_states(scores) == 2
    assert sel.best_gain(scores) > 0.05


def test_cross_validation_keeps_one_state_for_stationary_sessions():
    seqs, _ = make_sequences(
        np.random.default_rng(21),
        [60] * 20,
        weights=np.array([[1.5, -1.0]]),
        transition=np.ones((1, 1)),
    )
    scores = sel.cross_validate(seqs, folds_of(20), (1, 2, 3), hmm.Prior(), **FAST)
    assert sel.select_n_states(scores) == 1
    assert sel.best_gain(scores) < 0.02


def test_only_scored_trials_count():
    seqs, _ = make_sequences(np.random.default_rng(22), [30] * 10)
    scored = seqs.mask.copy()
    scored[:, 0] = False
    scores = sel.cross_validate(
        seqs._replace(scored=scored), folds_of(10), (1,), hmm.Prior(), **FAST
    )
    assert scores["n_scored"].sum() == 290


def test_validation():
    seqs, _ = make_sequences(np.random.default_rng(23), [20] * 6)
    with pytest.raises(ValueError, match="one state"):
        sel.cross_validate(seqs, folds_of(6, 3), (2, 3), hmm.Prior(), **FAST)
    with pytest.raises(ValueError, match="folds"):
        sel.cross_validate(seqs, np.zeros(5, int), (1,), hmm.Prior(), **FAST)
    with pytest.raises(ValueError, match="fold"):
        sel.cross_validate(seqs, np.array([0, 0, 0, 2, 2, 2]), (1,), hmm.Prior(), **FAST)
    with pytest.raises(ValueError, match="one state"):
        sel.selection_table(scores_table({2: 0.6, 3: 0.6}))
