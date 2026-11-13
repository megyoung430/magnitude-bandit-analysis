"""Matching fitted states to true ones and scoring the recovery."""

import numpy as np
import pytest

from src.behavior_analysis.glm_hmm import glm_hmm_recovery as rec

TRUE = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]])


def test_matching_undoes_a_permutation():
    fitted = TRUE[[2, 0, 1]] + 0.1
    assert rec.match_states(TRUE, fitted) == {0: 1, 1: 2, 2: 0}


def test_matching_with_fewer_fitted_states_leaves_true_states_unmatched():
    assert rec.match_states(TRUE, TRUE[[1]] + 0.05) == {1: 0}


def test_matching_with_extra_fitted_states_ignores_the_extras():
    fitted = np.vstack([TRUE[[1, 0]], [[9.0, 9.0, 9.0]]])
    assert rec.match_states(TRUE[:2], fitted) == {0: 1, 1: 0}


def test_state_accuracy_counts_unmatched_true_states_as_wrong():
    true = np.array([0, 0, 1, 1, 2, 2])
    gamma = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0]], float)
    assert rec.state_accuracy(gamma, true, {0: 0, 1: 1}) == pytest.approx(4 / 6)
    assert rec.state_accuracy(gamma, true, {0: 1, 1: 0}) == 0.0


def test_weight_and_dwell_recovery_tables():
    fitted = TRUE[[1, 0, 2]] * 0.9
    matching = {0: 1, 1: 0, 2: 2}
    table = rec.weight_recovery(TRUE, fitted, matching, ("a", "b", "c"))
    assert len(table) == 9
    row = table[(table["state"] == 0) & (table["feature"] == "a")].iloc[0]
    assert (row["true"], row["recovered"]) == (3.0, pytest.approx(2.7))
    summary = rec.weight_summary(table)
    assert summary["correlation"] == pytest.approx(1.0)
    assert summary["mean_abs_error"] == pytest.approx(0.1 * 3 * 3 / 9)
    true_a = np.array([[0.98, 0.02, 0], [0.01, 0.98, 0.01], [0, 0.02, 0.98]])
    fit_a = np.array([[0.97, 0.03, 0], [0.02, 0.96, 0.02], [0, 0.02, 0.98]])
    dwell = rec.dwell_recovery(true_a, fit_a, {0: 0, 1: 1, 2: 2})
    assert dwell["true"].round(1).tolist() == [50.0, 50.0, 50.0]
    assert dwell["recovered"].round(1).tolist() == [33.3, 25.0, 50.0]
