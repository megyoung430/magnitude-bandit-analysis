"""The GLM-HMM: forward-backward, EM, held-out scoring and the restarts, on synthetic data."""

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.behavior_analysis import glm_hmm as hmm

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


def brute_force(seqs, params):
    """Marginal log likelihood, posteriors and summed transitions by enumerating every path."""
    k = len(params.initial)
    log_emit = hmm.emission_log_probs(params.weights, seqs.X, seqs.chosen)
    total, gamma, xi = 0.0, np.zeros(log_emit.shape), np.zeros((k, k))
    for s in range(len(seqs.chosen)):
        n = int(seqs.mask[s].sum())
        paths = list(itertools.product(range(k), repeat=n))
        logs = np.array(
            [
                np.log(params.initial[p[0]])
                + log_emit[s, 0, p[0]]
                + sum(
                    np.log(params.transition[p[t - 1], p[t]]) + log_emit[s, t, p[t]]
                    for t in range(1, n)
                )
                for p in paths
            ]
        )
        total += logsumexp(logs)
        for w, path in zip(np.exp(logs - logsumexp(logs)), paths, strict=True):
            for t, z in enumerate(path):
                gamma[s, t, z] += w
                if t:
                    xi[path[t - 1], z] += w
    return total, gamma, xi


def toy_params(k=2, n_features=2, seed=0):
    rng = np.random.default_rng(seed)
    return hmm.Params(
        rng.normal(size=(k, n_features)),
        rng.dirichlet(np.ones(k)),
        rng.dirichlet(np.ones(k) * 2, size=k),
    )


def run_forward_backward(seqs, params):
    emit = hmm.emission_log_probs(params.weights, seqs.X, seqs.chosen)
    return hmm.forward_backward(emit, params, seqs.mask)


def test_forward_backward_matches_enumeration():
    seqs, _ = make_sequences(np.random.default_rng(1), [4, 3, 5])
    params = toy_params(2)
    result = run_forward_backward(seqs, params)
    total, gamma, xi = brute_force(seqs, params)
    assert result.log_likelihood.sum() == pytest.approx(total)
    np.testing.assert_allclose(result.gamma[seqs.mask], gamma[seqs.mask], atol=1e-10)
    np.testing.assert_allclose(result.xi, xi, atol=1e-10)


def test_forward_backward_three_states():
    weights = np.array([[2.0, 0], [0, 2.0], [-2.0, 1.0]])
    seqs, _ = make_sequences(np.random.default_rng(2), [4, 4], weights=weights)
    params = toy_params(3, seed=3)
    result = run_forward_backward(seqs, params)
    total, gamma, xi = brute_force(seqs, params)
    assert result.log_likelihood.sum() == pytest.approx(total)
    np.testing.assert_allclose(result.gamma[seqs.mask], gamma[seqs.mask], atol=1e-10)
    np.testing.assert_allclose(result.xi, xi, atol=1e-10)


def test_sessions_are_independent_and_padding_does_not_matter():
    seqs, _ = make_sequences(np.random.default_rng(3), [6, 4])
    params = toy_params(2)
    both = run_forward_backward(seqs, params).log_likelihood
    first = hmm.Sequences(*(a[:1] for a in seqs))
    second = hmm.Sequences(*(a[1:2, :4] for a in seqs))
    np.testing.assert_allclose(
        both,
        [run_forward_backward(s, params).log_likelihood[0] for s in (first, second)],
    )
    swapped = hmm.Sequences(*(a[::-1] for a in seqs))
    np.testing.assert_allclose(run_forward_backward(swapped, params).log_likelihood, both[::-1])


def test_state_resets_to_the_initial_distribution_at_a_session_start():
    # with no transitions each session stays in its first state, drawn from `initial` alone: the
    # second session's likelihood must not depend on the first session's states
    seqs, _ = make_sequences(np.random.default_rng(4), [3, 3])
    params = hmm.Params(np.array([[3.0, 0.0], [0.0, 3.0]]), np.array([0.2, 0.8]), np.eye(2))
    emit = hmm.emission_log_probs(params.weights, seqs.X, seqs.chosen)
    result = hmm.forward_backward(emit, params, seqs.mask)
    for s in range(2):
        per_state = [emit[s, :, z].sum() + np.log(params.initial[z]) for z in range(2)]
        assert result.log_likelihood[s] == pytest.approx(logsumexp(per_state))


def test_predictive_log_probs_sum_to_the_marginal_likelihood():
    seqs, _ = make_sequences(np.random.default_rng(5), [8, 5])
    result = run_forward_backward(seqs, toy_params(2))
    np.testing.assert_allclose(
        np.where(seqs.mask, result.log_predictive, 0).sum(axis=1), result.log_likelihood
    )
    assert np.all(result.log_predictive[~seqs.mask] == 0)


def test_heldout_log_likelihood_counts_only_scored_trials():
    seqs, _ = make_sequences(np.random.default_rng(6), [6, 6])
    params = toy_params(2)
    scored = seqs.mask.copy()
    scored[:, 0] = False
    predictive = hmm.predictive_log_probs(params, seqs)
    full = hmm.heldout_log_likelihood(params, seqs)
    part = hmm.heldout_log_likelihood(params, seqs._replace(scored=scored))
    assert full - part == pytest.approx(predictive[:, 0].sum())
    assert part == pytest.approx(predictive[scored].sum())


def test_em_never_decreases_the_penalized_log_likelihood():
    rng = np.random.default_rng(7)
    seqs, _ = make_sequences(rng, [40] * 8)
    init = hmm.Params(rng.normal(size=(2, 2)), np.array([0.5, 0.5]), sticky(2, 0.9))
    fit = hmm.fit_em(seqs, init, hmm.Prior(), max_iterations=8, tolerance=0.0)
    steps = np.diff(fit.objective_trace)
    assert len(fit.objective_trace) == 9
    assert np.all(steps >= -1e-6)
    assert steps.sum() > 0


def test_one_state_equals_a_plain_ridge_conditional_logit():
    weights = np.array([[1.5, -1.0]])
    seqs, _ = make_sequences(
        np.random.default_rng(8), [50] * 6, weights=weights, transition=np.array([[1.0]])
    )
    prior = hmm.Prior(l2=1.0)
    fit = hmm.fit_restarts(seqs, 1, prior, n_restarts=3, seed=0)
    X, y = seqs.X[seqs.mask], seqs.chosen[seqs.mask]

    def log_likelihood(w):
        u = X @ w
        return (u[np.arange(len(y)), y] - logsumexp(u, axis=1)).sum()

    def objective(w):
        return -log_likelihood(w) + 0.5 * prior.l2 * w @ w

    options = {"xatol": 1e-9, "fatol": 1e-12}
    best = minimize(objective, np.zeros(2), method="Nelder-Mead", options=options)
    np.testing.assert_allclose(fit.params.weights[0], best.x, atol=1e-4)
    assert fit.log_likelihood == pytest.approx(log_likelihood(best.x), abs=1e-4)


def test_recovers_a_two_state_toy():
    seqs, states = make_sequences(np.random.default_rng(9), [60] * 12)
    fit = hmm.fit_restarts(seqs, 2, hmm.Prior(), n_restarts=6, seed=1)
    assert fit.converged
    np.testing.assert_allclose(fit.params.weights, TWO_STATES, atol=0.7)
    decoded = hmm.posteriors(fit.params, seqs).argmax(axis=-1)
    assert (decoded[seqs.mask] == states[seqs.mask]).mean() > 0.9
    assert np.all(np.diag(fit.params.transition) > 0.85)


def test_two_states_beat_one_held_out_on_switching_data():
    rng = np.random.default_rng(10)
    train, _ = make_sequences(rng, [60] * 10)
    test, _ = make_sequences(rng, [60] * 6)
    one = hmm.fit_restarts(train, 1, hmm.Prior(), n_restarts=2, seed=0)
    two = hmm.fit_restarts(train, 2, hmm.Prior(), n_restarts=6, seed=0)
    gain = hmm.heldout_log_likelihood(two.params, test) - hmm.heldout_log_likelihood(
        one.params, test
    )
    assert gain > 5


def test_states_are_sorted_by_the_chosen_feature():
    params = hmm.Params(
        np.array([[0.1, 5.0], [2.0, 1.0], [-1.0, 0.0]]),
        np.array([0.2, 0.3, 0.5]),
        np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]]),
    )
    ordered = hmm.sort_states(params, feature=0)
    assert ordered.weights[:, 0].tolist() == [2.0, 0.1, -1.0]
    np.testing.assert_allclose(ordered.initial, [0.3, 0.2, 0.5])
    assert ordered.transition[0, 0] == 0.7 and ordered.transition[0, 1] == 0.2
    assert ordered.transition[2, 0] == 0.3 and ordered.transition[1, 1] == 0.8


def test_dwell_times_and_occupancy():
    np.testing.assert_allclose(hmm.dwell_times(np.array([[0.99, 0.01], [0.5, 0.5]])), [100.0, 2.0])
    gamma = np.array([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    mask = np.array([[True, True, False]])
    np.testing.assert_allclose(hmm.occupancy(gamma, mask), [0.5, 0.5])


def test_no_restart_converging_fails_loudly():
    seqs, _ = make_sequences(np.random.default_rng(11), [30] * 4)
    with pytest.raises(RuntimeError, match="converge"):
        hmm.fit_restarts(
            seqs, 2, hmm.Prior(), n_restarts=2, seed=0, max_iterations=1, tolerance=0.0
        )


def test_inputs_are_validated():
    seqs, _ = make_sequences(np.random.default_rng(12), [5, 3])
    prior = hmm.Prior()
    with pytest.raises(ValueError, match="states"):
        hmm.fit_restarts(seqs, 0, prior, n_restarts=2, seed=0)
    with pytest.raises(ValueError, match="restarts"):
        hmm.fit_restarts(seqs, 2, prior, n_restarts=0, seed=0)
    with pytest.raises(ValueError, match="finite"):
        hmm.fit_restarts(seqs._replace(X=np.full_like(seqs.X, np.nan)), 2, prior, 2, 0)
    with pytest.raises(ValueError, match="chosen"):
        hmm.fit_restarts(seqs._replace(chosen=seqs.chosen + 3), 2, prior, 2, 0)
    gappy = seqs.mask.copy()
    gappy[0, 1] = False
    with pytest.raises(ValueError, match="padding"):
        hmm.fit_restarts(seqs._replace(mask=gappy), 2, prior, 2, 0)
    with pytest.raises(ValueError, match="scored"):
        hmm.heldout_log_likelihood(toy_params(2), seqs._replace(scored=~seqs.mask))
    with pytest.raises(ValueError, match="l2"):
        hmm.Prior(l2=-1.0)
    with pytest.raises(ValueError, match="transition"):
        hmm.Params(np.zeros((2, 2)), np.array([0.5, 0.5]), np.array([[0.5, 0.6], [0.5, 0.5]]))


def test_the_sticky_prior_raises_the_diagonal():
    counts = np.array([[5.0, 5.0], [5.0, 5.0]])
    flat = hmm.transition_from_counts(counts, hmm.Prior(transition_alpha=1.0, transition_kappa=0.0))
    sticky_prior = hmm.transition_from_counts(counts, hmm.Prior(transition_kappa=20.0))
    assert flat[0, 0] == pytest.approx(0.5)
    assert sticky_prior[0, 0] > 0.8
    np.testing.assert_allclose(sticky_prior.sum(axis=1), 1.0)
