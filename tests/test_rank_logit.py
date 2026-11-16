"""The small binary logistic regression behind the rank tests."""

import numpy as np
import pytest
from scipy.special import expit

from src.behavior_analysis.rank_tests import rank_logit as rl


def simulate(n=4000, beta=(-0.5, 1.2, -0.8), seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = (rng.random(n) < expit(X @ np.array(beta))).astype(float)
    return X, y


def test_the_fit_recovers_known_weights_with_a_matching_standard_error():
    X, y = simulate()
    fit = rl.fit_logit(X, y, ("intercept", "a", "b"), l2=0.0)
    np.testing.assert_allclose(fit.beta, [-0.5, 1.2, -0.8], atol=0.15)
    assert np.all(np.abs(fit.beta - [-0.5, 1.2, -0.8]) < 4 * fit.se)
    assert fit.se.shape == (3,) and np.all(fit.se > 0)


def test_the_intercept_is_not_penalized():
    X, y = simulate(beta=(2.0, 0.0, 0.0))
    fit = rl.fit_logit(X, y, ("intercept", "a", "b"), l2=1e4)
    assert fit.beta[0] == pytest.approx(2.0, abs=0.15)
    assert np.all(np.abs(fit.beta[1:]) < 0.05)


def test_the_fit_is_the_stationary_point_of_the_penalized_likelihood():
    X, y = simulate(n=500)
    fit = rl.fit_logit(X, y, ("intercept", "a", "b"), l2=1.0)
    penalty = np.r_[0.0, fit.beta[1:]]  # the intercept is free
    gradient = X.T @ (expit(X @ fit.beta) - y) + penalty
    np.testing.assert_allclose(gradient, 0, atol=1e-5)


def test_covariance_gives_the_se_of_a_contrast():
    X, y = simulate()
    fit = rl.fit_logit(X, y, ("intercept", "a", "b"))
    assert rl.contrast_se(fit, {"a": 1.0, "b": 1.0}) == pytest.approx(
        np.sqrt(fit.cov[1, 1] + fit.cov[2, 2] + 2 * fit.cov[1, 2])
    )


def test_a_perfectly_separated_outcome_still_fits_because_of_the_penalty():
    x = np.r_[np.zeros(50), np.ones(50)]
    X = np.column_stack([np.ones(100), x])
    fit = rl.fit_logit(X, x, ("intercept", "a"))
    assert np.all(np.isfinite(fit.beta)) and fit.beta[1] > 2


def test_held_out_likelihood_prefers_the_model_with_the_true_feature():
    X, y = simulate(n=3000)
    groups = np.arange(len(y)) % 5
    both = rl.cross_validate(X, y, groups, ("intercept", "a", "b"))
    one = rl.cross_validate(X[:, :2], y, groups, ("intercept", "a"))
    assert both.log_likelihood > one.log_likelihood
    assert both.n == one.n == 3000


def test_cross_validation_needs_two_groups():
    X, y = simulate(n=100)
    with pytest.raises(ValueError, match="two groups"):
        rl.cross_validate(X, y, np.zeros(100, int), ("intercept", "a", "b"))


@pytest.mark.parametrize(
    "y", [np.array([0.0, 1.0, 2.0] * 10), np.zeros(30), np.array([0.0, 1.0] * 5)]
)
def test_an_outcome_that_is_not_binary_or_has_one_class_or_the_wrong_length_raises(y):
    X = np.column_stack([np.ones(30), np.arange(30.0)])
    with pytest.raises(ValueError):
        rl.fit_logit(X, y, ("intercept", "a"))


def test_a_nan_feature_raises():
    X, y = simulate(n=50)
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        rl.fit_logit(X, y, ("intercept", "a", "b"))
