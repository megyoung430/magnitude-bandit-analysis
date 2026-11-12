"""The Q-learner with a choice kernel, forgetting and asymmetric learning rates."""

import numpy as np
import pytest
from scipy.optimize import check_grad

from src.agents import kernel_q_agent as kq
from src.agents import null_agents as na
from src.agents import value_agent
from tests.switch_regression_fixtures import table_from_rows

PLUS = np.array([0.0, 1.0, 0.0])
# alpha_pos, alpha_neg, zeta, alpha_k, beta, beta_k, plus_bias
ALL_ACTIVE = (0.6, 0.2, 0.8, 0.3, 1.5, 0.9, 0.2)


def reference_log_likelihood(params, actions, rewards, session_start, plus):
    """One problem, scalar loops written straight from the update equations."""
    alpha_pos, alpha_neg, zeta, alpha_k, beta, beta_k, plus_bias = params
    q, k, total = np.zeros(3), np.zeros(3), 0.0
    for action, reward, start in zip(actions, rewards, session_start, strict=True):
        if start:
            k = np.zeros(3)
        logits = beta * q + beta_k * k + plus_bias * plus
        total += logits[action] - np.log(np.exp(logits).sum())
        delta = reward - q[action]
        for arm in range(3):
            if arm == action:
                q[arm] += (alpha_pos if delta > 0 else alpha_neg) * delta
                k[arm] += alpha_k * (1 - k[arm])
            else:
                q[arm] *= zeta
                k[arm] *= 1 - alpha_k
    return total


def _simulate(params, n_problems=3, n_trials=360, session_length=60, seed=0):
    """Choices of the agent on a bandit whose payoffs rotate every 40 trials."""
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(n_problems):
        q, k = np.zeros(3), np.zeros(3)
        actions, rewards, starts = [], [], []
        for t in range(n_trials):
            if t % session_length == 0:
                k = np.zeros(3)
            payoff = np.roll([4.0, 1.0, 0.0], (t // 40) % 3)
            a = rng.choice(3, p=kq.choice_probabilities(q, k, PLUS, params))
            q, k = kq.update_state(q, k, a, payoff[a], params)
            actions.append(a)
            rewards.append(payoff[a])
            starts.append(t % session_length == 0)
        problems.append(dict(actions=actions, rewards=rewards, session_start=starts, plus=PLUS))
    return value_agent.pad_sequences(problems)


def _problem(actions, rewards, starts):
    return dict(actions=actions, rewards=rewards, session_start=starts, plus=PLUS)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def test_state_update_by_hand_with_asymmetry_forgetting_and_kernel():
    q, k = kq.update_state(np.array([1.0, 2.0, 0.5]), np.array([0.5, 0.0, 0.2]), 0, 3.0,
                           (0.6, 0.2, 0.8, 0.3, 1.0, 1.0, 0.0))  # fmt: skip
    # delta = 3 - 1 = 2 > 0 so the rate is alpha_pos; the other arms keep zeta of their value
    assert q == pytest.approx([1.0 + 0.6 * 2.0, 2.0 * 0.8, 0.5 * 0.8])
    assert k == pytest.approx([0.5 + 0.3 * 0.5, 0.0, 0.2 * 0.7])
    q, _ = kq.update_state(np.array([1.0, 2.0, 0.5]), np.zeros(3), 1, 0.0, ALL_ACTIVE)
    assert q[1] == pytest.approx(2.0 + 0.2 * (0.0 - 2.0))  # delta < 0: alpha_neg


def test_zero_prediction_error_uses_the_negative_rate():
    q, _ = kq.update_state(np.array([2.0, 0.0, 0.0]), np.zeros(3), 0, 2.0, ALL_ACTIVE)
    assert q[0] == 2.0  # delta = 0 moves nothing whichever rate applies


def test_likelihood_of_a_toy_with_everything_active_by_hand():
    actions, rewards, starts = [0, 0, 1, 2, 2], [2.0, 0.0, 1.0, 4.0, 0.0], [1, 0, 0, 1, 0]
    seqs = value_agent.pad_sequences([_problem(actions, rewards, starts)])
    expected = reference_log_likelihood(ALL_ACTIVE, actions, rewards, starts, PLUS)
    assert kq.log_likelihood(ALL_ACTIVE, seqs) == pytest.approx(expected)
    # first trial: all values and kernels are 0, so only the plus bias separates the ports
    first = np.exp(0.2 * PLUS) / np.exp(0.2 * PLUS).sum()
    assert kq.trial_log_probabilities(ALL_ACTIVE, seqs)[0, 0] == pytest.approx(np.log(first[0]))


def test_likelihood_matches_the_reference_on_simulated_problems_with_padding():
    seqs = _simulate(ALL_ACTIVE, n_problems=2, n_trials=150)
    short = _problem([1, 2], [1.0, 0.0], [True, False])
    both = value_agent.pad_sequences(
        [
            _problem(seqs.actions[0].tolist(), seqs.rewards[0].tolist(), seqs.session_start[0]),
            short,
        ]
    )
    expected = reference_log_likelihood(
        ALL_ACTIVE, seqs.actions[0], seqs.rewards[0], seqs.session_start[0], PLUS
    ) + reference_log_likelihood(ALL_ACTIVE, [1, 2], [1.0, 0.0], [True, False], PLUS)
    assert kq.log_likelihood(ALL_ACTIVE, both) == pytest.approx(expected)


def test_special_case_is_the_value_agent():
    """alpha_pos = alpha_neg, no forgetting and a one-trial kernel is the value agent."""
    seqs = _simulate((0.4, 0.4, 1.0, 1.0, 1.2, 0.8, 0.3), n_problems=3)
    rate, beta, kappa, bias = 0.35, 0.9, 0.7, 0.25
    value = value_agent.log_likelihood((rate, beta, kappa, bias), seqs)
    kernel = kq.log_likelihood(kq.value_agent_parameters(rate, beta, kappa, bias), seqs)
    assert kernel == pytest.approx(value)
    # the per-trial log probabilities agree too, so the two predict the same choices
    assert np.allclose(
        kq.trial_log_probabilities(kq.value_agent_parameters(rate, beta, kappa, bias), seqs),
        value_agent.trial_log_probabilities((rate, beta, kappa, bias), seqs),
        equal_nan=True,
    )


def test_reset_convention_values_carry_across_sessions_and_kernel_does_not():
    """As the value agent: values persist over a session start, the kernel (stickiness) resets."""
    params = (0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 0.0)
    one_session = _problem([0, 0, 1], [3.0, 3.0, 0.0], [True, False, False])
    two_sessions = _problem([0, 0, 1], [3.0, 3.0, 0.0], [True, False, True])
    one = kq.trial_log_probabilities(params, value_agent.pad_sequences([one_session]))[0]
    two = kq.trial_log_probabilities(params, value_agent.pad_sequences([two_sessions]))[0]
    assert two[:2] == pytest.approx(one[:2])
    # trial 3 starts a session: the values from trials 1-2 (Q_0 = 2.25) still count ...
    q = np.zeros(3)
    for a, r in ((0, 3.0), (0, 3.0)):
        q, _ = kq.update_state(q, np.zeros(3), a, r, params)
    expected = kq.choice_probabilities(q, np.zeros(3), PLUS, params)[1]
    assert two[2] == pytest.approx(np.log(expected))
    # ... but the previous choice (arm 0) no longer sticks
    assert two[2] != pytest.approx(one[2])
    value = value_agent.trial_log_probabilities(
        (0.5, 1.0, 2.0, 0.0), value_agent.pad_sequences([two_sessions])
    )[0]
    assert two == pytest.approx(value)


def test_kernel_is_the_wsls_choice_trace_scaled_by_alpha_k():
    """With alpha_k = 0.1 the kernel is 1/10 of the trace h <- 0.9 h + chosen of WSLS."""
    ports = ("A1", "B3", "C2")
    choices = [0, 1, 1, 2, 0, 2, 2, 1, 0, 0]
    rows = [(1, 0, t, False, ports[c], 1, dict.fromkeys(ports, 1)) for t, c in enumerate(choices)]
    trials = table_from_rows(rows)
    d = na.decisions(trials, 0.9)
    params = (0.5, 0.5, 1.0, 0.1, 1.0, 1.0, 0.0)
    q, k = np.zeros(3), np.zeros(3)
    checked = 0
    for t, choice in enumerate(choices):
        if d.switched[t]:
            first, second = na.OTHERS[choices[t - 1]]
            assert d.delta_h[t] == pytest.approx(10 * (k[first] - k[second]))
            checked += 1
        q, k = kq.update_state(q, k, choice, 0.0, params)
    assert checked >= 4


# ---------------------------------------------------------------------------
# Gradient and variants
# ---------------------------------------------------------------------------


def _relative_gradient_error(function, gradient, params):
    error = check_grad(function, gradient, np.asarray(params, dtype=float))
    return error / np.linalg.norm(gradient(np.asarray(params, dtype=float)))


def test_gradient_is_exact_for_all_seven_parameters():
    seqs = _simulate(ALL_ACTIVE, n_problems=2, n_trials=200)
    params = np.array([0.5, 0.3, 0.7, 0.4, 1.1, 0.6, 0.2])
    error = _relative_gradient_error(
        lambda p: kq.log_likelihood(p, seqs),
        lambda p: kq.log_likelihood(p, seqs, return_gradient=True)[1],
        params,
    )
    assert error < 1e-5


@pytest.mark.parametrize("variant", kq.VARIANTS.values(), ids=list(kq.VARIANTS))
def test_gradient_of_every_variant_is_exact_in_its_free_parameters(variant):
    seqs = _simulate(ALL_ACTIVE, n_problems=2, n_trials=200)
    free = np.array([kq.START_CENTERS[name] for name in variant.free])
    error = _relative_gradient_error(
        lambda p: kq.variant_log_likelihood(variant, p, seqs),
        lambda p: kq.variant_log_likelihood(variant, p, seqs, return_gradient=True)[1],
        free,
    )
    assert error < 1e-5


def test_variants_expand_to_full_parameters():
    no_forgetting_free = [0.6, 0.2, 0.3, 1.5, 0.9, 0.2]
    assert kq.expand(kq.VARIANTS["no_asymmetry"], [0.4, 0.8, 0.3, 1.5, 0.9, 0.2]) == pytest.approx(
        (0.4, 0.4, 0.8, 0.3, 1.5, 0.9, 0.2)
    )
    assert kq.expand(kq.VARIANTS["no_forgetting"], no_forgetting_free)[2] == 1.0
    no_kernel = kq.expand(kq.VARIANTS["no_kernel"], [0.6, 0.2, 0.8, 1.5, 0.2])
    assert no_kernel[5] == 0.0  # beta_k
    assert kq.VARIANTS["no_kernel"].free == ("alpha_pos", "alpha_neg", "zeta", "beta", "plus_bias")
    only = kq.VARIANTS["kernel_only"]
    assert kq.expand(only, [0.4, 0.3, 1.5, 0.9, 0.2]) == pytest.approx(
        (0.4, 0.4, 1.0, 0.3, 1.5, 0.9, 0.2)
    )


def test_expand_rejects_the_wrong_number_of_free_values():
    with pytest.raises(ValueError, match="free parameters"):
        kq.expand(kq.VARIANTS["no_forgetting"], [0.5, 0.5])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_fit_recovers_the_parameters_of_simulated_choices():
    true = (0.7, 0.25, 0.7, 0.2, 2.0, 1.2, 0.4)
    seqs = _simulate(true, n_problems=4, n_trials=800, seed=5)
    fit = kq.fit(seqs, n_starts=5, seed=0)
    assert fit.n_trials == 4 * 800
    # the likelihood at the fit is at least that at the truth
    assert fit.log_likelihood >= kq.log_likelihood(true, seqs) - 1e-6
    assert fit.parameters[kq.INDEX["beta"]] == pytest.approx(true[4], rel=0.3)
    assert fit.parameters[kq.INDEX["beta_k"]] == pytest.approx(true[5], abs=0.5)
    assert fit.parameters[kq.INDEX["alpha_k"]] == pytest.approx(true[3], abs=0.1)
    assert fit.parameters[kq.INDEX["zeta"]] == pytest.approx(true[2], abs=0.2)


def test_a_variant_fit_returns_full_parameters_with_the_fixed_ones_in_place():
    seqs = _simulate(ALL_ACTIVE, n_problems=2, n_trials=200)
    fit = kq.fit(seqs, n_starts=2, seed=0, variant=kq.VARIANTS["no_forgetting"])
    assert len(fit.parameters) == len(kq.PARAMETER_NAMES)
    assert fit.parameters[kq.INDEX["zeta"]] == 1.0
    full = kq.fit(seqs, n_starts=2, seed=0)
    assert full.log_likelihood >= fit.log_likelihood - 1e-6  # nested: never worse


def test_parameters_at_bounds_names_only_free_parameters():
    params = (0.5, 0.5, 1.0, 0.3, 1.0, 0.0, 0.0)
    assert kq.parameters_at_bounds(params) == ("zeta",)
    assert kq.parameters_at_bounds(params, kq.VARIANTS["no_forgetting"]) == ()
    assert "beta_k" not in kq.parameters_at_bounds(params, kq.VARIANTS["no_kernel"])
    at_upper = (1.0, 0.5, 0.5, 1.0, 20.0, -20.0, 5.0)
    assert kq.parameters_at_bounds(at_upper) == (
        "alpha_pos", "alpha_k", "beta", "beta_k", "plus_bias"
    )  # fmt: skip


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="no trials"):
        value_agent.pad_sequences([])
    seqs = _simulate(ALL_ACTIVE, n_problems=1, n_trials=30)
    with pytest.raises(ValueError, match="n_starts"):
        kq.fit(seqs, n_starts=0)
    with pytest.raises(ValueError, match="seven parameters"):
        kq.log_likelihood((0.5, 1.0), seqs)
    with pytest.raises(ValueError, match="finite"):
        kq.log_likelihood((0.5, 0.5, 0.5, 0.5, np.nan, 0.0, 0.0), seqs)
    with pytest.raises(ValueError, match="zeta"):
        kq.log_likelihood((0.5, 0.5, 1.5, 0.5, 1.0, 0.0, 0.0), seqs)


def test_a_fit_that_never_converges_raises(monkeypatch):
    seqs = _simulate(ALL_ACTIVE, n_problems=1, n_trials=30)

    class Failed:
        success, fun, x = False, 0.0, np.zeros(7)

    monkeypatch.setattr(kq, "minimize", lambda *args, **kwargs: Failed())
    with pytest.raises(RuntimeError, match="converge"):
        kq.fit(seqs, n_starts=2)


def test_choice_probabilities_sum_to_one_and_follow_the_logits():
    p = kq.choice_probabilities(np.array([1.0, 0.0, 2.0]), np.array([0.0, 1.0, 0.5]), PLUS,
                                ALL_ACTIVE)  # fmt: skip
    logits = 1.5 * np.array([1.0, 0.0, 2.0]) + 0.9 * np.array([0.0, 1.0, 0.5]) + 0.2 * PLUS
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx(np.exp(logits) / np.exp(logits).sum())
