"""The choice logit: design matrix, likelihood, fitting, model values and cross-validation."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.special import logsumexp

from src.agents import drifting_simulation as sim
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit.choice_logit_cv import (
    cross_validate_mouse,
    model_comparison,
)
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table

PORTS = ["A1", "B3", "C2"]


def trial_frame(choices, magnitudes, session_starts, *, ports=PORTS, problem=1, mouse="m1"):
    """A drifting trial table for one problem; *session_starts* are the session-first trials."""
    n = len(choices)
    in_session, since = [], 0
    for t in range(n):
        since = 0 if t in session_starts else since + 1
        in_session.append(since)
    mags = [dict(zip(ports, m, strict=True)) for m in magnitudes]
    return pd.DataFrame(
        dict(
            mouse=mouse,
            problem=problem,
            session=[f"s{sum(s <= t for s in session_starts)}" for t in range(n)],
            session_index=[sum(s <= t for s in session_starts) - 1 for t in range(n)],
            trial=range(n),
            trial_in_session=in_session,
            choice=choices,
            magnitudes=mags,
            reward=[m[c] for m, c in zip(mags, choices, strict=True)],
            initiation="B2",
            block=None,
            after_gap=False,
        )
    )


@pytest.fixture
def hand_built():
    # sessions start at trials 0 and 3
    return trial_frame(
        ["B3", "A1", "A1", "C2"],
        [(1, 3, 0), (2, 3, 0), (4, 3, 0), (1, 1, 2)],
        session_starts={0, 3},
    )


def test_design_reward_lags_carry_across_sessions_and_zero_pad(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2)
    names = list(design.feature_names)
    r1, r2 = names.index("reward_lag1"), names.index("reward_lag2")
    # received reward per trial and arm (A1, B3, C2): t0 B3 paid 3; t1 A1 paid 2; t2 A1 paid 4
    np.testing.assert_array_equal(design.X[0, :, r1], [0, 0, 0])
    np.testing.assert_array_equal(design.X[1, :, r1], [0, 3, 0])
    assert (design.X[1, :, r2] == 0).all()  # fewer than two previous trials: zero-padded
    np.testing.assert_array_equal(design.X[3, :, r1], [4, 0, 0])  # across the session start
    np.testing.assert_array_equal(design.X[3, :, r2], [2, 0, 0])


def test_design_choice_lags_reset_at_session_start(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2)
    names = list(design.feature_names)
    c1, c2 = names.index("choice_lag1"), names.index("choice_lag2")
    np.testing.assert_array_equal(design.X[2, :, c1], [1, 0, 0])  # t1 chose A1
    np.testing.assert_array_equal(design.X[2, :, c2], [0, 1, 0])  # t0 chose B3
    assert (design.X[3, :, [c1, c2]] == 0).all()  # t3 is a session start


def test_design_choice_lags_reset_after_a_dropped_trial(hand_built):
    hand_built.loc[2, "after_gap"] = True
    design = cl.design_matrix(hand_built, n_lags=2)
    names = list(design.feature_names)
    assert (design.X[2, :, names.index("choice_lag1")] == 0).all()
    # the reward history is not broken by the gap
    np.testing.assert_array_equal(design.X[2, :, names.index("reward_lag1")], [2, 0, 0])


def test_design_tower_bias_drops_the_reference_tower(hand_built):
    design = cl.design_matrix(hand_built, n_lags=1)
    assert design.feature_names == (
        "reward_lag1",
        "choice_lag1",
        "tower_B3",
        "tower_C2",
    )
    np.testing.assert_array_equal(design.X[0, :, 2], [0, 1, 0])
    np.testing.assert_array_equal(design.X[0, :, 3], [0, 0, 1])
    np.testing.assert_array_equal(design.chosen, [1, 0, 0, 2])


def test_design_has_no_distance_feature(hand_built):
    names = cl.design_matrix(hand_built, n_lags=3).feature_names
    assert not any("dist" in name for name in names)


def test_design_rejects_bad_inputs(hand_built):
    with pytest.raises(ValueError, match="n_lags"):
        cl.design_matrix(hand_built, n_lags=0)
    with pytest.raises(ValueError, match="one mouse"):
        cl.design_matrix(pd.concat([hand_built, hand_built.assign(mouse="m2")]), n_lags=1)
    changed = hand_built.copy()
    changed.at[2, "magnitudes"] = {"A1": 1, "B3": 1, "D4": 1}
    changed.at[2, "choice"] = "A1"
    with pytest.raises(ValueError, match="offered ports change"):
        cl.design_matrix(changed, n_lags=1)


def test_design_orders_trials_within_a_problem(hand_built):
    shuffled = hand_built.sample(frac=1, random_state=0)
    a, b = cl.design_matrix(hand_built, 2), cl.design_matrix(shuffled, 2)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.chosen, b.chosen)


def test_design_does_not_leak_history_between_problems(hand_built):
    second = hand_built.assign(problem=2, trial=range(4))
    both = cl.design_matrix(pd.concat([hand_built, second]), n_lags=2)
    alone = cl.design_matrix(second, n_lags=2)
    np.testing.assert_array_equal(both.X[4:, :, :4], alone.X[:, :, :4])
    assert (both.X[4, :, :4] == 0).all()


def random_problem(rng, n=400, n_features=5, beta=None):
    X = rng.normal(size=(n, 3, n_features))
    beta = np.arange(1, n_features + 1) * 0.3 if beta is None else beta
    utility = X @ beta
    p = np.exp(utility - logsumexp(utility, axis=1, keepdims=True))
    chosen = (rng.random(n)[:, None] > p.cumsum(axis=1)).sum(axis=1).clip(max=2)
    return X, chosen, beta


def test_nll_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X, chosen, _ = random_problem(rng)
    beta = rng.normal(size=X.shape[2])
    _, gradient = cl.negative_log_likelihood(beta, X, chosen, l2=0.7)
    numeric = approx_fprime(
        beta, lambda b: cl.negative_log_likelihood(b, X, chosen, l2=0.7)[0], 1e-6
    )
    np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-5)


def test_nll_at_zero_is_n_log_3():
    rng = np.random.default_rng(1)
    X, chosen, _ = random_problem(rng, n=50)
    value, _ = cl.negative_log_likelihood(np.zeros(X.shape[2]), X, chosen, l2=0.0)
    assert value == pytest.approx(50 * np.log(3))


def test_fit_recovers_known_weights():
    rng = np.random.default_rng(2)
    X, chosen, beta = random_problem(rng, n=30000)
    names = tuple(f"f{i}" for i in range(X.shape[2]))
    fit = cl.fit_logit(X, chosen, names, l2=1e-3)
    np.testing.assert_allclose(fit.beta, beta, atol=0.06)
    assert fit.n_trials == 30000
    assert fit.feature_names == names


def test_l2_shrinks_weights():
    rng = np.random.default_rng(3)
    X, chosen, _ = random_problem(rng, n=200)
    names = tuple(f"f{i}" for i in range(X.shape[2]))
    weak = cl.fit_logit(X, chosen, names, l2=0.01)
    strong = cl.fit_logit(X, chosen, names, l2=500.0)
    assert np.linalg.norm(strong.beta) < 0.5 * np.linalg.norm(weak.beta)


def test_fit_fails_loudly_without_convergence(monkeypatch):
    rng = np.random.default_rng(4)
    X, chosen, _ = random_problem(rng, n=100)
    names = tuple(f"f{i}" for i in range(X.shape[2]))
    with pytest.raises(RuntimeError, match="did not converge"):
        cl.fit_logit(X, chosen, names, l2=1.0, max_iterations=1)


def test_fit_rejects_invalid_choice_index():
    X = np.zeros((4, 3, 2))
    with pytest.raises(ValueError, match="chosen"):
        cl.fit_logit(X, np.array([0, 1, 3, 0]), ("a", "b"))


def test_model_value_uses_only_the_reward_weights():
    X = np.random.default_rng(5).normal(size=(20, 3, 4))
    names = ("reward_lag1", "reward_lag2", "choice_lag1", "tower_B3")
    base = cl.LogitFit(np.array([0.5, 0.25, 1.0, 2.0]), names, 0.0, 20, 1.0)
    changed = base._replace(beta=np.array([0.5, 0.25, -7.0, 9.0]))
    expected = X[:, :, 0] * 0.5 + X[:, :, 1] * 0.25
    np.testing.assert_allclose(cl.model_values(X, base), expected)
    np.testing.assert_allclose(cl.model_values(X, changed), expected)


def test_select_features_keeps_groups_in_order(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2)
    reward_only = cl.select_features(design, (cl.REWARD,))
    assert reward_only.feature_names == ("reward_lag1", "reward_lag2")
    assert reward_only.X.shape == (4, 3, 2)
    np.testing.assert_array_equal(reward_only.chosen, design.chosen)
    with pytest.raises(ValueError, match="unknown feature group"):
        cl.select_features(design, ("distance",))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def three_problem_design(seed=0, n=300):
    """A design with random features, three problems, and an extra tower-bias column."""
    rng = np.random.default_rng(seed)
    names = ("reward_lag1", "reward_lag2", "choice_lag1", "tower_B3")
    parts, chosen, problems = [], [], []
    for problem in (1, 2, 3):
        # a different reward weight per problem, so folds that include or exclude it differ
        X, y, _ = random_problem(
            rng, n=n, n_features=4, beta=np.array([problem * 0.4, 0.2, 0.5, 0])
        )
        parts.append(X)
        chosen.append(y)
        problems += [problem] * n
    meta = pd.DataFrame(dict(mouse="m1", problem=problems, trial=np.tile(np.arange(n), 3)))
    return cl.LogitDesign(np.concatenate(parts), np.concatenate(chosen), names, meta, None)


def test_held_out_values_come_from_the_fold_that_excluded_the_problem():
    design = three_problem_design()
    result = cross_validate_mouse(design, l2=1.0)
    for problem in (1, 2, 3):
        train = (design.meta["problem"] != problem).to_numpy()
        fit = cl.fit_logit(design.X[train], design.chosen[train], design.feature_names, l2=1.0)
        held_out = ~train
        np.testing.assert_allclose(
            result.values[held_out], cl.model_values(design.X[held_out], fit), atol=1e-4
        )
    everything = cl.fit_logit(design.X, design.chosen, design.feature_names, l2=1.0)
    assert not np.allclose(result.values, cl.model_values(design.X, everything), atol=1e-3)


def test_scores_cover_every_problem_and_model():
    design = three_problem_design()
    result = cross_validate_mouse(design, l2=1.0)
    scores = result.scores
    assert set(scores["model"]) == {"reward", "reward+choice", "full"}
    assert set(scores["problem"]) == {1, 2, 3}
    assert (scores["n_trials"] == 300).all()
    assert (scores["held_out_log_likelihood"] < 0).all()


def test_model_comparison_ranks_nested_models_on_data_generated_by_the_full_model():
    rng = np.random.default_rng(8)
    names = ("reward_lag1", "choice_lag1", "tower_B3")
    parts, chosen, problems = [], [], []
    for problem in (1, 2, 3, 4):
        X, y, _ = random_problem(rng, n=1500, n_features=3, beta=np.array([0.6, 1.0, 0.8]))
        parts.append(X)
        chosen.append(y)
        problems += [problem] * 1500
    meta = pd.DataFrame(dict(mouse="m1", problem=problems))
    design = cl.LogitDesign(np.concatenate(parts), np.concatenate(chosen), names, meta, None)
    table = model_comparison(cross_validate_mouse(design, l2=1.0).scores)
    nats = table.set_index("model")["nats_per_choice"]
    assert nats["full"] < nats["reward+choice"] < nats["reward"] < np.log(3)
    gain = table.set_index("model")["gain_over_chance"]
    assert gain["full"] == pytest.approx(np.log(3) - nats["full"])


def test_cross_validation_needs_two_problems():
    design = three_problem_design()
    one = design.meta["problem"].to_numpy() == 1
    single = cl.LogitDesign(
        design.X[one], design.chosen[one], design.feature_names, design.meta[one], None
    )
    with pytest.raises(ValueError, match="at least two problems"):
        cross_validate_mouse(single, l2=1.0)


def test_a_tower_absent_from_training_gets_no_weight():
    design = three_problem_design()
    X = design.X.copy()
    problem3 = (design.meta["problem"] == 3).to_numpy()
    X[~problem3, :, 3] = 0  # tower_B3 only ever appears in problem 3
    unseen = design._replace(X=X)
    fit = cl.fit_logit(X[~problem3], unseen.chosen[~problem3], unseen.feature_names, l2=1.0)
    assert fit.beta[3] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# A Q-learner's reward kernel
# ---------------------------------------------------------------------------


def simulated_q_learner_table(learning_rate=0.3, beta=1.2, n_problems=3, seed=0):
    rng = np.random.default_rng(seed)
    rules = sim.TaskRules(0.15, np.arange(30, 36))
    frames = {}
    for k, ports in enumerate((["A1", "B3", "C2"], ["A3", "B2", "C1"], ["A2", "B1", "C3"]), 1):
        if k > n_problems:
            break
        layout = sim.MouseLayout(ports, "B2", dict(zip(ports, (4, 2, 0), strict=True)), [500] * 3)
        played = sim.simulate_problem(
            (learning_rate, beta, 0.0, 0.0), "magnitude", layout, rules, rng
        )
        frames[k] = {"q": sim.simulated_sessions("q", k, layout, played)}
    return drifting_trial_table(frames)


@pytest.mark.filterwarnings("ignore:the problems")
def test_q_learner_reward_kernel_is_positive_and_decays():
    table = simulated_q_learner_table()
    design = cl.design_matrix(table, n_lags=5)
    fit = cl.fit_logit(design.X, design.chosen, design.feature_names, l2=1.0)
    kernel = fit.beta[:5]
    assert (kernel > 0).all()
    assert kernel[0] > kernel[1] > kernel[2]  # recent rewards weigh most
    assert kernel[:2].mean() > 1.5 * kernel[3:].mean()


# ---------------------------------------------------------------------------
# Reward tail, tower sets, soft targets
# ---------------------------------------------------------------------------


def test_tail_is_the_decayed_sum_of_rewards_older_than_the_lags():
    choices = ["A1", "A1", "B3", "A1", "C2", "A1", "A1"]
    magnitudes = [(1, 0, 2), (2, 0, 0), (3, 2, 0), (4, 0, 0), (1, 1, 3), (2, 0, 0), (1, 0, 0)]
    frame = trial_frame(choices, magnitudes, session_starts={0})
    design = cl.design_matrix(frame, n_lags=1, tail_decay=0.5)
    assert design.feature_names[:2] == ("reward_lag1", "reward_tail")
    received = np.zeros((7, 3))
    for t, (c, m) in enumerate(zip(choices, magnitudes, strict=True)):
        received[t, PORTS.index(c)] = m[PORTS.index(c)]
    for t in range(7):
        expected = sum(0.5 ** (j - 2) * received[t - j] for j in range(2, t + 1))
        np.testing.assert_allclose(design.X[t, :, 1], expected if t >= 2 else 0)


def test_tail_resets_at_a_problem_start(hand_built):
    second = hand_built.assign(problem=2, trial=range(4))
    both = cl.design_matrix(pd.concat([hand_built, second]), n_lags=1, tail_decay=0.9)
    assert (both.X[4:6, :, 1] == 0).all()


def test_model_value_includes_the_tail_but_no_choice_or_tower_terms():
    X = np.random.default_rng(6).normal(size=(10, 3, 4))
    names = ("reward_lag1", "reward_tail", "choice_lag1", "tower_B3")
    fit = cl.LogitFit(np.array([0.5, 0.2, 9.0, 9.0]), names, 0.0, 10, 1.0)
    np.testing.assert_allclose(cl.model_values(X, fit), X[:, :, 0] * 0.5 + X[:, :, 1] * 0.2)


def test_tail_weight_is_in_the_reward_group_without_a_lag(hand_built):
    design = cl.design_matrix(hand_built, n_lags=2, tail_decay=0.9)
    table = cl.weights_table("m", design.feature_names, np.arange(len(design.feature_names)))
    tail = table[table["feature"] == "reward_tail"].iloc[0]
    assert tail["group"] == "reward" and np.isnan(tail["lag"])
    assert cl.select_features(design, (cl.REWARD,)).feature_names == (
        "reward_lag1", "reward_lag2", "reward_tail",
    )  # fmt: skip


def test_tail_decay_must_be_a_proper_fraction(hand_built):
    with pytest.raises(ValueError, match="tail_decay"):
        cl.design_matrix(hand_built, n_lags=1, tail_decay=1.0)


def test_reference_tower_may_be_absent_from_a_problem():
    first = trial_frame(["B3", "A1", "C2", "B3"], [(1, 2, 0)] * 4, {0})
    second = trial_frame(
        ["B3", "C2", "C2", "B3"], [(1, 2, 0)] * 4, {0}, ports=["B3", "C2", "D4"], problem=2
    )
    design = cl.design_matrix(pd.concat([first, second]), n_lags=1)
    assert "tower_A1" not in design.feature_names  # A1 is the reference
    assert (design.X[4:, :, design.feature_names.index("tower_B3")].sum(axis=1) == 1).all()
    fit = cl.fit_logit(design.X, design.chosen, design.feature_names)
    assert np.isfinite(fit.beta).all()


def test_disconnected_tower_sets_warn_that_only_the_penalty_fixes_the_offset():
    first = trial_frame(["B3", "A1", "C2", "B3"], [(1, 2, 0)] * 4, {0})
    second = trial_frame(
        ["B1", "A3", "C3", "B1"], [(1, 2, 0)] * 4, {0}, ports=["A3", "B1", "C3"], problem=2
    )
    with pytest.warns(UserWarning, match="share no tower"):
        cl.design_matrix(pd.concat([first, second]), n_lags=1)


def test_a_session_whose_first_trial_was_dropped_still_resets_choice_history():
    frame = trial_frame(["B3", "A1", "A1"], [(1, 2, 0)] * 3, session_starts={0})
    frame["trial_in_session"] = [4, 5, 6]  # trials 0-3 of the session were dropped
    frame["after_gap"] = [True, False, False]
    design = cl.design_matrix(frame, n_lags=1)
    c1 = list(design.feature_names).index("choice_lag1")
    assert (design.X[0, :, c1] == 0).all()
    np.testing.assert_array_equal(design.X[1, :, c1], [0, 1, 0])


def test_soft_target_fit_recovers_the_generating_weights_exactly():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 3, 4))
    beta = np.array([0.4, -0.3, 0.9, 0.1])
    names = tuple(f"f{i}" for i in range(4))
    targets = np.exp(X @ beta - logsumexp(X @ beta, axis=1, keepdims=True))
    fit = cl.fit_soft_targets(X, targets, names, l2=0.0)
    np.testing.assert_allclose(fit.beta, beta, atol=1e-4)
