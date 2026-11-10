"""The switch regression: switch trials, choice sets, recency features, fits and validation."""

import numpy as np
import pandas as pd
import pytest

from src.agents import null_agents as na
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import switch_regression as sr
from tests.switch_regression_fixtures import DECAY, simulate_switch_mouse, table_from_rows

D = 0.8  # trace decay of the hand-computed features
CAP = 10
# the two hand-built problems share no tower
pytestmark = pytest.mark.filterwarnings("ignore:the problems of m1 fall into 2 groups")


def _p1(a1, b3, c2):
    return {"A1": a1, "B3": b3, "C2": c2}


def _p2(a3, b1, c3):
    return {"A3": a3, "B1": b1, "C3": c3}


# (problem, session, trial_in_session, after_gap, choice, reward, magnitudes); ports of problem 1
# sort as A1, B3, C2 (arms 0, 1, 2) and of problem 2 as A3, B1, C3.
HAND_ROWS = [
    (1, 0, 0, False, "A1", 2, _p1(2, 1, 0)),  # p1 pos0: session start
    (1, 0, 1, False, "A1", 0, _p1(0, 3, 1)),  # pos1: stay
    (1, 0, 2, False, "B3", 3, _p1(1, 3, 4)),  # pos2: switch after 0
    (1, 0, 3, False, "C2", 1, _p1(2, 0, 1)),  # pos3: switch after 3
    (1, 0, 4, True, "A1", 0, _p1(0, 2, 2)),  # pos4: after a dropped trial, no pair
    (1, 0, 5, False, "B3", 0, _p1(4, 0, 3)),  # pos5: switch after 0
    (1, 1, 0, False, "B3", 2, _p1(1, 2, 0)),  # pos6: session start, no pair
    (1, 1, 1, False, "C2", 0, _p1(3, 1, 0)),  # pos7: switch after 2
    (2, 0, 0, False, "A3", 1, _p2(1, 2, 3)),  # p2 pos0: problem start, no pair with p1
    (2, 0, 1, False, "B1", 4, _p2(0, 4, 2)),  # pos1: switch after 1
    (2, 0, 2, False, "B1", 1, _p2(1, 1, 1)),  # pos2: stay
]


@pytest.fixture(scope="module")
def hand():
    return sr.switch_data(table_from_rows(HAND_ROWS), n_lags=4, trace_decay=D, recency_cap=CAP)


def _column(data, name):
    return data.X[:, :, data.feature_names.index(name)]


# ---------------------------------------------------------------------------
# Switch trials and choice sets
# ---------------------------------------------------------------------------


def test_switch_trials_respect_session_gap_and_problem_starts(hand):
    assert list(hand.rows) == [2, 3, 5, 7, 9]  # rows of the design (problem 1 then problem 2)
    assert list(hand.after_zero) == [True, False, True, False, False]
    assert list(hand.problem) == [1, 1, 1, 1, 2]


def test_choice_set_is_the_two_other_arms_in_port_order(hand):
    np.testing.assert_array_equal(hand.alternatives, [[1, 2], [0, 2], [1, 2], [0, 2], [1, 2]])
    np.testing.assert_array_equal(hand.chosen, [0, 1, 0, 1, 0])


def test_switch_trials_are_those_of_the_null_agents_pairs():
    trials = simulate_switch_mouse("m1", 3, value_weight=0.0, trials_per_session=150)
    data = sr.switch_data(trials, n_lags=5, trace_decay=DECAY)
    decisions = na.decisions(trials, DECAY)
    np.testing.assert_array_equal(data.rows, np.flatnonzero(decisions.switched))
    np.testing.assert_array_equal(data.after_zero, decisions.prev_magnitude[data.rows] == 0)


def test_a_problem_without_three_offered_arms_fails_loudly():
    rows = [
        (1, 0, 0, False, "A1", 1, {"A1": 1, "B3": 2}),
        (1, 0, 1, False, "B3", 2, {"A1": 1, "B3": 2}),
    ]
    with pytest.raises(ValueError, match="three offered ports"):
        sr.switch_data(table_from_rows(rows), n_lags=2, trace_decay=D)


# ---------------------------------------------------------------------------
# Recency features, hand-computed
# ---------------------------------------------------------------------------


def test_feature_names_drop_lag_one_and_end_with_the_towers(hand):
    names = hand.feature_names
    assert names[:6] == ("choice_lag2", "choice_lag3", "choice_lag4", "trace", "recency",
                         "never_chosen")  # fmt: skip
    assert all(n.startswith("tower_") for n in names[6:])
    assert "choice_lag1" not in names  # the previous choice is never in the choice set


def test_choice_lags_are_reset_at_segment_starts(hand):
    np.testing.assert_array_equal(
        _column(hand, "choice_lag2"), [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]
    )
    np.testing.assert_array_equal(
        _column(hand, "choice_lag3"), [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]
    )
    np.testing.assert_array_equal(_column(hand, "choice_lag4"), np.zeros((5, 2)))


def test_trace_is_the_decayed_choice_count_within_the_segment(hand):
    expected = [[0, 0], [D + D**2, 0], [0, 0], [0, 0], [0, 0]]
    np.testing.assert_allclose(_column(hand, "trace"), expected)


def test_recency_counts_trials_within_the_problem_across_sessions(hand):
    never = np.log1p(CAP)
    expected = [[never, never], [np.log(3), never], [np.log(4), np.log(3)],
                [np.log(4), np.log(5)], [never, never]]  # fmt: skip
    np.testing.assert_allclose(_column(hand, "recency"), expected)
    np.testing.assert_array_equal(
        _column(hand, "never_chosen"), [[1, 1], [0, 1], [0, 0], [0, 0], [1, 1]]
    )


def test_recency_is_capped():
    rows = [(1, 0, 0, False, "A1", 1, _p1(1, 1, 1)), (1, 0, 1, False, "B3", 1, _p1(1, 1, 1))]
    rows += [(1, 0, i, False, "B3", 1, _p1(1, 1, 1)) for i in range(2, 30)]
    rows += [(1, 0, 30, False, "C2", 1, _p1(1, 1, 1))]
    data = sr.switch_data(table_from_rows(rows), n_lags=2, trace_decay=D, recency_cap=CAP)
    last = data.feature_names.index("recency")
    # the last switch leaves B3 for A1 (chosen 29 trials ago, capped) or C2 (never chosen)
    np.testing.assert_allclose(data.X[-1, :, last], [np.log1p(CAP), np.log1p(CAP)])
    np.testing.assert_array_equal(_column(data, "never_chosen")[-1], [0, 1])


def test_never_chosen_is_the_believed_values_never_visited_indicator():
    trials = simulate_switch_mouse("m1", 4, value_weight=0.5, trials_per_session=150)
    data = sr.switch_data(trials, n_lags=5, trace_decay=DECAY)
    design = cl.design_matrix(trials, 5, None, extras=(cl.BELIEVED,))
    unvisited = design.X[:, :, design.feature_names.index("believed_unvisited")]
    expected = np.take_along_axis(unvisited[data.rows], data.alternatives, axis=1)
    np.testing.assert_array_equal(_column(data, "never_chosen"), expected)


def test_trace_difference_is_the_null_agents_own():
    trials = simulate_switch_mouse("m1", 5, value_weight=0.0, trace_weight=1.0)
    data = sr.switch_data(trials, n_lags=5, trace_decay=DECAY)
    decisions = na.decisions(trials, DECAY)
    trace = _column(data, "trace")
    np.testing.assert_allclose(trace[:, 0] - trace[:, 1], decisions.delta_h[data.rows])


# ---------------------------------------------------------------------------
# Values and leakage
# ---------------------------------------------------------------------------


def test_true_value_is_the_magnitude_offered_on_the_switch_trial(hand):
    np.testing.assert_array_equal(hand.values["true"], [[3, 4], [2, 1], [0, 3], [3, 0], [4, 2]])


def test_believed_value_is_the_last_magnitude_received_in_the_problem(hand):
    np.testing.assert_array_equal(hand.values["believed"], [[0, 0], [0, 0], [3, 1], [0, 1], [0, 0]])


def test_true_value_follows_task_values_convention():
    from src.pipelines import choice_logit_values as pipe

    trials = simulate_switch_mouse("m1", 6, value_weight=0.5, trials_per_session=150)
    analysis = pipe.analyse_trial_table(trials, n_lags=2, tail_decay=None, min_trials=10)
    data = sr.switch_data(trials, n_lags=2, trace_decay=DECAY)
    task = pipe.task_values(analysis)
    np.testing.assert_array_equal(
        data.values["true"], np.take_along_axis(task[data.rows], data.alternatives, axis=1)
    )


def test_features_use_only_trials_before_the_switch():
    trials = simulate_switch_mouse("m1", 7, value_weight=0.5, trials_per_session=150)
    data = sr.switch_data(trials, n_lags=5, trace_decay=DECAY)
    cut = int(data.rows[len(data.rows) // 3])  # a switch trial of problem 1
    changed = trials.copy()
    later = (changed["problem"] == 1) & (changed["trial"] > cut)
    changed.loc[later, "choice"] = changed.loc[later, "choice"].iloc[::-1].to_numpy()
    changed["reward"] = [
        m[c] for m, c in zip(changed["magnitudes"], changed["choice"], strict=True)
    ]
    # the reward of the switch trial itself is not a feature of it either
    at_cut = (changed["problem"] == 1) & (changed["trial"] == cut)
    changed.loc[at_cut, "reward"] = 99
    other = sr.switch_data(changed, n_lags=5, trace_decay=DECAY)
    upto = data.rows <= cut
    np.testing.assert_array_equal(other.rows[: upto.sum()], data.rows[upto])
    np.testing.assert_allclose(other.X[: upto.sum()], data.X[upto])
    for kind in ("true", "believed"):
        np.testing.assert_allclose(other.values[kind][: upto.sum()], data.values[kind][upto])


def test_model_values_must_be_aligned_with_the_trials(hand):
    trials = table_from_rows(HAND_ROWS)
    design = cl.design_matrix(trials, 2)
    values = np.arange(len(trials) * 3, dtype=float).reshape(-1, 3)
    model = sr.ModelValues(
        design.meta["problem"].to_numpy(), design.meta["trial"].to_numpy(), values
    )
    data = sr.switch_data(trials, n_lags=4, trace_decay=D, recency_cap=CAP, model_values=model)
    expected = np.take_along_axis(values[data.rows], data.alternatives, axis=1)
    np.testing.assert_array_equal(data.values["model"], expected)
    shuffled = model._replace(trial=model.trial[::-1])
    with pytest.raises(ValueError, match="not aligned"):
        sr.switch_data(trials, n_lags=4, trace_decay=D, model_values=shuffled)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _logit_data(beta, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, len(beta)))
    utility = X @ np.asarray(beta)
    p_first = 1 / (1 + np.exp(utility[:, 1] - utility[:, 0]))
    chosen = (rng.random(n) >= p_first).astype(int)
    return X, chosen


def test_swapping_the_alternatives_changes_no_weight_or_likelihood():
    X, chosen = _logit_data([0.7, -0.4, 0.0], 800, 0)
    names = ("a", "b", "c")
    fit = sr.fit_switch_logit(X, chosen, names, l2=1.0)
    swapped = sr.fit_switch_logit(X[:, ::-1], 1 - chosen, names, l2=1.0)
    np.testing.assert_allclose(swapped.beta, fit.beta, atol=1e-6)
    np.testing.assert_allclose(swapped.se, fit.se, rtol=1e-6)
    assert swapped.nll == pytest.approx(fit.nll, abs=1e-8)


def test_fit_recovers_known_weights_with_their_standard_errors():
    beta = np.array([0.8, -0.5, 0.0])
    X, chosen = _logit_data(beta, 4000, 1)
    fit = sr.fit_switch_logit(X, chosen, ("a", "b", "c"), l2=1.0)
    assert np.all(np.abs(fit.beta - beta) < 4 * fit.se)
    assert np.all((fit.se > 0.01) & (fit.se < 0.1))


def test_fit_validates_its_inputs():
    X, chosen = _logit_data([0.5], 50, 2)
    with pytest.raises(ValueError, match="two alternatives"):
        sr.fit_switch_logit(np.zeros((50, 3, 1)), chosen, ("a",), l2=1.0)
    with pytest.raises(ValueError, match="0 or 1"):
        sr.fit_switch_logit(X, chosen + 1, ("a",), l2=1.0)


@pytest.mark.parametrize("value_weight", [0.0, 0.6])
def test_value_weight_is_recovered_from_a_mouse_that_follows_the_model(value_weight):
    trials = simulate_switch_mouse("m1", 11, value_weight=value_weight, trace_weight=1.0)
    result = sr.analyze_mouse(trials, n_lags=5, trace_grid=(0.5, DECAY), min_switches=100)
    weights = sr.value_table(result.scores, result.weights).set_index(["condition", "value"])
    for condition in sr.CONDITIONS:
        row = weights.loc[(condition, "true")]
        assert abs(row["beta"] - value_weight) < 4 * row["se"], (condition, row)
    assert result.trace_decay == DECAY  # the generating decay wins the held-out comparison
    if value_weight > 0:
        assert (weights.xs("true", level="value")["gain"] > 0.01).all()
    else:
        assert (weights.xs("true", level="value")["gain"].abs() < 0.01).all()


def test_too_few_switches_in_a_condition_fail_loudly():
    trials = simulate_switch_mouse("m1", 12, value_weight=0.0, trials_per_session=60)
    with pytest.raises(ValueError, match="switches after"):
        sr.analyze_mouse(trials, n_lags=5, trace_grid=(DECAY,), min_switches=10_000)


def test_switch_counts_per_condition(hand):
    counts = sr.switch_counts(hand)
    assert counts == {"after_zero": 2, "after_nonzero": 3}


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def test_group_summary_needs_six_mice_for_a_mean_and_a_test():
    rng = np.random.default_rng(0)
    rows = [
        dict(mouse=f"m{i}", condition=c, value="true", beta=b, se=0.1, gain=g)
        for i, (b, g) in enumerate(
            zip(rng.normal(0.5, 0.1, 7), rng.normal(0.02, 0.01, 7), strict=True)
        )
        for c in sr.CONDITIONS
    ]
    table = pd.DataFrame(rows)
    many = sr.group_summary(table).set_index(["condition", "value", "term"])
    row = many.loc[("after_zero", "true", "beta")]
    assert row["n"] == 7 and row["mean"] > 0 and row["p_wilcoxon"] < 0.05
    assert np.isfinite(row["re_mean"]) and row["re_se"] > 0
    few = sr.group_summary(table[table["mouse"].isin(["m0", "m1", "m2"])])
    assert few["mean"].isna().all() and few["p_wilcoxon"].isna().all()
    assert few["re_mean"].isna().all()


def test_random_effects_summary_matches_hand_computation():
    beta = np.array([0.2, 0.5, 0.1, 0.4, 0.3, 0.6])
    se = np.array([0.1, 0.2, 0.1, 0.15, 0.1, 0.2])
    mean, error, tau2 = sr.random_effects(beta, se)
    w = 1 / se**2
    fixed = (w * beta).sum() / w.sum()
    q = (w * (beta - fixed) ** 2).sum()
    expected_tau2 = max(0.0, (q - 5) / (w.sum() - (w**2).sum() / w.sum()))
    w_star = 1 / (se**2 + expected_tau2)
    assert tau2 == pytest.approx(expected_tau2)
    assert mean == pytest.approx((w_star * beta).sum() / w_star.sum())
    assert error == pytest.approx(1 / np.sqrt(w_star.sum()))


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------


def _null_rows(mouse, betas, gains):
    return pd.DataFrame(
        dict(mouse=mouse, condition="after_zero", value="true", beta=betas, se=0.1, gain=gains,
             replicate=np.arange(len(betas)))
    )  # fmt: skip


def test_null_calibration_gives_z_percentile_and_empirical_p():
    null = _null_rows("m1", np.arange(-9, 10) / 10.0, np.zeros(19))  # mean 0, sd 0.5627
    observed = pd.DataFrame(
        dict(mouse=["m1"], condition="after_zero", value="true", beta=[0.85], se=0.1, gain=[0.0])
    )
    table = sr.null_calibration(observed, null).set_index("term")
    beta = table.loc["beta"]
    assert beta["n_null"] == 19
    assert beta["null_mean"] == pytest.approx(0.0)
    assert beta["z"] == pytest.approx(0.85 / np.std(np.arange(-9, 10) / 10.0, ddof=1))
    assert beta["percentile"] == pytest.approx(18 / 19)  # 18 of 19 null values below 0.85
    # two-sided: |null| >= 0.85 for -0.9 and 0.9, so p = (1 + 2) / (19 + 1)
    assert beta["p_empirical"] == pytest.approx(3 / 20)
    assert beta["null_low"] < -0.8 and beta["null_high"] > 0.8


def test_an_observation_beyond_every_null_value_has_the_smallest_p():
    null = _null_rows("m1", np.linspace(-0.1, 0.1, 9), np.zeros(9))
    observed = pd.DataFrame(
        dict(mouse=["m1"], condition="after_zero", value="true", beta=[1.0], se=0.1, gain=[0.0])
    )
    beta = sr.null_calibration(observed, null).set_index("term").loc["beta"]
    assert beta["p_empirical"] == pytest.approx(1 / 10)
    assert beta["percentile"] == 1.0


def test_null_calibration_needs_a_null_for_every_observation():
    observed = pd.DataFrame(
        dict(mouse=["m2"], condition="after_zero", value="true", beta=[1.0], se=0.1, gain=[0.0])
    )
    with pytest.raises(ValueError, match="no null"):
        sr.null_calibration(observed, _null_rows("m1", np.zeros(5), np.zeros(5)))


def test_calibrated_summary_tests_the_z_of_six_mice():
    rows = [
        dict(mouse=f"m{i}", condition="after_zero", value="true", term="beta", z=z, p_empirical=0.1)
        for i, z in enumerate([1.0, 2.0, 1.5, 0.5, 2.5, 1.2])
    ]
    summary = sr.calibrated_summary(pd.DataFrame(rows)).iloc[0]
    assert summary["n"] == 6 and summary["mean"] == pytest.approx(1.45)
    assert summary["p_wilcoxon"] == pytest.approx(2 / 64)
    few = sr.calibrated_summary(pd.DataFrame(rows[:5])).iloc[0]
    assert np.isnan(few["mean"]) and np.isnan(few["p_wilcoxon"])


def test_firth_shrinks_the_small_sample_bias_of_a_rare_event_fit():
    # a strong effect on few switches: maximum likelihood overestimates it, Firth less
    estimates = {False: [], True: []}
    for seed in range(40):
        X, chosen = _logit_data([1.5], 40, 100 + seed)
        for firth in estimates:
            estimates[firth].append(
                sr.fit_switch_logit(X, chosen, ("a",), 0.0, firth=firth).beta[0]
            )
    bias = {k: np.mean(v) - 1.5 for k, v in estimates.items()}
    assert abs(bias[True]) < abs(bias[False])


def test_firth_gradient_matches_finite_differences():
    X, chosen = _logit_data([0.6, -0.3], 60, 3)
    difference, first = X[:, 0] - X[:, 1], (chosen == 0).astype(float)
    beta = np.array([0.2, -0.1])
    _, gradient = sr._objective(beta, difference, first, 0.5, True)
    step = 1e-6
    numeric = [
        (sr._objective(beta + step * e, difference, first, 0.5, True)[0]
         - sr._objective(beta - step * e, difference, first, 0.5, True)[0]) / (2 * step)
        for e in np.eye(2)
    ]  # fmt: skip
    np.testing.assert_allclose(gradient, numeric, rtol=1e-5)
