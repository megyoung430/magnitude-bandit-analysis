"""Held-out comparison of WSLS (Magnitude), Heuristic+ and the Q-learner, and model recovery."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import heuristic_agent as ha
from src.agents import null_agents as na
from src.agents import value_agent as va
from src.behavior_analysis import strategy_models as sm
from tests.strategy_fixtures import PARAMETERS, simulate_mouse
from tests.switch_regression_fixtures import table_from_rows

PORTS = ("A1", "B3", "C2")
pytestmark = pytest.mark.filterwarnings("ignore:the problems of")


def q_toy_table():
    """One problem, two sessions (r0-r3, r4-r7); reward = magnitude of the chosen arm."""
    choices = [1, 0, 0, 2, 2, 1, 0, 0]
    mags = [(1, 3, 0), (2, 3, 0), (2, 3, 0), (2, 3, 1), (2, 3, 1), (0, 4, 1), (3, 4, 1), (3, 4, 1)]
    rows = []
    for t, (choice, m) in enumerate(zip(choices, mags, strict=True)):
        rows.append((1, t // 4, t % 4, False, PORTS[choice], m[choice], dict(zip(PORTS, m,
                     strict=True))))  # fmt: skip
    return table_from_rows(rows)


# ---------------------------------------------------------------------------
# Held-out scores
# ---------------------------------------------------------------------------


def test_q_learner_score_matches_a_direct_computation_and_skips_session_starts():
    trials = q_toy_table()
    params = (0.5, 1.0, 0.3, 0.2)
    layout = sim.MouseLayout(list(PORTS), "B2", {"A1": 1, "B3": 3, "C2": 0}, [4, 4])
    policy, expected = sim.ValuePolicy(params, "magnitude", layout), 0.0
    for row in trials.itertuples():
        if row.trial_in_session == 0:
            policy.start_session()
        else:
            expected += np.log(policy.probabilities()[PORTS.index(row.choice)])
        policy.update(PORTS.index(row.choice), row.reward)  # a start still teaches the agent
    assert sm.SPECS[sm.Q_LEARNER].score(trials, params) == pytest.approx(expected)


def test_q_learner_fit_on_other_problems_is_scored_on_the_held_out_one():
    trials = simulate_mouse(sm.Q_LEARNER, seed=2, n_problems=2, trials=200)
    codes, _ = pd.factorize(trials["problem"])
    train, test = trials[codes == 0], trials[codes == 1]
    spec = sm.SPECS[sm.Q_LEARNER]
    params = spec.fit(train, 2, 7)
    seqs = sim.problem_sequences(test, "magnitude")
    direct = va.trial_log_probabilities(params, seqs)[seqs.mask & ~seqs.session_start].sum()
    assert spec.score(test, params) == pytest.approx(direct)
    assert np.allclose(params, va.fit(sim.problem_sequences(train, "magnitude"), 2, 7).parameters)


def test_wsls_score_is_the_null_agents_likelihood_without_the_session_starts():
    trials = simulate_mouse(sm.WSLS_MAGNITUDE, seed=1, n_problems=2)
    params = PARAMETERS[sm.WSLS_MAGNITUDE]
    n_starts = na.decisions(trials, params.decay).start.sum()
    expected = na.log_likelihood(trials, params) - n_starts * np.log(1 / 3)
    assert sm.SPECS[sm.WSLS_MAGNITUDE].score(trials, params) == pytest.approx(expected)


def test_heuristic_score_is_its_likelihood_without_the_session_starts():
    trials = simulate_mouse(sm.HEURISTIC_PLUS, seed=1, n_problems=2)
    params = PARAMETERS[sm.HEURISTIC_PLUS]
    expected = ha.log_likelihood(trials, params, include_starts=False)
    assert sm.SPECS[sm.HEURISTIC_PLUS].score(trials, params) == pytest.approx(expected)


def test_a_heuristic_with_no_extra_weights_scores_like_wsls_magnitude():
    trials = simulate_mouse(sm.WSLS_MAGNITUDE, seed=4, n_problems=2)
    wsls = na.fit_null_agents(trials, "wsls_magnitude")["m"]
    reduced = ha.HeuristicPlusParams(wsls.a_win, wsls.a_lose, wsls.b, wsls.w, 0.0, 0.0)
    assert sm.SPECS[sm.HEURISTIC_PLUS].score(trials, reduced) == pytest.approx(
        sm.SPECS[sm.WSLS_MAGNITUDE].score(trials, wsls)
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison():
    trials = simulate_mouse(sm.HEURISTIC_PLUS, seed=3, n_problems=3, sessions=2, trials=150)
    return trials, sm.cross_validate(trials, n_starts=1, seed=0)


def test_every_problem_is_held_out_once_for_every_model(comparison):
    trials, result = comparison
    scores = result.scores
    assert len(scores) == 3 * len(sm.MODELS)
    assert set(scores["model"]) == set(sm.MODELS)
    assert scores.groupby("model")["problem"].nunique().eq(3).all()
    assert set(scores["mouse"]) == {"m"}


def test_every_choice_with_a_previous_choice_is_scored_once(comparison):
    trials, result = comparison
    n_pairs = int((~na.decisions(trials, na.DEFAULT_DECAY).start).sum())
    totals = result.scores.groupby("model")["n_choices"].sum()
    assert (totals == n_pairs).all()


def test_scores_are_finite_log_likelihoods_below_zero(comparison):
    scores = comparison[1].scores
    assert np.isfinite(scores["held_out_log_likelihood"]).all()
    assert (scores["held_out_log_likelihood"] < 0).all()


def test_full_fit_parameters_are_listed_for_every_model(comparison):
    assert set(comparison[1].fits) == set(sm.MODELS)  # the fitted objects, for simulation
    parameters = comparison[1].parameters
    assert set(parameters["model"]) == set(sm.MODELS)
    heuristic = parameters[parameters["model"] == sm.HEURISTIC_PLUS].set_index("parameter")
    assert set(heuristic.index) == {"a_win", "a_lose", "b", "w", "v", "u", "decay"}
    q = parameters[parameters["model"] == sm.Q_LEARNER].set_index("parameter")
    assert set(q.index) == set(va.PARAMETER_NAMES)
    assert q["at_bound"].dtype == bool


def test_one_problem_cannot_be_cross_validated():
    trials = simulate_mouse(sm.WSLS_MAGNITUDE, n_problems=1)
    assert "1 problem" in sm.skip_reason(trials)
    with pytest.raises(ValueError, match="at least two problems"):
        sm.cross_validate(trials, n_starts=1)


def test_a_mouse_whose_training_folds_cannot_be_fitted_has_a_reason():
    rows = [(p, 0, t, False, PORTS[t % 3], 0, dict.fromkeys(PORTS, 0)) for p in (1, 2)
            for t in range(80)]  # fmt: skip
    never_rewarded = table_from_rows(rows)
    reason = sm.skip_reason(never_rewarded, min_trials=1)
    assert reason is not None and "rewarded" in reason


def test_a_fittable_mouse_has_no_skip_reason():
    assert sm.skip_reason(simulate_mouse(sm.HEURISTIC_PLUS, n_problems=3)) is None


def test_too_few_trials_has_a_reason():
    trials = simulate_mouse(sm.HEURISTIC_PLUS, n_problems=3, trials=100)
    assert "trials" in sm.skip_reason(trials, min_trials=10_000)


# ---------------------------------------------------------------------------
# Tables across mice
# ---------------------------------------------------------------------------


def _scores(nats_by_model, n_choices=100, mouse="m1"):
    """Two folds of ``n_choices / 2`` choices each, scoring *nats* per choice for every model."""
    half = n_choices // 2
    return pd.DataFrame(
        [
            dict(mouse=mouse, problem=k, model=model, n_choices=half,
                 held_out_log_likelihood=-nats * half)
            for model, nats in nats_by_model.items()
            for k in (1, 2)
        ]
    )  # fmt: skip


def _three(mouse, wsls, heuristic, q):
    nats = {sm.WSLS_MAGNITUDE: wsls, sm.HEURISTIC_PLUS: heuristic, sm.Q_LEARNER: q}
    return _scores(nats, 100, mouse)


def test_nats_per_choice_pools_the_folds_of_a_mouse():
    table = sm.nats_table(_scores({sm.WSLS_MAGNITUDE: 1.0, sm.Q_LEARNER: 0.9}))
    q = table[table["model"] == sm.Q_LEARNER].iloc[0]
    assert (q["n_choices"], q["nats_per_choice"]) == (100, pytest.approx(0.9))


def test_differences_are_the_first_model_minus_the_second_per_mouse():
    scores = pd.concat(
        [
            _three("a", 1.00, 0.95, 0.90),
            _three("b", 0.80, 0.78, 0.82),
        ]
    )  # fmt: skip
    differences = sm.difference_table(sm.nats_table(scores)).set_index("mouse")
    assert differences.loc["a", "heuristic_plus_minus_q_learner"] == pytest.approx(0.05)
    assert differences.loc["b", "heuristic_plus_minus_q_learner"] == pytest.approx(-0.04)
    assert differences.loc["a", "wsls_magnitude_minus_q_learner"] == pytest.approx(0.10)
    assert differences.loc["b", "heuristic_plus_minus_wsls_magnitude"] == pytest.approx(-0.02)


def test_the_winner_is_the_lowest_held_out_nats_and_wins_are_counted():
    scores = pd.concat(
        [
            _three("a", 1.00, 0.95, 0.90),
            _three("b", 0.80, 0.78, 0.82),
            _three("c", 0.80, 0.85, 0.82),
        ]
    )  # fmt: skip
    winners = sm.winners(sm.nats_table(scores))
    assert dict(zip(winners["mouse"], winners["winner"], strict=True)) == {
        "a": sm.Q_LEARNER, "b": sm.HEURISTIC_PLUS, "c": sm.WSLS_MAGNITUDE,
    }  # fmt: skip
    assert sm.win_counts(winners).to_dict() == {
        sm.WSLS_MAGNITUDE: 1, sm.HEURISTIC_PLUS: 1, sm.Q_LEARNER: 1,
    }  # fmt: skip


def test_difference_summary_gives_mean_se_and_tests_from_six_mice():
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "mouse": [f"m{i}" for i in range(8)],
            "heuristic_plus_minus_q_learner": rng.normal(-0.02, 0.005, 8),
            "wsls_magnitude_minus_q_learner": rng.normal(0.02, 0.005, 8),
            "heuristic_plus_minus_wsls_magnitude": rng.normal(-0.04, 0.005, 8),
        }
    )
    summary = sm.difference_summary(table).set_index("comparison")
    row = summary.loc["heuristic_plus_minus_q_learner"]
    assert row["n"] == 8 and row["mean"] == pytest.approx(table.iloc[:, 1].mean())
    assert row["p_wilcoxon"] < 0.05 and row["n_positive"] == 0
    few = sm.difference_summary(table.iloc[:5]).set_index("comparison")
    assert few["mean"].isna().all() and few["p_wilcoxon"].isna().all()


# ---------------------------------------------------------------------------
# Model and parameter recovery
# ---------------------------------------------------------------------------


def test_recovery_matrix_counts_and_fractions():
    winners = pd.DataFrame(
        dict(generator=[sm.Q_LEARNER] * 4 + [sm.HEURISTIC_PLUS] * 2,
             winner=[sm.Q_LEARNER, sm.Q_LEARNER, sm.Q_LEARNER, sm.HEURISTIC_PLUS,
                     sm.HEURISTIC_PLUS, sm.WSLS_MAGNITUDE])
    )  # fmt: skip
    counts, fractions = sm.recovery_matrix(winners)
    assert counts.loc[sm.Q_LEARNER, sm.Q_LEARNER] == 3
    assert fractions.loc[sm.Q_LEARNER, sm.HEURISTIC_PLUS] == pytest.approx(0.25)
    assert fractions.loc[sm.HEURISTIC_PLUS, sm.WSLS_MAGNITUDE] == pytest.approx(0.5)
    assert list(counts.columns) == list(sm.MODELS)
    assert (
        counts.loc[sm.WSLS_MAGNITUDE].sum() == 0 and fractions.loc[sm.WSLS_MAGNITUDE].isna().all()
    )


def test_parameter_recovery_summary_reports_bias_error_and_correlation():
    true = pd.DataFrame(dict(mouse=list("abcd"), parameter="v", value=[0.2, 0.5, 0.8, 1.1]))
    fitted = pd.DataFrame(dict(mouse=list("abcd"), parameter="v", value=[0.3, 0.5, 0.9, 1.0]))
    table = sm.parameter_recovery_table(true, fitted)
    assert list(table["error"].round(3)) == [0.1, 0.0, 0.1, -0.1]
    summary = sm.parameter_recovery_summary(table).set_index("parameter")
    assert summary.loc["v", "n"] == 4
    assert summary.loc["v", "bias"] == pytest.approx(0.025)
    assert summary.loc["v", "rmse"] == pytest.approx(np.sqrt(0.0075))
    assert summary.loc["v", "correlation"] > 0.95


def test_parameter_recovery_needs_the_same_mice_and_parameters_on_both_sides():
    true = pd.DataFrame(dict(mouse=["a", "b"], parameter="v", value=[0.2, 0.5]))
    with pytest.raises(ValueError, match="no fitted"):
        sm.parameter_recovery_table(true, true.iloc[:1])


@pytest.fixture(scope="module")
def recovered():
    """Two seeded mice per generating model, each fitted and compared with all three models."""
    rows = []
    for model in sm.MODELS:
        for seed in (1, 2):
            trials = simulate_mouse(model, seed=seed, n_problems=4, sessions=2, trials=250)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nats = sm.nats_table(sm.cross_validate(trials, n_starts=2, seed=0).scores)
            rows.append(sm.winners(nats).assign(generator=model))
    return pd.concat(rows, ignore_index=True)


def test_the_generating_model_wins_more_often_than_chance(recovered):
    correct = (recovered["winner"] == recovered["generator"]).mean()
    assert len(recovered) == 6
    assert correct >= 4 / 6  # chance is 1/3


def test_every_model_is_recovered_at_least_once(recovered):
    right = recovered[recovered["winner"] == recovered["generator"]]
    assert set(right["generator"]) == set(sm.MODELS)
