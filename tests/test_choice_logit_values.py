"""The choice-logit pipeline: loading, per-mouse analysis, simulated cohorts and the curves."""

import numpy as np
import pandas as pd
import pytest

from src.agents.choice_logit_truth import true_utilities
from src.behavior_analysis import model_value_psychometrics as mvp
from src.pipelines import choice_logit_values as pipe
from tests.all_cohorts_fixtures import synthetic_cohort

PARAMETER_SETS = [(0.3, 1.2, 0.0, 0.0), (0.3, 1.2, 0.8, 0.6)]
SIZE = dict(n_problems=6, sessions_per_problem=2, trials_per_session=150)


@pytest.fixture(scope="module")
def synthetic():
    return pipe.simulate_cohort(PARAMETER_SETS, mice_per_set=2, seed=1, **SIZE)


@pytest.fixture(scope="module")
def analysis(synthetic):
    return pipe.analyse_trial_table(synthetic.trials, n_lags=4, tail_decay=0.9)


def test_load_pools_the_drifting_problems_of_the_cohorts():
    cohort = synthetic_cohort(["m1", "m2"])
    table = pipe.load_drifting_trial_table([1], trials_by_cohort={1: cohort})
    assert sorted(table["mouse"].unique()) == ["m1", "m2"]
    assert sorted(table["problem"].unique()) == [(1, 1), (1, 2)]


def test_load_problems_keys_them_by_cohort_and_number():
    cohort = synthetic_cohort(["m1", "m2"])
    problems = pipe.load_drifting_problems([1], trials_by_cohort={1: cohort})
    assert sorted(problems) == [(1, 1), (1, 2)]
    assert sorted(problems[(1, 1)]) == ["m1", "m2"]


def test_load_rejects_a_repeated_cohort():
    with pytest.raises(ValueError, match="once each"):
        pipe.load_drifting_problems([1, 1], trials_by_cohort={1: {}})


def test_simulated_cohort_is_reproducible_and_labelled(synthetic):
    again = pipe.simulate_cohort(PARAMETER_SETS, mice_per_set=2, seed=1, **SIZE)
    pd.testing.assert_frame_equal(synthetic.trials, again.trials)
    assert list(synthetic.parameters["mouse"]) == ["sim01", "sim02", "sim03", "sim04"]
    assert list(synthetic.parameters.columns) == [
        "mouse", "learning_rate", "beta", "kappa", "plus_bias",
    ]  # fmt: skip
    assert synthetic.trials.groupby("mouse").size().eq(6 * 2 * 150).all()


def test_every_simulated_tower_is_offered_in_at_least_two_problems(synthetic):
    one_mouse = synthetic.trials[synthetic.trials["mouse"] == "sim01"]
    offered = one_mouse.groupby("problem")["magnitudes"].first().map(lambda m: set(m))
    counts = pd.Series([t for towers in offered for t in towers]).value_counts()
    assert counts.min() >= 2


def test_the_start_values_are_not_tied_to_a_tower(synthetic):
    one_mouse = synthetic.trials[synthetic.trials["mouse"] == "sim01"]
    first = one_mouse.sort_values("trial").groupby("problem")["magnitudes"].first()
    best_port_rank = [sorted(m).index(max(m, key=m.get)) for m in first]
    assert len(set(best_port_rank)) > 1  # not always the same port rank


def test_analysis_has_a_fit_and_validation_for_every_mouse(analysis, synthetic):
    mice = sorted(synthetic.trials["mouse"].unique())
    assert sorted(analysis.fits) == mice
    assert sorted(analysis.comparison["mouse"].unique()) == mice
    assert set(analysis.weights["group"]) == {"reward", "choice", "tower"}
    assert "reward_tail" in set(analysis.weights["feature"])
    assert analysis.skipped == {}
    assert len(analysis.values) == len(analysis.meta) == len(analysis.chosen)
    assert analysis.baseline.shape == analysis.values.shape
    np.testing.assert_allclose(analysis.baseline.sum(axis=1), 1.0)
    assert np.isfinite(analysis.values).all()


def test_the_tail_leaves_fewer_arms_without_a_value(synthetic):
    plain = pipe.analyse_trial_table(synthetic.trials, n_lags=4, tail_decay=None)
    tailed = pipe.analyse_trial_table(synthetic.trials, n_lags=4, tail_decay=0.9)
    assert mvp.zero_fraction(tailed.values) < mvp.zero_fraction(plain.values)


def test_held_out_model_values_track_the_generating_values(analysis, synthetic):
    for mouse, row in synthetic.parameters.set_index("mouse").iterrows():
        trials = synthetic.trials[synthetic.trials["mouse"] == mouse]
        _, q = true_utilities(trials, tuple(row))
        mine = (analysis.meta["mouse"] == mouse).to_numpy()
        rho = pipe.centred_spearman(q, analysis.values[mine])
        assert rho > 0.4, (mouse, rho)


def test_the_full_model_beats_chance_on_held_out_problems(analysis):
    full = analysis.comparison.query("model == 'full'")
    assert (full["gain_over_chance"] > 0.02).all()


def test_mice_with_a_single_problem_are_skipped_loudly(synthetic):
    first = synthetic.trials["problem"].iloc[0]
    one_problem = synthetic.trials[
        (synthetic.trials["mouse"] != "sim01") | (synthetic.trials["problem"] == first)
    ]
    with pytest.warns(UserWarning, match="sim01"):
        result = pipe.analyse_trial_table(one_problem, n_lags=2, tail_decay=None)
    assert list(result.skipped) == ["sim01"]
    assert "sim01" not in result.fits


def test_no_analysable_mouse_is_an_error(synthetic):
    one = synthetic.trials[synthetic.trials["mouse"] == "sim01"]
    one = one[one["problem"] == one["problem"].iloc[0]]
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no mouse"):
        pipe.analyse_trial_table(one, n_lags=2)


def test_tail_decay_is_chosen_by_held_out_likelihood(synthetic):
    selection = pipe.select_tail_decay(synthetic.trials, grid=(0.5, 0.9), n_lags=3)
    assert selection.decay in (0.5, 0.9)
    table = selection.table.set_index("tail_decay")
    assert len(table) == 3  # the two decays and "no tail" (NaN)
    tailed = table.drop(index=np.nan)["nats_per_choice"]
    assert table.loc[selection.decay, "nats_per_choice"] == tailed.min()
    assert selection.analysis.fits["sim01"].feature_names.count("reward_tail") == 1


def test_value_curves_cover_every_mouse_and_have_a_baseline(analysis, synthetic):
    n_mice = synthetic.trials["mouse"].nunique()
    curves, baselines = pipe.value_curves(analysis, n_bins=4)
    assert len(curves["best"]["best_val"].mice) == n_mice
    assert baselines["best"]["best_val"].x.tolist() == curves["best"]["best_val"].x.tolist()
    pairs = pipe.value_pairs(analysis)
    assert set(pairs["mouse"]) == set(synthetic.trials["mouse"])
    assert pairs["stay"].between(0, 1).all() and pairs["base_stay"].between(0, 1).all()


def test_task_and_believed_values_are_aligned_with_the_trials(analysis, synthetic):
    task = pipe.task_values(analysis)
    assert task.shape == analysis.values.shape
    # the chosen arm's task value is the reward the trial paid
    rows = np.arange(len(task))
    np.testing.assert_array_equal(task[rows, analysis.chosen], analysis.meta["reward"].to_numpy())
    believed = pipe.believed_values(analysis)
    assert np.isnan(believed[0]).all()  # nothing seen at a problem's first trial
    assert np.isfinite(believed).any()
    first = analysis.meta.groupby(["mouse", "problem"], sort=False).head(1).index
    assert np.isnan(believed[first]).all()


def test_believed_value_is_the_last_reward_paid_by_the_arm(analysis):
    believed = pipe.believed_values(analysis)
    chosen, reward = analysis.chosen, analysis.meta["reward"].to_numpy()
    # second trial of the first problem: only the first trial's chosen arm has a value
    assert np.isfinite(believed[1]).sum() == 1
    assert believed[1, chosen[0]] == reward[0]


def test_agreement_with_the_generating_values_has_a_believed_value_baseline(analysis, synthetic):
    table = pipe.value_agreement(analysis, synthetic)
    assert set(table.columns) == {"mouse", "model_value", "believed_value"}
    assert len(table) == synthetic.trials["mouse"].nunique()
    # the model value beats "the last reward each arm paid" on average, not for every agent
    # (the stickiest agent's Q is close to its last reward), and is never much worse
    assert table["model_value"].mean() > table["believed_value"].mean()
    assert (table["model_value"] - table["believed_value"]).min() > -0.05


def test_centred_spearman_ignores_a_per_trial_offset():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    assert pipe.centred_spearman(a, a + rng.normal(size=(50, 1))) == pytest.approx(1.0)


def test_reference_weights_align_with_the_fitted_ones(analysis, synthetic):
    reference = pipe.reference_weights(analysis, synthetic)
    assert set(reference.columns) == {"mouse", "feature", "group", "lag", "weight"}
    fitted = analysis.weights.set_index(["mouse", "feature"]).index
    assert reference.set_index(["mouse", "feature"]).index.equals(fitted)
    first = reference.query("feature == 'choice_lag1'").set_index("mouse")["weight"]
    assert first["sim03"] > first["sim01"]  # a stickier agent has the larger choice-lag-1 weight


def test_kernel_comparison_lists_bounds_fit_and_reference(analysis, synthetic):
    table = pipe.kernel_comparison(analysis, synthetic)
    assert list(table.columns) == ["mouse", "lag", "lower", "upper", "fitted", "reference"]
    assert set(table["lag"]) == {1.0, 2.0, 3.0, 4.0}
    assert len(table) == 4 * len(analysis.designs)


def test_a_value_blind_sticky_agent_shows_no_effect_beyond_its_baseline():
    blind = pipe.simulate_cohort([(0.3, 0.0, 2.5, 0.0)], mice_per_set=3, seed=5, **SIZE)
    qlearner = pipe.simulate_cohort([(0.3, 2.0, 1.0, 0.0)], mice_per_set=3, seed=5, **SIZE)
    effects = {}
    for name, cohort in (("blind", blind), ("q", qlearner)):
        result = pipe.analyse_trial_table(cohort.trials, n_lags=5, tail_decay=0.9)
        curves, baselines = pipe.value_curves(result, n_bins=5)
        best = curves["best"]["best_val"].mean
        effects[name] = (best[-1] - best[0], mvp.value_effect(curves, baselines))
    raw_blind, corrected_blind = effects["blind"]
    raw_q, corrected_q = effects["q"]
    assert raw_blind > 0.15  # the raw curve rises: recently chosen arms are the ones with a value
    assert abs(corrected_blind) < 0.08  # but nothing beyond what stickiness predicts
    # a value-driven agent keeps a clear effect; the baseline-corrected number is smaller than
    # the raw rise because the held-out baseline absorbs some of what the value explains
    assert corrected_q > 0.08
    assert corrected_q - corrected_blind > 0.08
    assert raw_q > 0.15
