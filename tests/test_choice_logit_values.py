"""The choice-logit pipeline: loading, per-mouse analysis, synthetic cohorts and the curves."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_logit as cl
from src.pipelines import choice_logit_values as pipe
from tests.all_cohorts_fixtures import synthetic_cohort

PARAMETER_SETS = [(0.3, 1.2, 0.0, 0.0), (0.3, 1.2, 0.8, 0.6)]


@pytest.fixture(scope="module")
def synthetic():
    return pipe.simulate_cohort(
        PARAMETER_SETS, mice_per_set=2, n_problems=3, sessions_per_problem=2,
        trials_per_session=250, seed=1,
    )  # fmt: skip


@pytest.fixture(scope="module")
def analysis(synthetic):
    return pipe.analyse_trial_table(synthetic.trials, n_lags=4)


def test_load_pools_the_drifting_problems_of_the_cohorts():
    cohort = synthetic_cohort(["m1", "m2"])
    table = pipe.load_drifting_trial_table([1], trials_by_cohort={1: cohort})
    assert sorted(table["mouse"].unique()) == ["m1", "m2"]
    assert sorted(table["problem"].unique()) == [(1, 1), (1, 2)]


def test_simulated_cohort_is_reproducible_and_labelled(synthetic):
    again = pipe.simulate_cohort(
        PARAMETER_SETS, mice_per_set=2, n_problems=3, sessions_per_problem=2,
        trials_per_session=250, seed=1,
    )  # fmt: skip
    pd.testing.assert_frame_equal(synthetic.trials, again.trials)
    assert list(synthetic.parameters["mouse"]) == ["sim01", "sim02", "sim03", "sim04"]
    assert list(synthetic.parameters.columns) == [
        "mouse", "learning_rate", "beta", "kappa", "plus_bias",
    ]  # fmt: skip
    assert synthetic.trials.groupby("mouse").size().eq(3 * 2 * 250).all()


def test_analysis_has_a_fit_and_validation_for_every_mouse(analysis, synthetic):
    mice = sorted(synthetic.trials["mouse"].unique())
    assert sorted(analysis.fits) == mice
    assert sorted(analysis.comparison["mouse"].unique()) == mice
    assert set(analysis.weights["group"]) == {"reward", "choice", "tower"}
    assert analysis.skipped == {}
    assert len(analysis.values) == len(analysis.meta) == len(analysis.chosen)
    assert np.isfinite(analysis.values).all()


def test_held_out_model_values_track_the_generating_values(analysis, synthetic):
    from scipy.stats import spearmanr

    from src.agents.choice_logit_truth import true_utilities

    for mouse, row in synthetic.parameters.set_index("mouse").iterrows():
        trials = synthetic.trials[synthetic.trials["mouse"] == mouse]
        _, q = true_utilities(trials, tuple(row))
        mine = (analysis.meta["mouse"] == mouse).to_numpy()
        centred = lambda a: a - a.mean(axis=1, keepdims=True)  # noqa: E731
        rho = spearmanr(centred(q).ravel(), centred(analysis.values[mine]).ravel()).statistic
        assert rho > 0.4, (mouse, rho)


def test_the_full_model_beats_chance_on_held_out_problems(analysis):
    full = analysis.comparison.query("model == 'full'")
    assert (full["gain_over_chance"] > 0.02).all()


def test_mice_with_a_single_problem_are_skipped_loudly(synthetic):
    one_problem = synthetic.trials.copy()
    first = one_problem["problem"].iloc[0]
    one_problem = one_problem[(one_problem["mouse"] != "sim01") | (one_problem["problem"] == first)]
    with pytest.warns(UserWarning, match="sim01"):
        result = pipe.analyse_trial_table(one_problem, n_lags=2)
    assert list(result.skipped) == ["sim01"]
    assert "sim01" not in result.fits


def test_no_analysable_mouse_is_an_error(synthetic):
    one = synthetic.trials[synthetic.trials["mouse"] == "sim01"]
    one = one[one["problem"] == one["problem"].iloc[0]]
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no mouse"):
        pipe.analyse_trial_table(one, n_lags=2)


def test_value_curves_cover_every_mouse(analysis, synthetic):
    n_mice = synthetic.trials["mouse"].nunique()
    curves = pipe.model_value_curves(analysis, n_bins=4)
    assert len(curves["best"]["best_val"].mice) == n_mice
    pairs = pipe.model_value_pairs(analysis, n_bins=4)
    assert set(pairs["mouse"]) == set(synthetic.trials["mouse"])
    assert pairs["stay"].between(0, 1).all()


def test_reference_weights_align_with_the_fitted_ones(analysis, synthetic):
    reference = pipe.reference_weights(analysis, synthetic)
    assert set(reference.columns) == {"mouse", "feature", "group", "lag", "weight"}
    fitted = analysis.weights.set_index(["mouse", "feature"]).index
    assert reference.set_index(["mouse", "feature"]).index.equals(fitted)
    # a stickier agent has the larger first choice-lag reference weight
    first = reference.query("feature == 'choice_lag1'").set_index("mouse")["weight"]
    assert first["sim03"] > first["sim01"]


def test_design_features_are_the_requested_lags(analysis):
    names = analysis.designs["sim01"].feature_names
    assert names[:4] == tuple(f"{cl.GROUP_PREFIX[cl.REWARD]}{k}" for k in range(1, 5))
