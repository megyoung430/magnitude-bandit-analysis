"""The regression-extensions pipeline: per-mouse variants, source tables, skipping and caching."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import choice_logit_extensions as ext
from src.pipelines import choice_logit_values as pipe
from src.pipelines import regression_extensions as rx
from tests.choice_logit_fixtures import simulate_logit_cohort, simulate_logit_trials

SETTINGS = dict(n_lags=2, tail_decay=None, l2=1.0)
SIZE = dict(n_problems=4, sessions_per_problem=2, trials_per_session=120)


@pytest.fixture(scope="module")
def cohort():
    return simulate_logit_cohort(3, 10, binary_weight=2.0, choice_weight=1.0, **SIZE)


@pytest.fixture(scope="module")
def source(cohort):
    return rx.analyse_extensions(cohort, **SETTINGS)


def test_every_mouse_gets_every_variant(source):
    comparison = source.comparison
    assert sorted(comparison["mouse"].unique()) == ["m1", "m2", "m3"]
    for _, rows in comparison.groupby("mouse"):
        assert list(rows["model"]) == list(ext.VARIANTS)
    assert (comparison["n_trials"] == 4 * 240).all()


def test_weights_are_those_of_the_believed_variant(source):
    features = set(source.weights["feature"])
    assert {"believed_value", "believed_unvisited", "reward_lag1", "choice_lag1"} <= features
    assert not any(f.startswith("binary_") for f in features)
    assert sorted(source.weights["mouse"].unique()) == ["m1", "m2", "m3"]


def test_trials_probabilities_and_mice_are_aligned(source, cohort):
    n = len(cohort)
    assert source.chosen.shape == (n,)
    assert source.mice.shape == (n,)
    assert set(source.probabilities) == set(rx.CALIBRATION_MODELS)
    for probabilities in source.probabilities.values():
        assert probabilities.shape == (n, 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert list(pd.unique(source.mice)) == ["m1", "m2", "m3"]


def test_settings_are_recorded(source):
    assert (source.n_lags, source.tail_decay, source.l2) == (2, None, 1.0)


def test_mice_that_generate_binary_choices_prefer_the_binary_coding(source):
    tables = rx.extension_tables(source)
    assert (tables.binary_pulse["preferred"] == "binary").all()
    assert set(tables.partition.columns) >= {"unique_reward", "shared_reward_choice"}
    assert set(tables.calibration) == set(rx.CALIBRATION_MODELS)
    assert set(tables.calibration_summary) == set(rx.CALIBRATION_MODELS)
    assert sorted(tables.believed["mouse"]) == ["m1", "m2", "m3"]


def test_design_believed_value_agrees_with_the_pipeline_believed_values(cohort):
    analysis = pipe.analyse_trial_table(cohort, **SETTINGS)
    expected = pipe.believed_values(analysis)
    parts = []
    for mouse in analysis.designs:
        design = cl.design_matrix(cohort[cohort["mouse"] == mouse], 2, None, extras=(cl.BELIEVED,))
        names = list(design.feature_names)
        parts.append(
            (
                design.X[:, :, names.index("believed_value")],
                design.X[:, :, names.index("believed_unvisited")],
            )
        )
    value = np.concatenate([p[0] for p in parts])
    unvisited = np.concatenate([p[1] for p in parts])
    np.testing.assert_array_equal(unvisited == 1, np.isnan(expected))
    np.testing.assert_array_equal(value[~np.isnan(expected)], expected[~np.isnan(expected)])
    assert (value[np.isnan(expected)] == 0).all()


def test_a_short_mouse_is_skipped_with_a_warning(cohort):
    short = simulate_logit_trials(
        "m9", 99, n_problems=4, sessions_per_problem=1, trials_per_session=10
    )
    with pytest.warns(UserWarning, match="m9 skipped"):
        result = rx.analyse_extensions(pd.concat([cohort, short]), **SETTINGS)
    assert "m9" in result.skipped
    assert "m9" not in set(result.comparison["mouse"])


def test_no_analysable_mouse_raises():
    short = simulate_logit_trials(
        "m9", 99, n_problems=4, sessions_per_problem=1, trials_per_session=10
    )
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no mouse can be analysed"):
        rx.analyse_extensions(short, **SETTINGS)


def test_the_cache_returns_the_same_result_and_is_written(cohort, tmp_path, source):
    first = rx.analyse_extensions(cohort, cache_dir=tmp_path, **SETTINGS)
    assert any(tmp_path.rglob("*.pkl"))
    again = rx.analyse_extensions(cohort, cache_dir=tmp_path, **SETTINGS)
    pd.testing.assert_frame_equal(first.comparison, again.comparison)
    pd.testing.assert_frame_equal(first.comparison, source.comparison)
