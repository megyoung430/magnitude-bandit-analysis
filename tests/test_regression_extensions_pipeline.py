"""The regression-extensions pipeline: per-mouse variants, source tables, skipping and caching."""

from types import SimpleNamespace

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


# --- several sources ---------------------------------------------------------------------------

MICE = [f"m{i}" for i in range(1, 7)]


def analysis_stub(n_lags=2, tail_decay=None, l2=1.0):
    return SimpleNamespace(n_lags=n_lags, tail_decay=tail_decay, l2=l2)


def test_comparison_inputs_list_the_reference_then_each_null_agent(cohort):
    kinds = ("wsls", "wsls_recency", "wsls_magnitude")
    comparison = SimpleNamespace(
        controls=SimpleNamespace(cohorts={k: SimpleNamespace(trials=cohort) for k in kinds}),
        analyses={
            label: analysis_stub()
            for label in ("Mice", "WSLS", "WSLS (Recency)", "WSLS (Magnitude)")
        },
    )
    inputs = rx.comparison_inputs("Mice", cohort, comparison)
    assert list(inputs) == ["Mice", "WSLS", "WSLS (Recency)", "WSLS (Magnitude)"]
    assert inputs["Mice"][0] is cohort
    assert inputs["WSLS"][1] is comparison.analyses["WSLS"]


def test_analyse_sources_uses_each_sources_logit_settings(cohort):
    inputs = {"A": (cohort, analysis_stub(n_lags=2)), "B": (cohort, analysis_stub(n_lags=3))}
    sources = rx.analyse_sources(inputs)
    assert list(sources) == ["A", "B"]
    assert (sources["A"].n_lags, sources["B"].n_lags) == (2, 3)


def test_source_summaries_report_mean_se_and_tests_per_source():
    steps = np.arange(1, 7) * 0.01
    believed = pd.DataFrame(
        {
            "mouse": MICE,
            "believed_weight": 0.2,
            "unvisited_weight": -0.1,
            "gain_believed": steps,
            "gain_no_reward_believed": -steps,
        }
    )
    binary = pd.DataFrame({"mouse": MICE, "gain_binary": steps, "gain_binary_pulse": 0.0})
    partition = pd.DataFrame(
        {
            "mouse": MICE,
            "unique_tower": 0.01,
            "unique_reward": 0.03,
            "unique_choice": 0.06,
            "shared_reward_choice": 0.02,
        }
    )
    summary = {
        m: pd.DataFrame({"mouse": MICE, "mean_abs_error": 0.02, "weighted_abs_error": 0.01})
        for m in rx.CALIBRATION_MODELS
    }
    tables = {
        label: rx.ExtensionTables(believed, binary, partition, {}, summary)
        for label in ("Mice", "WSLS")
    }
    summaries = rx.source_summaries(tables)
    gain = summaries.gains.query("source == 'Mice' and term == 'gain_believed'").iloc[0]
    assert gain["n"] == 6 and gain["mean"] == pytest.approx(0.035)
    assert gain["p_wilcoxon"] == pytest.approx(2 / 64)
    assert set(summaries.gains["source"]) == {"Mice", "WSLS"}
    assert set(summaries.gains["term"]) == {
        "gain_believed",
        "gain_no_reward_believed",
        "gain_binary",
        "gain_binary_pulse",
    }
    weight = summaries.weights.query("source == 'WSLS' and term == 'unvisited_weight'").iloc[0]
    assert weight["mean"] == pytest.approx(-0.1)
    shared = summaries.partition.query("term == 'shared_reward_choice'").iloc[0]
    assert shared["mean"] == pytest.approx(0.02)
    full = summaries.calibration.query("model == 'full' and term == 'mean_abs_error'").iloc[0]
    assert full["mean"] == pytest.approx(0.02)
