"""The regression-extensions pipeline: per-mouse variants, source tables, skipping and caching."""

import inspect
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


def test_no_analyzable_mouse_raises():
    short = simulate_logit_trials(
        "m9", 99, n_problems=4, sessions_per_problem=1, trials_per_session=10
    )
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no mouse can be analyzed"):
        rx.analyse_extensions(short, **SETTINGS)


def test_the_cache_returns_the_same_result_and_is_written(cohort, tmp_path, source):
    first = rx.analyse_extensions(cohort, cache_dir=tmp_path, **SETTINGS)
    assert any(tmp_path.rglob("*.pkl"))
    again = rx.analyse_extensions(cohort, cache_dir=tmp_path, **SETTINGS)
    pd.testing.assert_frame_equal(first.comparison, again.comparison)
    pd.testing.assert_frame_equal(first.comparison, source.comparison)


def mouse_arguments(cohort, variants=None):
    trials = cohort[cohort["mouse"] == "m1"].reset_index(drop=True)
    return (trials, 2, None, 1.0, variants or ext.VARIANTS, cl.EXTRAS, (None,), rx.code_version())


def test_changing_a_variant_definition_misses_the_cache(cohort, tmp_path):
    cached = rx.mouse_analyser(tmp_path)
    arguments = mouse_arguments(cohort)
    cached(*arguments)
    assert cached.check_call_in_cache(*arguments)
    changed = {**ext.VARIANTS, "no tower": (cl.REWARD,)}
    assert not cached.check_call_in_cache(*mouse_arguments(cohort, changed))


def test_changing_the_extras_or_the_binary_decays_misses_the_cache(cohort, tmp_path):
    cached = rx.mouse_analyser(tmp_path)
    arguments = mouse_arguments(cohort)
    cached(*arguments)
    fewer_extras = (*arguments[:5], (cl.BELIEVED,), *arguments[6:])
    other_decays = (*arguments[:6], (None, None), arguments[7])
    assert not cached.check_call_in_cache(*fewer_extras)
    assert not cached.check_call_in_cache(*other_decays)


def test_the_code_version_follows_the_logit_module_sources(monkeypatch):
    before = rx.code_version()
    assert before == rx.code_version()
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda module: real(module) + "# edit\n")
    assert rx.code_version() != before


def test_worker_rejects_variants_without_the_models_it_needs(cohort):
    arguments = list(mouse_arguments(cohort))
    arguments[4] = {"full": ext.VARIANTS["full"]}
    with pytest.raises(ValueError, match="full \\+ believed"):
        rx.analyse_mouse_extensions(*arguments)


def test_warnings_of_the_workers_reach_the_parent_also_from_the_cache(tmp_path):
    # problems 1 and 2 of the simulated layouts share no tower
    table = simulate_logit_cohort(
        2, 20, n_problems=2, sessions_per_problem=2, trials_per_session=100
    )
    for _ in range(2):  # the second run is served from the cache
        with pytest.warns(UserWarning, match="share no tower"):
            rx.analyse_extensions(table, **SETTINGS, n_jobs=2, cache_dir=tmp_path)


# --- the binary coding's own tail decay ---------------------------------------------------------


def scores_with(ll_binary: float, n=100):
    return pd.DataFrame(
        {
            "mouse": "m1",
            "problem": [1, 2],
            "model": ["binary", "binary"],
            "n_trials": [n, n],
            "held_out_log_likelihood": [ll_binary, ll_binary],
        }
    )


def test_the_binary_decay_with_the_fewest_pooled_nats_is_chosen():
    per_mouse = [
        {0.5: scores_with(-40.0), 0.9: scores_with(-35.0)},
        {0.5: scores_with(-30.0), 0.9: scores_with(-32.5)},
    ]
    decay, table = rx.select_binary_decay(per_mouse)
    assert decay == 0.9  # pooled nats 0.5: 140/400 = 0.35, 0.9: 135/400 = 0.3375
    assert list(table["binary_tail_decay"]) == [0.5, 0.9]
    assert table["nats_per_choice"].tolist() == pytest.approx([0.35, 0.3375])


def test_without_a_tail_there_is_no_binary_decay_to_choose():
    decay, table = rx.select_binary_decay([{None: scores_with(-70.0)}])
    assert decay is None and len(table) == 1


def test_the_analysis_chooses_the_binary_decay_by_the_binary_models_own_likelihood(cohort):
    result = rx.analyse_extensions(cohort, 2, 0.9, 1.0, binary_tail_grid=(0.5, 0.9))
    table = result.binary_decay_table
    best = table.loc[table["nats_per_choice"].idxmin(), "binary_tail_decay"]
    assert result.binary_tail_decay == best
    binary = result.comparison.query("model == 'binary'")
    pooled = np.average(binary["nats_per_choice"], weights=binary["n_trials"])
    assert pooled == pytest.approx(table["nats_per_choice"].min())


# --- several sources ---------------------------------------------------------------------------

MICE = [f"m{i}" for i in range(1, 7)]


def analysis_stub(n_lags=2, tail_decay=None, l2=1.0):
    return SimpleNamespace(n_lags=n_lags, tail_decay=tail_decay, l2=l2)


def test_comparison_inputs_list_the_reference_then_each_null_agent(cohort):
    kinds = ("wsls", "wsls_recency", "wsls_magnitude", "wsls_graded")
    comparison = SimpleNamespace(
        controls=SimpleNamespace(cohorts={k: SimpleNamespace(trials=cohort) for k in kinds}),
        analyses={
            label: analysis_stub()
            for label in ("Mice", "WSLS", "WSLS (Recency)", "WSLS (Magnitude)", "WSLS (Graded)")
        },
        extra_cohorts=None,
    )
    inputs = rx.comparison_inputs("Mice", cohort, comparison)
    assert list(inputs) == ["Mice", "WSLS", "WSLS (Recency)", "WSLS (Magnitude)", "WSLS (Graded)"]
    assert inputs["Mice"][0] is cohort
    assert inputs["WSLS"][1] is comparison.analyses["WSLS"]


def test_comparison_inputs_end_with_the_extra_cohorts(cohort):
    kinds = ("wsls", "wsls_recency", "wsls_magnitude", "wsls_graded")
    labels = (
        "Mice",
        "WSLS",
        "WSLS (Recency)",
        "WSLS (Magnitude)",
        "WSLS (Graded)",
        "Q-Learner (Fitted)",
    )
    fitted = SimpleNamespace(trials=cohort.iloc[:10])
    comparison = SimpleNamespace(
        controls=SimpleNamespace(cohorts={k: SimpleNamespace(trials=cohort) for k in kinds}),
        analyses={label: analysis_stub() for label in labels},
        extra_cohorts={"Q-Learner (Fitted)": fitted},
    )
    inputs = rx.comparison_inputs("Mice", cohort, comparison)
    assert list(inputs) == list(labels)
    assert inputs["Q-Learner (Fitted)"][0] is fitted.trials
    assert inputs["Q-Learner (Fitted)"][1] is comparison.analyses["Q-Learner (Fitted)"]


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
        m: pd.DataFrame(
            {
                "mouse": MICE,
                "mean_abs_error": 0.02,
                "weighted_abs_error": 0.01,
                "noise_floor_mean_abs_error": 0.015,
                "noise_floor_weighted_abs_error": 0.008,
                "nats_per_choice": 0.8,
            }
        )
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
    floor = summaries.calibration.query(
        "model == 'full' and term == 'noise_floor_mean_abs_error'"
    ).iloc[0]
    assert floor["mean"] == pytest.approx(0.015)
    nats = summaries.calibration.query("model == 'choice only' and term == 'nats_per_choice'")
    assert nats.iloc[0]["mean"] == pytest.approx(0.8)


def small_tables(n):
    mice = MICE[:n]
    believed = pd.DataFrame(
        {
            "mouse": mice,
            "believed_weight": 0.2,
            "unvisited_weight": -0.1,
            "gain_believed": 0.01,
            "gain_no_reward_believed": -0.01,
        }
    )
    binary = pd.DataFrame(
        {"mouse": mice, "gain_binary": 0.02, "gain_binary_pulse": 0.0, "preferred": "binary"}
    )
    partition = pd.DataFrame(
        {
            "mouse": mice,
            "unique_tower": 0.01,
            "unique_reward": 0.03,
            "unique_choice": 0.06,
            "shared_reward_choice": 0.02,
        }
    )
    summary = {
        m: pd.DataFrame(
            {
                "mouse": mice,
                "mean_abs_error": 0.02,
                "weighted_abs_error": 0.01,
                "noise_floor_mean_abs_error": 0.015,
                "noise_floor_weighted_abs_error": 0.008,
                "nats_per_choice": 0.8,
            }
        )
        for m in rx.CALIBRATION_MODELS
    }
    return rx.ExtensionTables(believed, binary, partition, {}, summary)


def test_summaries_of_fewer_than_six_mice_have_no_mean_error_or_test():
    summaries = rx.source_summaries({"Mice": small_tables(5)})
    for table in (summaries.gains, summaries.weights, summaries.partition, summaries.calibration):
        assert (table["n"] == 5).all()
        assert table["mean"].isna().all() and table["se"].isna().all()
    assert summaries.gains["p_wilcoxon"].isna().all()


def test_per_mouse_summary_lists_every_mouse_of_every_source_with_all_terms():
    table = rx.per_mouse_summary({"Mice": small_tables(2), "WSLS": small_tables(3)})
    assert list(table["source"]) == ["Mice", "Mice", "WSLS", "WSLS", "WSLS"]
    assert {
        "gain_believed", "gain_binary", "unique_reward", "shared_reward_choice",
        "mean_abs_error_full", "noise_floor_mean_abs_error_full", "nats_per_choice_choice_only",
    } <= set(table.columns)  # fmt: skip
