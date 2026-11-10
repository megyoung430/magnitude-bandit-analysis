"""The switch-regression pipeline: sources, model values, skipping, caching and summaries."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import switch_regression as sr
from src.pipelines import choice_logit_values as logit_pipe
from src.pipelines import switch_regression as pipe
from tests.switch_regression_fixtures import simulate_switch_mouse

GRID = (0.5, 0.9)
SIZE = dict(n_problems=4, sessions_per_problem=2, trials_per_session=250)


def _cohort(n_mice, seed, **kwargs):
    return pd.concat(
        [
            simulate_switch_mouse(f"m{i}", seed + i, value_weight=0.5, trace_weight=0.5,
                                  **{**SIZE, **kwargs})
            for i in range(1, n_mice + 1)
        ],
        ignore_index=True,
    )  # fmt: skip


@pytest.fixture(scope="module")
def cohort():
    return _cohort(3, 30)


@pytest.fixture(scope="module")
def analysis(cohort):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return logit_pipe.analyse_trial_table(cohort, n_lags=3, tail_decay=0.9, min_trials=100)


@pytest.fixture(scope="module")
def source(cohort, analysis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe.analyse_switch_source(cohort, analysis, trace_grid=GRID, min_switches=50)


def test_every_mouse_has_every_condition_and_value(source):
    values = source.values
    assert sorted(values["mouse"].unique()) == ["m1", "m2", "m3"]
    assert len(values) == 3 * len(sr.CONDITIONS) * len(sr.VALUES)
    assert set(values["value"]) == set(sr.VALUES)
    assert np.isfinite(values[["beta", "se", "gain"]].to_numpy()).all()


def test_the_settings_are_those_of_the_logit_analysis(source, analysis):
    assert (source.n_lags, source.l2) == (analysis.n_lags, analysis.l2)
    assert source.trace_grid == GRID
    assert set(source.decays["trace_decay"]) <= set(GRID)


def test_model_values_are_the_analysis_held_out_values(cohort, analysis, source):
    trials = cohort[cohort["mouse"] == "m2"]
    design = analysis.designs["m2"]
    model = sr.ModelValues(
        design.meta["problem"].to_numpy(), design.meta["trial"].to_numpy(), analysis.cv["m2"].values
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sr.analyse_mouse(
            trials, analysis.n_lags, GRID, analysis.l2, min_switches=50, model_values=model
        )
    expected = sr.value_table(direct.scores, direct.weights)
    mine = source.values[source.values["mouse"] == "m2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, expected)


def test_switch_counts_and_history_weights_are_kept(source):
    counts = source.counts
    assert set(counts["condition"]) == set(sr.CONDITIONS)
    assert (counts["n_switches"] >= 50).all()
    features = set(source.history["feature"])
    assert {"choice_lag2", "choice_lag3", "trace", "recency", "never_chosen"} == features


def test_without_an_analysis_there_is_no_model_value(cohort):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipe.analyse_switch_source(cohort, None, trace_grid=(0.9,), min_switches=50)
    assert set(result.values["value"]) == {"true", "believed"}
    assert result.n_lags == sr.cl.DEFAULT_LAGS


def test_a_mouse_with_too_few_switches_is_skipped_with_a_warning(cohort):
    short = simulate_switch_mouse(
        "m9", 99, value_weight=0.0, n_problems=4, sessions_per_problem=1, trials_per_session=40
    )
    with pytest.warns(UserWarning, match="m9 skipped: .*switches after"):
        result = pipe.analyse_switch_source(
            pd.concat([cohort, short]), None, trace_grid=(0.9,), min_switches=50
        )
    assert "m9" in result.skipped
    assert "m9" not in set(result.values["mouse"])


def test_no_mouse_with_enough_switches_raises(cohort):
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no mouse has enough"):
        pipe.analyse_switch_source(cohort, None, trace_grid=(0.9,), min_switches=10_000)


def test_mice_the_logit_skipped_are_not_analyzed(cohort, analysis):
    extra = simulate_switch_mouse("m8", 8, value_weight=0.0, **SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipe.analyse_switch_source(
            pd.concat([cohort, extra]), analysis, trace_grid=(0.9,), min_switches=50
        )
    assert "m8" not in set(result.values["mouse"])
    assert "not in the logit analysis" in result.skipped["m8"]


def test_the_cache_returns_the_same_result_and_is_written(cohort, analysis, source, tmp_path):
    kwargs = dict(trace_grid=GRID, min_switches=50, cache_dir=tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = pipe.analyse_switch_source(cohort, analysis, **kwargs)
        assert any(tmp_path.rglob("*.pkl"))
        again = pipe.analyse_switch_source(cohort, analysis, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(first.values, again.values)
    pd.testing.assert_frame_equal(first.values, source.values)


def test_changing_a_setting_misses_the_cache(cohort, tmp_path):
    cached = pipe.mouse_analyser(tmp_path)
    trials = cohort[cohort["mouse"] == "m1"].reset_index(drop=True)
    arguments = (trials, 3, (0.9,), 1.0, sr.RECENCY_CAP, 50, None, pipe.code_version())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cached(*arguments)
    assert cached.check_call_in_cache(*arguments)
    other_cap = (*arguments[:4], 50, *arguments[5:])
    other_grid = (arguments[0], 3, (0.5,), *arguments[3:])
    assert not cached.check_call_in_cache(*other_cap)
    assert not cached.check_call_in_cache(*other_grid)


def test_the_code_version_follows_the_module_sources(monkeypatch):
    before = pipe.code_version()
    assert before == pipe.code_version()
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda module: real(module) + "# edit\n")
    assert pipe.code_version() != before


def test_warnings_of_the_workers_reach_the_parent_also_from_the_cache(tmp_path):
    # problems 1 and 2 of the simulated layouts share no tower
    table = _cohort(2, 40, n_problems=2, trials_per_session=400)
    for _ in range(2):  # the second run is served from the cache
        with pytest.warns(UserWarning, match="share no tower"):
            pipe.analyse_switch_source(
                table, None, trace_grid=(0.9,), min_switches=50, n_jobs=2, cache_dir=tmp_path
            )


def test_sources_are_analyzed_in_order_with_their_own_analysis(cohort, analysis):
    inputs = {"A": (cohort, analysis), "B": (cohort, None)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sources = pipe.analyse_switch_sources(inputs, trace_grid=(0.9,), min_switches=50)
    assert list(sources) == ["A", "B"]
    assert "model" in set(sources["A"].values["value"])
    assert "model" not in set(sources["B"].values["value"])


def test_source_summaries_and_per_mouse_table(source):
    sources = {"Mice": source, "Q-Learner": source}
    summary = pipe.source_summaries(sources)
    assert list(summary["source"].unique()) == ["Mice", "Q-Learner"]
    assert (summary["n"] == 3).all()
    assert summary["mean"].isna().all()  # three mice: no mean, error or test
    table = pipe.per_mouse_table(sources)
    assert {"source", "mouse", "condition", "value", "beta", "se", "gain", "n_switches",
            "trace_decay"} <= set(table.columns)  # fmt: skip
    assert len(table) == 2 * len(source.values)
