"""The long-lag choice kernel: decay fits, the per-mouse analysis and a known-kernel recovery."""

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.behavior_analysis.choice_logit import choice_kernel_timescale as ts
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.pipelines.choice_logit import choice_kernel_long as long
from tests.strategy_fixtures import RULES, layouts


def kernel_weights(amplitude, decay, n_lags=15, mouse="m", noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lags = np.arange(1, n_lags + 1)
    weight = amplitude * decay ** (lags - 1) + rng.normal(0, noise, n_lags)
    return pd.DataFrame(dict(mouse=mouse, feature=[f"choice_lag{k}" for k in lags],
                             group="choice", lag=lags.astype(float), weight=weight))  # fmt: skip


# ---------------------------------------------------------------------------
# Timescale of a kernel
# ---------------------------------------------------------------------------


def test_the_kernel_of_the_agent_is_alpha_times_the_geometric_series():
    lags = np.arange(1, 6)
    assert ts.true_kernel(0.2, 3.0, lags) == pytest.approx(3.0 * 0.2 * 0.8 ** (lags - 1))


def test_a_geometric_kernel_gives_back_its_amplitude_and_decay():
    fits = ts.fit_decay(kernel_weights(0.9, 0.8))
    row = fits.iloc[0]
    assert row["amplitude"] == pytest.approx(0.9, rel=1e-3)
    assert row["decay"] == pytest.approx(0.8, abs=1e-3)
    assert row["tau"] == pytest.approx(-1 / np.log(0.8), rel=1e-2)


def test_a_flat_kernel_has_a_decay_near_one_and_a_long_timescale():
    flat = kernel_weights(0.25, 1.0)
    row = ts.fit_decay(flat).iloc[0]
    assert row["decay"] > 0.97 and row["tau"] > 30


def test_a_sharp_kernel_has_a_short_timescale():
    sharp = kernel_weights(0.9, 0.2)
    assert ts.fit_decay(sharp).iloc[0]["tau"] < 1.5


def test_a_kernel_without_positive_persistence_has_no_decay():
    negative = kernel_weights(-0.5, 0.8)
    row = ts.fit_decay(negative).iloc[0]
    assert np.isnan(row["decay"]) and np.isnan(row["tau"])


def test_fit_decay_handles_every_mouse_of_the_table():
    both = pd.concat([kernel_weights(0.9, 0.8, mouse="a"), kernel_weights(0.5, 0.5, mouse="b")])
    fits = ts.fit_decay(both).set_index("mouse")
    assert fits.loc["a", "decay"] == pytest.approx(0.8, abs=1e-3)
    assert fits.loc["b", "decay"] == pytest.approx(0.5, abs=1e-3)


def test_fit_decay_needs_at_least_four_lags():
    with pytest.raises(ValueError, match="at least 4 needed"):
        ts.fit_decay(kernel_weights(0.9, 0.8, n_lags=3))


# ---------------------------------------------------------------------------
# One mouse's long-lag analysis, and recovery of a known kernel
# ---------------------------------------------------------------------------


def kernel_only_mouse(alpha_k, beta_k, mouse="m", seed=0, n_problems=4, trials=300):
    params = (0.5, 0.5, 1.0, alpha_k, 1e-3, beta_k, 0.0)  # no value: choices follow the kernel
    played = sim.simulate_policy_mouse(
        lambda layout: sim.KernelQPolicy(params, "magnitude", layout),
        layouts(n_problems, 3, trials, seed),
        RULES,
        np.random.default_rng([seed, 5]),
        mouse,
    )
    return drifting_trial_table(played)


@pytest.fixture(scope="module")
def analysis():
    trials = kernel_only_mouse(0.3, 2.5, seed=1, n_problems=6, trials=400)
    return long.analyze_long_kernel(trials, n_lags=8, min_trials=100)


def test_the_analysis_lists_lag_weights_the_tail_and_held_out_nats(analysis):
    choice = analysis.weights[analysis.weights["group"] == cl.CHOICE]
    assert sorted(choice["lag"]) == list(range(1, 9))
    tail = analysis.weights_tail[analysis.weights_tail["feature"] == cl.CHOICE_TAIL]
    assert len(tail) == 1
    assert set(analysis.comparison["model"]) == {"full", "full + choice tail"}
    assert (analysis.comparison["nats_per_choice"] > 0).all()
    assert analysis.skipped == {}


def test_a_known_decaying_kernel_is_recovered_not_flattened(analysis):
    choice = analysis.weights[analysis.weights["group"] == cl.CHOICE].set_index("lag")["weight"]
    lags = choice.index.to_numpy()
    truth = ts.true_kernel(0.3, 2.5, lags)
    assert np.corrcoef(choice.to_numpy(), truth)[0, 1] > 0.95
    assert choice.loc[1] > 2 * choice.loc[5]  # still sharp: lag 1 is at least twice lag 5
    fit = ts.fit_decay(analysis.weights).iloc[0]
    assert fit["decay"] == pytest.approx(0.7, abs=0.15)


def test_a_mouse_with_too_few_trials_is_skipped_with_its_reason():
    trials = kernel_only_mouse(0.3, 2.5, trials=40)
    with pytest.raises(ValueError, match="no mouse can be analyzed"):
        with pytest.warns(UserWarning, match="skipped"):
            long.analyze_long_kernel(trials, n_lags=4, min_trials=10_000)


def test_the_recovery_table_compares_fitted_with_generating_weights():
    trials = kernel_only_mouse(0.3, 2.5, seed=2, n_problems=6, trials=400)
    table = long.recover_choice_kernel(
        trials, alpha_ks=(0.3,), beta_k=2.5, rules=RULES, n_lags=8, seed=0, min_trials=100
    )
    assert list(table.columns) == ["alpha_k", "lag", "true", "fitted_mean", "fitted_se", "n_mice"]
    assert sorted(table["lag"]) == list(range(1, 9))
    assert table["n_mice"].eq(1).all()
    assert np.corrcoef(table["true"], table["fitted_mean"])[0, 1] > 0.9


def test_the_analysis_is_cached_per_mouse(tmp_path):
    trials = kernel_only_mouse(0.3, 2.5, seed=3, n_problems=3, trials=200)
    kwargs = dict(n_lags=4, min_trials=100, cache_dir=tmp_path)
    first = long.analyze_long_kernel(trials, **kwargs)
    assert any(tmp_path.rglob("*.pkl"))
    again = long.analyze_long_kernel(trials, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(first.weights, again.weights)
