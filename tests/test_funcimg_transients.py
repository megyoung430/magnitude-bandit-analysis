"""Transient check: private calcium transients make green skewed and one-sided; red is the noise."""

import numpy as np
import pytest

from src.funcimg_analysis.transients import (
    channel_comparison,
    count_events,
    most_skewed,
    robust_skewness,
    run_transients,
    transient_statistics,
)

N_FRAMES, N_CELLS, PERIOD = 4000, 40, 0.133
SETTINGS = {"detrend_window_s": 30.0, "z_threshold": 3.0, "min_gap_s": 1.0}


def noise(rng, n_cells=N_CELLS):
    """Symmetric noise, a little smoothed like a ΔF/F0 trace."""
    raw = rng.normal(size=(n_cells, N_FRAMES + 2))
    return (raw[:, :-2] + raw[:, 1:-1] + raw[:, 2:]) / np.sqrt(3)


def add_transients(traces, rng, n_planted, per_cell=12, amplitude=8.0, decay_frames=15):
    """Fast rise, slow exponential decay, on the first *n_planted* cells."""
    kernel = np.exp(-np.arange(6 * decay_frames) / decay_frames)
    out = traces.copy()
    for cell in range(n_planted):
        for start in rng.choice(N_FRAMES - len(kernel), per_cell, replace=False):
            out[cell, start : start + len(kernel)] += amplitude * kernel
    return out


def stats(traces, **overrides):
    return transient_statistics(traces, PERIOD, **{**SETTINGS, **overrides})


def test_symmetric_noise_has_no_skew_and_as_many_positive_as_negative_excursions():
    result = stats(noise(np.random.default_rng(0)))
    assert abs(np.median(result["skew"])) < 0.1 and abs(np.median(result["robust_skew"])) < 0.3
    total = result["rate_pos"].sum(), result["rate_neg"].sum()
    assert total[0] == pytest.approx(total[1], rel=0.35)


def test_planted_exponential_transients_make_cells_positively_skewed_with_excess_positive_events():
    rng = np.random.default_rng(1)
    green = add_transients(noise(rng), rng, n_planted=10)
    result = stats(green)
    assert result["skew"][:10].min() > 0.5 and np.median(result["skew"][10:]) < 0.15
    assert result["robust_skew"][:10].min() > 0.5
    assert (result["rate_pos"][:10] - result["rate_neg"][:10]).min() > 0.5  # per minute
    assert np.median(result["rate_pos"][10:] - result["rate_neg"][10:]) == pytest.approx(0, abs=0.3)


def test_pure_noise_red_next_to_transient_green_shows_the_difference_in_the_table():
    rng = np.random.default_rng(2)
    green = stats(add_transients(noise(rng), rng, n_planted=25))
    red = stats(noise(rng))
    table = channel_comparison(green, red)
    assert table["median skew green"] > table["median skew red"] + 0.2
    assert abs(table["median skew red"]) < 0.1
    assert 0.5 < table["fraction green above red 95th"] < 0.8  # 25 of 40 are planted, plus chance
    assert table["median positive rate green"] > table["median positive rate red"]
    assert table["median negative rate red"] == pytest.approx(
        table["median positive rate red"], rel=0.5
    )


def test_events_closer_than_the_minimum_gap_count_once():
    z = np.zeros(200)
    z[[50, 55]] = 6.0  # two crossings 5 frames (0.67 s) apart
    z[[100, 120]] = 6.0  # two crossings 20 frames (2.7 s) apart
    assert count_events(z, "positive", 3.0, gap_frames=8) == 3
    assert count_events(z, "positive", 3.0, gap_frames=4) == 4
    assert count_events(-z, "negative", 3.0, gap_frames=8) == 3


def test_a_stretch_above_the_threshold_is_one_crossing():
    z = np.zeros(100)
    z[40:60] = 5.0
    assert count_events(z, "positive", 3.0, gap_frames=1) == 1


def test_a_slow_drift_does_not_create_skew_or_events():
    rng = np.random.default_rng(3)
    drift = 5 * np.sin(np.linspace(0, 3 * np.pi, N_FRAMES))[None, :]
    plain, drifting = stats(noise(rng, 10)), stats(noise(np.random.default_rng(3), 10) + drift)
    assert np.abs(drifting["skew"] - plain["skew"]).max() < 0.15
    assert drifting["rate_pos"].sum() == pytest.approx(plain["rate_pos"].sum(), abs=5)


def test_a_cell_without_data_has_nan_statistics_and_is_not_ranked():
    traces = noise(np.random.default_rng(4), 5)
    traces[2] = np.nan
    result = stats(traces)
    assert np.isnan(result["skew"][2]) and np.isnan(result["rate_pos"][2])
    assert 2 not in most_skewed(result["skew"], 5)


def test_rates_are_per_minute_of_frames_with_data():
    z = np.random.default_rng(7).normal(size=(1, 1200))
    z[0, [100, 300, 500]] += 40.0  # 3 events far above the noise
    z[0, 600:] = np.nan  # half the run has no data
    result = transient_statistics(z, PERIOD, detrend_window_s=5.0, z_threshold=8.0, min_gap_s=1.0)
    assert result["rate_pos"][0] == pytest.approx(3 / (600 * PERIOD / 60), rel=0.01)


def test_most_skewed_orders_by_skewness_and_leaves_out_undefined_cells():
    skew = np.array([0.1, np.nan, 2.0, 1.0, -0.5])
    assert most_skewed(skew, 3).tolist() == [2, 3, 0]


def test_robust_skewness_ignores_a_single_huge_outlier_that_moves_the_classic_skewness():
    z = np.random.default_rng(5).normal(size=(1, 5000))
    spiked = z.copy()
    spiked[0, 100] = 80.0
    classic = abs(transient_statistics(spiked, PERIOD, **SETTINGS)["skew"][0])
    assert classic > 1.0 and abs(robust_skewness(spiked)[0]) < 0.2


def test_a_run_is_summarised_per_channel_with_only_the_cells_kept_in_both_for_pairing():
    rng = np.random.default_rng(6)
    green = add_transients(noise(rng), rng, n_planted=10)
    red = noise(rng)
    kept = np.ones(N_CELLS, bool)
    green_dff = np.where(kept[:, None], green, np.nan)
    red_dff = red.copy()
    red_dff[3] = np.nan
    run = {
        "time_s": 20.0 + PERIOD * np.arange(N_FRAMES),
        "channels": {
            "green": {"dff": green_dff, "kept": kept},
            "red": {"dff": red_dff, "kept": ~np.isnan(red_dff).all(axis=1)},
        },
    }
    out = run_transients(run, **SETTINGS)
    assert set(out) == {"green", "red", "period_s"} and len(out["green"]["skew"]) == N_CELLS
    assert np.isnan(out["red"]["skew"][3]) and np.isfinite(out["green"]["skew"][3])
    assert out["period_s"] == pytest.approx(PERIOD)


def test_the_table_says_how_many_rois_each_channel_kept_because_survivors_skew_bright():
    rng = np.random.default_rng(10)
    green = stats(add_transients(noise(rng), rng, n_planted=10))
    red_traces = noise(rng)
    red_traces[:12] = np.nan  # twelve ROIs have no usable red
    table = channel_comparison(green, stats(red_traces))
    assert table["n ROIs"] == N_CELLS
    assert table["n cells green"] == N_CELLS and table["n cells red"] == N_CELLS - 12
