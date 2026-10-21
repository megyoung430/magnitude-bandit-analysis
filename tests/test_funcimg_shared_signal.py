"""Shared vs cell-specific signal: detrending, correlations, partial r, nulls, residuals (pure)."""

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from src.funcimg_analysis.shared_signal import (
    detrend,
    detrended_z,
    fit_background,
    leave_one_out_mean,
    mean_response,
    null_partial_r,
    null_percentile,
    null_r,
    paired_comparison,
    partial_row_correlation,
    remove_background,
    row_correlation,
    running_percentile,
    shared_signal_stats,
    shift_offsets,
    summary_row,
)

N_FRAMES = 3000


def smooth_noise(rng, n_frames=N_FRAMES, width=15):
    """Slowly varying noise (autocorrelated like a fluorescence trace), unit variance."""
    raw = rng.normal(size=n_frames + width)
    out = np.convolve(raw, np.ones(width) / np.sqrt(width), mode="valid")[:n_frames]
    return (out - out.mean()) / out.std()


def make_cells(rng, n_cells=12, field_weight=1.0, common_weight=0.0, private_weight=1.0):
    """Cells = field signal + an extra common signal + private noise; and the field itself."""
    field, common = smooth_noise(rng), smooth_noise(rng)
    private = np.array([smooth_noise(rng) for _ in range(n_cells)])
    cells = field_weight * field + common_weight * common + private_weight * private
    return cells, field


# --- detrending ---------------------------------------------------------------------------------


def test_running_percentile_equals_a_sliding_percentile_filter_inside_the_trace():
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(3, 400))
    window = 41
    ours = running_percentile(traces, window, 10.0, stride_frames=1)
    reference = np.percentile(sliding_window_view(traces, window, axis=1), 10.0, axis=-1)
    half = window // 2
    np.testing.assert_allclose(ours[:, half:-half], reference)


def test_running_percentile_of_a_constant_is_that_constant_even_with_a_stride():
    traces = np.full((2, 500), 7.0)
    np.testing.assert_allclose(running_percentile(traces, 101, 10.0), 7.0)


def test_detrending_removes_a_slow_drift_but_keeps_a_brief_event():
    rng = np.random.default_rng(1)
    t = np.arange(N_FRAMES)
    drift = 5.0 * t / N_FRAMES  # rises by 5 over the run
    trace = drift + 0.05 * rng.normal(size=N_FRAMES)
    trace[1500:1503] += 3.0  # a 3-frame transient
    out = detrend(trace[None], window_frames=451)[0]
    assert abs(np.median(out[:400]) - np.median(out[-400:])) < 0.3  # the drift is gone
    assert out[1500:1503].min() > 2.0  # the event is still there


def test_detrending_a_trace_with_nan_leaves_nan_only_near_the_gap():
    rng = np.random.default_rng(2)
    trace = rng.normal(size=(1, 1000))
    trace[0, 400:700] = np.nan
    out = detrend(trace, window_frames=101)
    assert np.isnan(out[0, 400:700]).all()
    assert np.isfinite(out[0, :300]).all() and np.isfinite(out[0, 800:]).all()


@pytest.mark.parametrize("window", [0, 2, 5000])
def test_a_window_that_is_tiny_or_longer_than_the_trace_raises(window):
    with pytest.raises(ValueError, match="window"):
        detrend(np.zeros((1, 1000)), window_frames=window)


def test_detrended_z_has_unit_robust_scale_and_flags_a_flat_cell():
    rng = np.random.default_rng(3)
    traces = np.vstack([100 + 20 * rng.normal(size=1000), np.full(1000, 5.0)])
    z, flat = detrended_z(traces, window_frames=101)
    assert flat.tolist() == [False, True]
    assert np.isnan(z[1]).all()
    assert np.nanmedian(z[0]) == pytest.approx(0, abs=1e-9)
    assert 1.4826 * np.nanmedian(np.abs(z[0])) == pytest.approx(1.0)


# --- correlations -------------------------------------------------------------------------------


def test_row_correlation_matches_numpy_and_uses_frames_finite_in_both():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(2, 4, 200))
    b = 0.5 * a + b
    np.testing.assert_allclose(
        row_correlation(a, b), [np.corrcoef(a[i], b[i])[0, 1] for i in range(4)]
    )
    a[0, :50] = np.nan
    b[1, 100:120] = np.nan
    r = row_correlation(a, b)
    ok0, ok1 = slice(50, None), np.r_[0:100, 120:200]
    assert r[0] == pytest.approx(np.corrcoef(a[0, ok0], b[0, ok0])[0, 1])
    assert r[1] == pytest.approx(np.corrcoef(a[1, ok1], b[1, ok1])[0, 1])


def test_row_correlation_broadcasts_a_single_trace():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(3, 100))
    ref = rng.normal(size=100)
    np.testing.assert_allclose(
        row_correlation(a, ref), [np.corrcoef(a[i], ref)[0, 1] for i in range(3)]
    )


def test_row_correlation_is_nan_for_a_constant_row_and_raises_on_a_length_mismatch():
    a = np.vstack([np.ones(50), np.arange(50.0)])
    assert np.isnan(row_correlation(a, np.arange(50.0))[0])
    with pytest.raises(ValueError, match="frames"):
        row_correlation(a, np.arange(40.0))


def test_leave_one_out_mean_excludes_the_cell_itself():
    z = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    loo = leave_one_out_mean(z)
    np.testing.assert_allclose(loo[0], [4.0, 8.0])  # mean of rows 1 and 2
    np.testing.assert_allclose(loo[2], [2.0, 4.0])
    z[1] = 1000.0  # changing a cell moves everyone's average except its own
    np.testing.assert_allclose(leave_one_out_mean(z)[1], [3.0, 6.0])


def test_leave_one_out_mean_skips_nan_and_needs_two_cells():
    z = np.array([[1.0, np.nan], [3.0, 6.0], [5.0, 10.0]])
    loo = leave_one_out_mean(z)
    np.testing.assert_allclose(loo[0], [4.0, 8.0])
    np.testing.assert_allclose(loo[1], [3.0, 10.0])
    with pytest.raises(ValueError, match="two cells"):
        leave_one_out_mean(z[:1])


def test_cells_sharing_only_the_field_signal_have_zero_partial_r_given_the_field():
    rng = np.random.default_rng(6)
    cells, field = make_cells(rng, n_cells=15, common_weight=0.0)
    loo = leave_one_out_mean(cells)
    partial = partial_row_correlation(cells, loo, field)
    assert row_correlation(cells, loo).mean() > 0.2  # they do look correlated ...
    assert abs(np.mean(partial)) < 0.08  # ... but only through the field
    assert abs(np.median(partial)) < 0.08


def test_cells_sharing_an_extra_common_signal_keep_a_positive_partial_r():
    rng = np.random.default_rng(7)
    cells, field = make_cells(rng, n_cells=15, common_weight=1.0)
    partial = partial_row_correlation(cells, leave_one_out_mean(cells), field)
    assert partial.min() > 0.15


def test_partial_correlation_equals_correlating_regression_residuals():
    rng = np.random.default_rng(8)
    cells, field = make_cells(rng, n_cells=5, common_weight=0.5)
    loo = leave_one_out_mean(cells)
    design = np.column_stack([np.ones(N_FRAMES), field])
    expected = []
    for cell, other in zip(cells, loo, strict=True):
        res_cell = cell - design @ np.linalg.lstsq(design, cell, rcond=None)[0]
        res_other = other - design @ np.linalg.lstsq(design, other, rcond=None)[0]
        expected.append(np.corrcoef(res_cell, res_other)[0, 1])
    np.testing.assert_allclose(partial_row_correlation(cells, loo, field), expected, atol=1e-10)


def test_partial_correlation_uses_only_frames_finite_in_all_three():
    rng = np.random.default_rng(9)
    cells, field = make_cells(rng, n_cells=4, common_weight=0.5)
    loo = leave_one_out_mean(cells)
    cells[0, :500] = np.nan
    field_nan = field.copy()
    field_nan[1000:1100] = np.nan
    ok = np.isfinite(cells[0]) & np.isfinite(field_nan)
    design = np.column_stack([np.ones(ok.sum()), field_nan[ok]])
    res = [
        x[ok] - design @ np.linalg.lstsq(design, x[ok], rcond=None)[0] for x in (cells[0], loo[0])
    ]
    got = partial_row_correlation(cells, loo, field_nan)[0]
    assert got == pytest.approx(np.corrcoef(*res)[0, 1], abs=1e-10)


# --- regression on the background ---------------------------------------------------------------


def test_fit_background_recovers_slope_intercept_and_r2():
    rng = np.random.default_rng(10)
    bg = smooth_noise(rng)
    cells = np.vstack([2.0 * bg + 1.0, -0.5 * bg, rng.normal(size=N_FRAMES)])
    fit = fit_background(cells, bg)
    np.testing.assert_allclose(fit["slope"][:2], [2.0, -0.5], atol=1e-9)
    np.testing.assert_allclose(fit["intercept"][:2], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(fit["r2"][:2], 1.0)
    assert fit["r2"][2] < 0.01


def test_remove_background_subtracts_only_the_fitted_slope_times_background():
    rng = np.random.default_rng(11)
    bg = smooth_noise(rng)
    cells = np.vstack([3.0 + 2.0 * bg + 0.1 * rng.normal(size=N_FRAMES)])
    residual = remove_background(cells, bg)
    slope = fit_background(cells, bg)["slope"]
    np.testing.assert_allclose(cells - residual, slope[:, None] * bg)  # offset and scale are kept
    assert abs(row_correlation(residual, bg)[0]) < 1e-9


# --- chance levels ------------------------------------------------------------------------------


def test_shift_offsets_respect_the_minimum_in_both_directions_and_the_seed():
    n, minimum = 1000, 200
    offsets = shift_offsets(n, minimum, 500, np.random.default_rng(0))
    assert offsets.shape == (500,)
    assert offsets.min() >= minimum and offsets.max() <= n - minimum
    np.testing.assert_array_equal(offsets, shift_offsets(n, minimum, 500, np.random.default_rng(0)))
    assert not np.array_equal(offsets, shift_offsets(n, minimum, 500, np.random.default_rng(1)))


@pytest.mark.parametrize(
    "n_frames, minimum, n_shifts", [(300, 200, 10), (1000, 0, 10), (1000, 100, 0)]
)
def test_impossible_shift_settings_raise(n_frames, minimum, n_shifts):
    with pytest.raises(ValueError):
        shift_offsets(n_frames, minimum, n_shifts, np.random.default_rng(0))


def test_a_shifted_background_correlates_with_cells_near_zero_where_the_real_one_does_not():
    rng = np.random.default_rng(12)
    cells, field = make_cells(rng, n_cells=8)
    offsets = shift_offsets(N_FRAMES, 450, 100, np.random.default_rng(1))
    null = null_r(cells, field, offsets)
    assert null.shape == (100, 8)
    real = row_correlation(cells, field)
    assert real.min() > np.percentile(null, 95)


def test_partial_r_null_has_one_row_per_shift_for_either_shifted_trace():
    rng = np.random.default_rng(13)
    cells, field = make_cells(rng, n_cells=12)
    loo = leave_one_out_mean(cells)
    offsets = shift_offsets(N_FRAMES, 450, 20, np.random.default_rng(2))
    for shifted in ("background", "average"):
        assert null_partial_r(cells, loo, field, offsets, shifted).shape == (20, 12)


def test_the_partial_r_of_cells_with_only_field_signal_is_inside_the_null_spread():
    rng = np.random.default_rng(13)
    cells, field = make_cells(rng, n_cells=12)
    loo = leave_one_out_mean(cells)
    offsets = shift_offsets(N_FRAMES, 450, 100, np.random.default_rng(2))
    null = null_partial_r(cells, loo, field, offsets, "average")
    assert np.median(partial_row_correlation(cells, loo, field)) < null_percentile(
        null, 95, np.nanmedian
    )


def test_extra_common_signal_puts_the_real_partial_r_above_the_null():
    rng = np.random.default_rng(14)
    cells, field = make_cells(rng, n_cells=12, common_weight=1.0)
    loo = leave_one_out_mean(cells)
    offsets = shift_offsets(N_FRAMES, 450, 100, np.random.default_rng(3))
    null = null_partial_r(cells, loo, field, offsets, "average")
    assert np.median(partial_row_correlation(cells, loo, field)) > null_percentile(
        null, 95, np.nanmedian
    )


def test_null_partial_r_rejects_an_unknown_shifted_trace():
    with pytest.raises(ValueError, match="shifted"):
        null_partial_r(np.zeros((2, 50)), np.zeros((2, 50)), np.zeros(50), np.array([5]), "cells")


def test_null_percentile_is_per_cell_by_default_and_per_run_with_a_statistic():
    null = np.arange(100.0)[:, None] * np.array([[1.0, 2.0]])
    np.testing.assert_allclose(null_percentile(null, 95), [94.05, 188.1])
    per_run = null_percentile(null, 95, np.nanmedian)
    assert per_run == pytest.approx(np.percentile(np.median(null, axis=1), 95))


# --- comparison across cells --------------------------------------------------------------------


def test_paired_comparison_reports_median_difference_and_a_wilcoxon_p():
    rng = np.random.default_rng(15)
    b = rng.normal(0.2, 0.05, 40)
    a = b + 0.15 + rng.normal(0, 0.02, 40)
    out = paired_comparison(a, b)
    assert out["median_difference"] == pytest.approx(np.median(a - b))
    assert out["n"] == 40 and out["p"] < 1e-6


def test_paired_comparison_of_identical_values_is_not_an_error():
    out = paired_comparison(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
    assert out["median_difference"] == 0 and out["p"] == 1.0


def test_paired_comparison_ignores_nan_pairs():
    out = paired_comparison(np.array([1.0, np.nan, 3.0, 4.0]), np.array([0.0, 1.0, 1.0, np.nan]))
    assert out["n"] == 2


# --- all statistics of one run ------------------------------------------------------------------


def test_cells_resemble_each_other_beyond_the_background_only_with_a_shared_extra_signal():
    rng = np.random.default_rng(16)
    offsets = shift_offsets(N_FRAMES, 450, 50, np.random.default_rng(4))
    field_only, field = make_cells(rng, n_cells=12, common_weight=0.0)
    extra, field2 = make_cells(rng, n_cells=12, common_weight=1.5)
    only = shared_signal_stats(field_only, field, offsets)
    more = shared_signal_stats(extra, field2, offsets)
    # field only: a cell resembles the background about as much as the other cells
    assert np.median(only["r_loo"]) - np.median(only["r_bg"]) < 0.15
    assert np.median(more["r_loo"]) - np.median(more["r_bg"]) > 0.2
    assert np.median(more["partial_r"]) > np.median(only["partial_r"]) + 0.15
    assert only["r_bg_avg"] > 0.5
    assert more["n_cells"] == 12 and more["n_undefined"] == 0
    for key in ("null_r_bg", "null_r_loo", "null_partial_background", "null_partial_average"):
        assert more[key].shape == (50, 12)
    assert more["null_r_bg_avg"].shape == (50,)


def test_stats_count_flat_cells_as_undefined_and_need_a_background_of_the_same_length():
    rng = np.random.default_rng(17)
    cells, field = make_cells(rng, n_cells=5)
    cells[2] = np.nan
    offsets = shift_offsets(N_FRAMES, 450, 5, np.random.default_rng(5))
    stats = shared_signal_stats(cells, field, offsets)
    assert stats["n_undefined"] == 1 and np.isnan(stats["r_loo"][2])
    with pytest.raises(ValueError, match="frames"):
        shared_signal_stats(cells, field[:-1], offsets)


def test_summary_row_collects_the_numbers_for_the_table():
    rng = np.random.default_rng(18)
    cells, field = make_cells(rng, n_cells=10, common_weight=1.0)
    offsets = shift_offsets(N_FRAMES, 450, 50, np.random.default_rng(6))
    stats = shared_signal_stats(cells, field, offsets)
    row = summary_row(stats)
    assert row["cells"] == 10
    assert row["r(bg, avg)"] == pytest.approx(stats["r_bg_avg"])
    assert row["median r(cell, LOO)"] == pytest.approx(np.nanmedian(stats["r_loo"]))
    assert row["median r(cell, bg)"] == pytest.approx(np.nanmedian(stats["r_bg"]))
    assert row["median partial r"] == pytest.approx(np.nanmedian(stats["partial_r"]))
    assert row["null 95th (median partial r)"] == pytest.approx(
        np.percentile(np.nanmedian(stats["null_partial_average"], axis=1), 95)
    )
    assert row["median R2"] == pytest.approx(np.nanmedian(stats["r2"]))


# --- mean response in a time range --------------------------------------------------------------


def test_mean_response_averages_the_requested_seconds_per_pulse_count():
    offsets = np.arange(-3, 6)
    period = 0.5  # offsets in s: -1.5 .. 2.5
    mean = np.where(offsets * period >= 0, 1.0, 0.0)
    mean[offsets * period > 2.0] = 100.0  # outside 0-2 s
    pooled = {("green", "F z"): {2: {"mean": mean, "sem": mean, "n_cells": 3, "n_runs": 1}}}
    out = mean_response(pooled, offsets, period, (0.0, 2.0))
    assert out[("green", "F z")][2] == pytest.approx(1.0)


def test_mean_response_rejects_an_empty_range():
    with pytest.raises(ValueError, match="no frames"):
        mean_response({}, np.arange(-3, 6), 0.5, (10.0, 12.0))
