import numpy as np
import pytest

from src.funcimg_analysis.event_frames import (
    baseline_subtract,
    event_frames,
    frame_period_s,
    frame_windows,
    limited_frame_times,
    mean_by_category,
    pooled_mean_sem,
    robust_z,
    top_cells_by_max,
    top_cells_by_score,
    window_offsets,
)

PERIOD = 0.133


def frame_times(n=200, start=10.0, missing=()):
    times = start + PERIOD * np.arange(n)
    times[list(missing)] = np.nan
    return times


def test_each_event_goes_to_its_nearest_frame_and_keeps_its_offset():
    times = frame_times()
    idx, offset, kept = event_frames(times, np.array([10.0 + 5 * PERIOD + 0.02, 10.0 - 1.0]))
    assert kept.tolist() == [True, False]  # the second is before the first frame
    assert idx[0] == 5 and offset[0] == pytest.approx(0.02)  # 20 ms after the frame starts
    assert frame_period_s(times) == pytest.approx(PERIOD)


def test_an_event_next_to_frames_without_a_time_is_dropped():
    times = frame_times(missing=range(50, 60))
    _, _, kept = event_frames(times, np.array([10.0 + 55 * PERIOD]))
    assert kept.tolist() == [False]


def test_windows_are_whole_frames_and_incomplete_ones_are_dropped():
    traces = np.arange(2 * 100, dtype=float).reshape(2, 100)  # cell 1 = cell 0 + 100
    valid = np.ones(100, bool)
    valid[80:] = False
    windows, kept = frame_windows(traces, np.array([1, 10, 50, 78]), 2, 3, valid)
    assert kept.tolist() == [False, True, True, False]
    assert windows.shape == (2, 2, 6)
    assert windows[0, 0].tolist() == [8, 9, 10, 11, 12, 13]
    assert windows[1, 1, 2] == 150
    assert window_offsets(2, 3).tolist() == [-2, -1, 0, 1, 2, 3]


def test_cell_means_by_pulse_count_ignore_nan_frames():
    windows = np.array([[[1.0, 2.0], [3.0, np.nan], [10.0, 10.0]]])  # 1 cell, 3 trials
    means = mean_by_category(windows, np.array([0, 0, 4]), [0, 1, 4])
    assert means[0].tolist() == [[2.0, 2.0]]
    assert means[4].tolist() == [[10.0, 10.0]]
    assert 1 not in means  # no trials, so no cells contribute


def test_pooled_mean_and_sem_across_all_cells_of_all_runs():
    run_a = {0: np.array([[1.0, 1.0], [3.0, 3.0]])}
    run_b = {0: np.array([[5.0, np.nan]]), 2: np.array([[2.0, 2.0]])}
    pooled = pooled_mean_sem([run_a, run_b], [0, 2])
    assert pooled[0]["mean"].tolist() == [3.0, 2.0]
    assert pooled[0]["n_cells"] == 3 and pooled[0]["n_runs"] == 2
    assert pooled[0]["sem"][0] == pytest.approx(2.0 / np.sqrt(3))
    assert pooled[2]["n_cells"] == 1 and np.isnan(pooled[2]["sem"]).all()


def test_cells_without_any_value_are_not_counted():
    pooled = pooled_mean_sem([{0: np.array([[1.0, 2.0], [np.nan, np.nan]])}], [0])
    assert pooled[0]["n_cells"] == 1


def test_too_few_frames_with_a_time_is_a_clear_error():
    with pytest.raises(ValueError, match="at least two frames"):
        event_frames(np.array([np.nan, 1.0, np.nan]), np.array([1.0]))


def test_most_responsive_cells_by_their_largest_dff():
    dff = np.array([[0.1, 0.5], [2.0, np.nan], [np.nan, np.nan], [0.9, 0.2]])
    assert top_cells_by_max(dff, 2).tolist() == [1, 3]
    assert top_cells_by_max(dff, 10).tolist() == [1, 3, 0]  # an all-NaN cell is left out


def test_frame_times_extend_only_to_the_daq_sync_pulses():
    matched = np.array([np.nan, 1.0, 2.0, np.nan, np.nan])
    extrapolated = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    frame_s = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    times = limited_frame_times(matched, extrapolated, frame_s, daq_rsync_s=np.array([0.7, 4.0]))
    assert np.isnan(times[0]) and np.isnan(times[4])  # outside the DAQ's own pulses
    assert times[1:4].tolist() == [1.0, 2.0, 3.0]


def test_robust_z_uses_the_median_and_scaled_mad_of_each_cell():
    rng = np.random.default_rng(0)
    traces = np.vstack([rng.normal(5, 2, 5000), rng.normal(-3, 0.1, 5000)])
    z, flat = robust_z(traces)
    assert not flat.any()
    assert np.allclose(np.median(z, axis=1), 0, atol=1e-9)
    assert np.allclose(1.4826 * np.median(np.abs(z), axis=1), 1, atol=1e-9)


def test_robust_z_ignores_nan_and_does_not_divide_by_a_zero_mad():
    traces = np.array([[1.0, 2.0, np.nan, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0, 9.0], [np.nan] * 5])
    z, flat = robust_z(traces)
    assert np.isfinite(z[0, [0, 1, 3, 4]]).all() and np.isnan(z[0, 2])
    assert flat.tolist() == [False, True, True]  # a MAD of 0 and an all-NaN cell are flagged
    assert np.isnan(z[1]).all() and np.isnan(z[2]).all()


def test_baseline_subtraction_zeroes_each_trials_pre_event_mean():
    windows = np.random.default_rng(1).normal(size=(3, 7, 12)) + 50.0
    out = baseline_subtract(windows, pre_frames=4)
    assert np.allclose(np.nanmean(out[:, :, :4], axis=2), 0)
    assert np.allclose(out - windows, -np.nanmean(windows[:, :, :4], axis=2)[:, :, None])


def test_baseline_subtraction_needs_pre_event_frames():
    with pytest.raises(ValueError, match="pre-event"):
        baseline_subtract(np.zeros((1, 2, 5)), pre_frames=0)


def test_cells_are_ranked_by_score_with_ties_broken_by_roi_index():
    dff = np.zeros((5, 10))
    dff[4] = np.nan  # no ΔF/F0 at all: cannot be plotted
    scores = np.array([0.5, 0.9, 0.5, 0.9, 1.0])
    assert top_cells_by_score(dff, scores, 3).tolist() == [1, 3, 0]
    assert top_cells_by_score(dff, scores, 10).tolist() == [1, 3, 0, 2]


def test_cells_without_a_score_are_left_out_of_a_score_ranking():
    dff = np.ones((3, 4))
    assert top_cells_by_score(dff, np.array([0.2, np.nan, 0.7]), 3).tolist() == [2, 0]


def test_baseline_leaves_out_the_last_pre_frame_of_trials_whose_event_it_spans():
    windows = np.zeros((1, 2, 8))
    windows[0, :, 3] = 100.0  # the frame at offset -1 of a window with 4 pre-event frames
    out = baseline_subtract(
        windows, pre_frames=4, last_pre_frame_spans_event=np.array([True, False])
    )
    assert out[0, 0, 0] == 0  # baseline of the first trial ignores the frame holding the event
    assert out[0, 1, 0] == -25  # the second trial's baseline includes it
