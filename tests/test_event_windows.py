"""Tests for cutting event-aligned windows out of a trace on pyControl time."""

import numpy as np
import pytest

from src.photometry_analysis.event_windows import (
    complete_window_mask,
    extract_event_windows,
    random_event_times,
    relative_time_grid,
    sync_coverage,
)

FS = 10.0


def make_trace(n=1000, first_valid=50, last_valid=949, start=100.0):
    """A ramp whose value equals its pyControl time, with NaN time outside sync coverage."""
    time = start + np.arange(n) / FS
    time[:first_valid] = np.nan
    time[last_valid + 1 :] = np.nan
    values = np.where(np.isnan(time), 0.0, time)
    return time, values


def test_grid_is_symmetric_in_samples_and_includes_both_ends():
    grid = relative_time_grid((-1.0, 2.0), FS)
    assert grid[0] == -1.0 and grid[-1] == 2.0
    assert len(grid) == 31
    np.testing.assert_allclose(np.diff(grid), 1 / FS)


def test_grid_rejects_window_not_spanning_zero():
    with pytest.raises(ValueError, match="span zero"):
        relative_time_grid((1.0, 2.0), FS)


def test_coverage_is_first_and_last_finite_time():
    time, _ = make_trace()
    first, last = sync_coverage(time)
    assert first == pytest.approx(105.0)
    assert last == pytest.approx(194.9)


def test_coverage_rejects_non_increasing_time():
    time, _ = make_trace()
    time[300] = time[299]
    with pytest.raises(ValueError, match="strictly increasing"):
        sync_coverage(time)


def test_coverage_rejects_all_nan_time():
    with pytest.raises(ValueError, match="no finite"):
        sync_coverage(np.full(10, np.nan))


def test_complete_window_mask_excludes_windows_touching_the_edges():
    coverage = (105.0, 194.9)
    events = np.array([105.5, 106.5, 150.0, 192.8, 193.0])
    mask = complete_window_mask(events, coverage, (-1.0, 2.0))
    # 105.5 - 1 < 105 and 193.0 + 2 > 194.9 stick out; the other three fit
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_extract_returns_the_trace_at_event_plus_relative_time():
    time, values = make_trace()
    events = np.array([150.0, 160.55])
    rel, trials, keep = extract_event_windows(time, values, events, (-1.0, 2.0), FS)
    assert keep.all()
    assert trials.shape == (2, len(rel))
    # the ramp's value is its own time, so linear interpolation is exact
    np.testing.assert_allclose(trials, events[:, None] + rel[None, :])


def test_extract_drops_incomplete_windows_and_reports_which():
    time, values = make_trace()
    events = np.array([100.0, 150.0, 400.0])  # before, inside, after the coverage
    _, trials, keep = extract_event_windows(time, values, events, (-1.0, 2.0), FS)
    np.testing.assert_array_equal(keep, [False, True, False])
    assert trials.shape[0] == 1


def test_extract_with_no_complete_window_returns_empty_matrix():
    time, values = make_trace()
    rel, trials, keep = extract_event_windows(time, values, np.array([0.0]), (-1.0, 2.0), FS)
    assert trials.shape == (0, len(rel))
    assert not keep.any()


def test_extract_rejects_nonfinite_event_and_mismatched_lengths():
    time, values = make_trace()
    with pytest.raises(ValueError, match="finite"):
        extract_event_windows(time, values, np.array([np.nan]), (-1.0, 2.0), FS)
    with pytest.raises(ValueError, match="same length"):
        extract_event_windows(time, values[:-1], np.array([150.0]), (-1.0, 2.0), FS)


def test_extract_rejects_nan_values_inside_a_kept_window():
    time, values = make_trace()
    values[500] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_event_windows(time, values, np.array([time[500]]), (-1.0, 2.0), FS)


def test_random_times_all_give_complete_windows_and_are_reproducible():
    coverage = (105.0, 194.9)
    window = (-1.0, 2.0)
    first = random_event_times(np.random.default_rng(0), 500, coverage, window)
    again = random_event_times(np.random.default_rng(0), 500, coverage, window)
    np.testing.assert_array_equal(first, again)
    assert complete_window_mask(first, coverage, window).all()
    # uniform: both ends of the allowed span are approached
    assert first.min() < 107 and first.max() > 192


def test_random_times_reject_coverage_shorter_than_window():
    with pytest.raises(ValueError, match="shorter than"):
        random_event_times(np.random.default_rng(0), 3, (0.0, 2.0), (-1.0, 2.0))
