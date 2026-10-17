"""Tests for cutting event-aligned windows out of a trace on pyControl time."""

import numpy as np
import pytest

from src.photometry_analysis.event_windows import (
    complete_window_mask,
    extract_event_windows,
    gap_spans,
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
    with pytest.raises(ValueError, match="no room"):
        random_event_times(np.random.default_rng(0), 3, (0.0, 2.0), (-1.0, 2.0))


def make_gapped_trace():
    """The ramp with NaN sample times from 140 s to 160 s: a gap inside the sync coverage."""
    time, values = make_trace()
    time[(time >= 140.0) & (time < 160.0)] = np.nan
    values = np.where(np.isnan(time), 0.0, time)
    return time, values


def test_gap_spans_report_the_nan_stretch_inside_coverage_only():
    time, _ = make_gapped_trace()
    ((start, stop),) = gap_spans(time)
    # bounded by the last finite sample before and the first after the gap
    assert start == pytest.approx(139.9) and stop == pytest.approx(160.0)


def test_extract_rejects_windows_overlapping_a_gap_and_keeps_those_away_from_it():
    time, values = make_gapped_trace()
    window = (-1.0, 2.0)
    # 150 is inside the gap; 138.5 and 159.5 have a window reaching into it (138.5+2 > 139.9;
    # 159.5-1 < 160); 130 and 170 are clear of it
    events = np.array([130.0, 138.5, 150.0, 159.5, 170.0])
    rel, trials, keep = extract_event_windows(time, values, events, window, FS)
    np.testing.assert_array_equal(keep, [True, False, False, False, True])
    np.testing.assert_allclose(trials, events[keep, None] + rel[None, :])


def test_a_window_can_touch_the_gap_edge_samples():
    time, values = make_gapped_trace()
    # window (-1, 2) after 139.9 - 2 = 137.9 ends exactly on the last sample before the gap
    _, _, keep = extract_event_windows(time, values, np.array([137.9, 161.0]), (-1.0, 2.0), FS)
    assert keep.all()


def test_random_times_avoid_the_gap_and_stay_inside_coverage():
    time, _ = make_gapped_trace()
    window = (-1.0, 2.0)
    coverage = sync_coverage(time)
    times = random_event_times(
        np.random.default_rng(1), 2000, coverage, window, gaps=gap_spans(time)
    )
    assert complete_window_mask(times, coverage, window, gaps=gap_spans(time)).all()
    # both sides of the gap are sampled
    assert (times < 139.9).any() and (times > 160.0).any()


def test_random_times_stay_inside_the_event_span():
    coverage = (105.0, 194.9)
    times = random_event_times(
        np.random.default_rng(2), 1000, coverage, (-1.0, 2.0), event_span=(120.0, 130.0)
    )
    assert times.min() >= 120.0 and times.max() <= 130.0


def test_random_times_raise_when_the_event_span_leaves_no_room():
    with pytest.raises(ValueError, match="no room"):
        random_event_times(
            np.random.default_rng(0), 3, (105.0, 194.9), (-1.0, 2.0), event_span=(300.0, 310.0)
        )
