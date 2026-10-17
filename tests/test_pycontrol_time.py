"""Tests for referencing other recording systems to pyControl time."""

import numpy as np
import pandas as pd
import pytest

from src.sync.pycontrol_time import (
    NoSyncPulsesError,
    align_to_pycontrol,
    photometry_sample_times,
    pycontrol_sync_times,
    sync_quality,
    to_pycontrol_time,
    video_frame_times,
)
from src.sync.rsync import RsyncError
from tests.sync_fixtures import sample_clock_pulses

OFFSET_S = 12.3


def _rng():
    return np.random.default_rng(11)


def _tsv(rows):
    return pd.DataFrame(rows, columns=["time", "type", "subtype", "content"])


# --- pycontrol_sync_times ---------------------------------------------------------------


def test_sync_times_are_the_rsync_events_in_seconds():
    df = _tsv(
        [
            (0.005, "event", "sync", "rsync"),
            (1.0, "state", None, "ITI"),
            (3.233, "event", "sync", "rsync"),
            (4.0, "event", "input", "A2_in"),
            (7.072, "event", "sync", "rsync"),
        ]
    )
    np.testing.assert_array_equal(pycontrol_sync_times(df), [0.005, 3.233, 7.072])


def test_sync_times_ignore_rows_that_are_not_events():
    df = _tsv([(1.0, "variable", "print", "rsync"), (2.0, "event", "sync", "rsync")])
    np.testing.assert_array_equal(pycontrol_sync_times(df), [2.0])


def test_sync_times_raise_when_the_session_has_no_rsync_events():
    df = _tsv([(1.0, "state", None, "ITI")])
    with pytest.raises(NoSyncPulsesError, match="rsync"):
        pycontrol_sync_times(df)


def test_no_sync_pulses_error_is_a_value_error():
    assert issubclass(NoSyncPulsesError, ValueError)


# --- align_to_pycontrol / to_pycontrol_time ----------------------------------------------


def test_nanosecond_clock_maps_onto_pycontrol_seconds():
    other_ms, pycontrol_s = sample_clock_pulses(_rng())
    other_ns = other_ms * 1e6
    aligner = align_to_pycontrol(other_ns, pycontrol_s, units_ms=1e-6)
    mapped = to_pycontrol_time(other_ns, aligner)
    paired = ~np.isnan(mapped)
    assert paired.sum() >= len(pycontrol_s) - 5
    np.testing.assert_allclose(mapped[paired], pycontrol_s[paired], atol=1e-9)


def test_times_outside_the_pulses_are_nan_unless_extrapolating():
    other_ms, pycontrol_s = sample_clock_pulses(_rng())
    aligner = align_to_pycontrol(other_ms, pycontrol_s, units_ms=1)
    before, after = other_ms[0] - 2000.0, other_ms[-1] + 2000.0
    assert np.isnan(to_pycontrol_time([before, after], aligner)).all()
    extrapolated = to_pycontrol_time([before, after], aligner, extrapolate=True)
    np.testing.assert_allclose(
        extrapolated, [OFFSET_S + before / 1000, OFFSET_S + after / 1000], atol=5e-3
    )


def test_a_mismatched_pair_of_pulse_trains_raises():
    other_ms, _ = sample_clock_pulses(_rng())
    _, unrelated_s = sample_clock_pulses(np.random.default_rng(99))
    with pytest.raises(RsyncError):
        align_to_pycontrol(other_ms, unrelated_s, units_ms=1)


# --- photometry_sample_times --------------------------------------------------------------


def _recording(rng, fs=130.0):
    pulses_ms, pycontrol_s = sample_clock_pulses(rng, n_pulse=60, sampling_rate=fs)
    n_samples = int(pulses_ms[-1] * fs / 1000) + 200  # recording runs on past the last pulse
    time_ms = np.arange(n_samples) * 1000 / fs
    return {"time": time_ms, "pulse_times_1": pulses_ms}, pycontrol_s


def test_photometry_samples_get_pycontrol_seconds():
    recording, pycontrol_s = _recording(_rng())
    sample_times, _ = photometry_sample_times(recording, pycontrol_s)
    inside = ~np.isnan(sample_times)
    assert inside.sum() > 0.8 * len(sample_times)
    np.testing.assert_allclose(
        sample_times[inside], OFFSET_S + recording["time"][inside] / 1000, atol=5e-3
    )


def test_photometry_samples_beyond_the_first_and_last_pulse_are_nan():
    recording, pycontrol_s = _recording(_rng())
    sample_times, _ = photometry_sample_times(recording, pycontrol_s)
    assert np.isnan(sample_times[0])
    assert np.isnan(sample_times[-1])
    assert not np.isnan(sample_times[len(sample_times) // 2])


# --- video_frame_times --------------------------------------------------------------------

FPS = 60
SYNC_HIGH_FRAMES = 3


def _gpio(rng, n_pulse=50):
    """Camera GPIO table with sync pulses on GPIO1, and the pyControl times of those pulses."""
    edge_frames = np.cumsum(rng.integers(6, 114, size=n_pulse))
    n_frames = edge_frames[-1] + 60  # recording runs on past the last pulse
    sync = np.zeros(n_frames, dtype=int)
    for frame in edge_frames:
        sync[frame : frame + SYNC_HIGH_FRAMES] = 1
    timestamp_ns = np.round(np.arange(n_frames) * 1e9 / FPS).astype(np.int64)
    gpio = pd.DataFrame(
        {"GPIO1": sync, "GPIO2": 1, "GPIO3": 0, "Timestamp": timestamp_ns},
    )
    pycontrol_s = OFFSET_S + timestamp_ns[edge_frames] / 1e9 + rng.normal(0, 2e-4, n_pulse)
    return gpio, pycontrol_s


def test_video_frames_get_pycontrol_seconds_from_the_sync_column():
    gpio, pycontrol_s = _gpio(_rng())
    frame_times, aligner = video_frame_times(gpio, pycontrol_s)
    inside = ~np.isnan(frame_times)
    assert inside.sum() > 0.8 * len(gpio)
    np.testing.assert_allclose(
        frame_times[inside], OFFSET_S + gpio["Timestamp"].to_numpy()[inside] / 1e9, atol=5e-3
    )
    assert sync_quality(aligner)["n_matched"] >= 40


def test_video_frames_use_the_named_sync_column():
    gpio, pycontrol_s = _gpio(_rng())
    gpio = gpio.rename(columns={"GPIO1": "GPIO3", "GPIO3": "GPIO1"})
    frame_times, _ = video_frame_times(gpio, pycontrol_s, sync_column="GPIO3")
    assert (~np.isnan(frame_times)).sum() > 0.8 * len(gpio)


def test_video_without_a_sync_column_raises():
    gpio, pycontrol_s = _gpio(_rng())
    with pytest.raises(ValueError, match="GPIO7"):
        video_frame_times(gpio, pycontrol_s, sync_column="GPIO7")


def test_video_with_no_rising_edges_raises():
    gpio, pycontrol_s = _gpio(_rng())
    gpio["GPIO1"] = 0
    with pytest.raises(ValueError, match="rising edge"):
        video_frame_times(gpio, pycontrol_s)


def test_video_with_a_non_binary_sync_column_raises():
    gpio, pycontrol_s = _gpio(_rng())
    gpio.loc[5, "GPIO1"] = 7
    with pytest.raises(ValueError, match="0 and 1"):
        video_frame_times(gpio, pycontrol_s)


def test_video_with_non_increasing_timestamps_raises():
    gpio, pycontrol_s = _gpio(_rng())
    gpio.loc[10, "Timestamp"] = gpio.loc[9, "Timestamp"]
    with pytest.raises(ValueError, match="increasing"):
        video_frame_times(gpio, pycontrol_s)


# --- sync_quality -------------------------------------------------------------------------


def test_sync_quality_summarises_a_clean_alignment():
    other_ms, pycontrol_s = sample_clock_pulses(_rng())
    aligner = align_to_pycontrol(other_ms, pycontrol_s, units_ms=1)
    quality = sync_quality(aligner)
    assert quality["n_pulses_other"] == len(other_ms)
    assert quality["n_pulses_pycontrol"] == len(pycontrol_s)
    assert quality["n_matched"] >= len(other_ms) - 5
    assert quality["fraction_matched"] == pytest.approx(quality["n_matched"] / len(other_ms))
    assert 0 <= quality["median_abs_interval_discrepancy_ms"] <= 1.0
    assert (
        quality["median_abs_interval_discrepancy_ms"] <= quality["max_abs_interval_discrepancy_ms"]
    )
    assert quality["max_abs_interval_discrepancy_ms"] < 3.0
    assert quality["first_matched_time_s"] == pytest.approx(pycontrol_s[0], abs=1e-9)
    assert quality["last_matched_time_s"] > quality["first_matched_time_s"]


def test_sync_quality_fraction_uses_the_shorter_train():
    other_ms, pycontrol_s = sample_clock_pulses(_rng(), n_pulse=80)
    # pyControl also logged 20 pulses that the other system never saw.
    extra = pycontrol_s[-1] + np.cumsum(np.random.default_rng(5).uniform(0.1, 1.9, 20))
    aligner = align_to_pycontrol(other_ms, np.r_[pycontrol_s, extra], units_ms=1)
    quality = sync_quality(aligner)
    assert quality["fraction_matched"] == pytest.approx(quality["n_matched"] / 80)
    assert quality["fraction_matched"] > 0.9
