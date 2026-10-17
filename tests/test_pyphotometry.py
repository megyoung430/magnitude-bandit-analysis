"""Tests for the pyPhotometry file importer."""

import numpy as np
import pytest

from src.photometry_import.pyphotometry import import_ppd, read_ppd_header
from tests.photometry_fixtures import (
    ADC_MAX_VALUE,
    SAMPLING_RATE,
    VOLTS_PER_DIVISION,
    pulse_trace,
    write_ppd,
)

N_SAMPLES = 400


def _write(path, **overrides):
    rng = np.random.default_rng(0)
    arrays = {
        "signal_on": rng.integers(3000, 6000, N_SAMPLES),
        "signal_baseline": rng.integers(500, 1000, N_SAMPLES),
        "control_on": rng.integers(2000, 4000, N_SAMPLES),
        "control_baseline": rng.integers(500, 1000, N_SAMPLES),
        "sync_trace": pulse_trace(N_SAMPLES, [10, 100, 250]),
    }
    arrays.update(overrides)
    write_ppd(path, subject_id="RT_1", **arrays)
    return arrays


def test_unfiltered_import_recovers_the_written_signals(tmp_path):
    arrays = _write(tmp_path / "a.ppd")
    data = import_ppd(tmp_path / "a.ppd", low_pass=None, high_pass=None)

    assert data["sampling_rate"] == SAMPLING_RATE
    assert data["subject_ID"] == "RT_1"
    on = arrays["signal_on"] * VOLTS_PER_DIVISION
    baseline = arrays["signal_baseline"] * VOLTS_PER_DIVISION
    np.testing.assert_allclose(data["analog_1_raw_LED_on"], on)
    np.testing.assert_allclose(data["analog_1_raw_baseline"], baseline)
    np.testing.assert_allclose(data["analog_1"], on - baseline)
    np.testing.assert_allclose(
        data["analog_2"], (arrays["control_on"] - arrays["control_baseline"]) * VOLTS_PER_DIVISION
    )
    assert data["analog_1_filt"] is None
    np.testing.assert_allclose(data["time"], np.arange(N_SAMPLES) * 1000 / SAMPLING_RATE)


def test_sync_pulse_times_are_the_rising_edges_in_ms(tmp_path):
    _write(tmp_path / "a.ppd")
    data = import_ppd(tmp_path / "a.ppd", low_pass=None, high_pass=None)
    np.testing.assert_allclose(
        data["pulse_times_1"], np.array([10, 100, 250]) * 1000 / SAMPLING_RATE
    )
    assert len(data["pulse_times_2"]) == 0


def test_samples_near_the_adc_ceiling_are_flagged_as_clipping(tmp_path):
    signal_on = np.full(N_SAMPLES, 5000)
    signal_on[7] = ADC_MAX_VALUE - 1
    _write(tmp_path / "a.ppd", signal_on=signal_on)
    data = import_ppd(tmp_path / "a.ppd", low_pass=None, high_pass=None)
    assert np.flatnonzero(data["analog_1_clipping"]).tolist() == [7]
    assert not data["analog_2_clipping"].any()


def test_header_is_readable_without_the_samples(tmp_path):
    _write(tmp_path / "a.ppd")
    header = read_ppd_header(tmp_path / "a.ppd")
    assert header["subject_ID"] == "RT_1"
    assert header["sampling_rate"] == SAMPLING_RATE
    assert header["LED_current"] == [90, 100]


def test_a_file_that_is_not_a_ppd_raises(tmp_path):
    bad = tmp_path / "bad.ppd"
    bad.write_bytes(b"\x05\x00not-json")
    with pytest.raises(ValueError, match="bad.ppd"):
        read_ppd_header(bad)


def test_an_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.ppd"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty.ppd"):
        read_ppd_header(empty)
