"""Synthetic pyPhotometry ``.ppd`` files and pyControl tables for the photometry import tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLING_RATE = 130
VOLTS_PER_DIVISION = 1e-4
ADC_MAX_VALUE = 1 << 15


def pulse_trace(n_samples: int, pulse_samples, width: int = 3) -> np.ndarray:
    """Digital trace that rises at each of *pulse_samples* and stays high for *width* samples."""
    trace = np.zeros(n_samples, dtype=int)
    for sample in pulse_samples:
        trace[sample : sample + width] = 1
    return trace


def write_ppd(
    path: Path,
    *,
    subject_id: str,
    signal_on: np.ndarray,
    signal_baseline: np.ndarray,
    control_on: np.ndarray,
    control_baseline: np.ndarray,
    sync_trace: np.ndarray,
    led_current: tuple[int, int] = (90, 100),
) -> Path:
    """Write a version 1.1 pulsed-mode ``.ppd`` file, the layout ``import_ppd`` reads.

    The file is a 2-byte little-endian header length, a JSON header, then little-endian
    uint16 words. Each sample is four words (signal LED on, signal baseline, control LED on,
    control baseline); each word holds a 15-bit ADC count in its upper bits, and the lowest
    bit of the first and third words carries digital inputs 1 and 2.

    Args:
        path (Path): File to write.
        subject_id (str): Subject recorded in the header.
        signal_on (np.ndarray): ADC counts of analog 1 with its LED on, one per sample.
        signal_baseline (np.ndarray): ADC counts of analog 1 with its LED off.
        control_on (np.ndarray): ADC counts of analog 2 with its LED on.
        control_baseline (np.ndarray): ADC counts of analog 2 with its LED off.
        sync_trace (np.ndarray): Digital input 1 (0/1) per sample.
        led_current (tuple[int, int]): LED currents recorded in the header (mA).

    Returns:
        Path: *path*.
    """
    n_samples = len(signal_on)
    words = np.zeros((n_samples, 4), dtype="<u2")
    words[:, 0] = (np.asarray(signal_on) << 1) | np.asarray(sync_trace)
    words[:, 1] = np.asarray(signal_baseline) << 1
    words[:, 2] = np.asarray(control_on) << 1
    words[:, 3] = np.asarray(control_baseline) << 1
    header = {
        "subject_ID": subject_id,
        "date_time": "2026-10-16T14:26:46.620",
        "end_time": "2026-10-16T15:11:56.712",
        "n_analog_signals": 2,
        "n_digital_signals": 2,
        "mode": "2EX_1EM_pulsed",
        "sampling_rate": SAMPLING_RATE,
        "volts_per_division": [VOLTS_PER_DIVISION, VOLTS_PER_DIVISION],
        "ADC_max_value": ADC_MAX_VALUE,
        "LED_current": list(led_current),
        "version": "1.1",
    }
    header_bytes = json.dumps(header).encode()
    path.write_bytes(len(header_bytes).to_bytes(2, "little") + header_bytes + words.tobytes())
    return path


def write_simple_ppd(path: Path, subject_id: str, sync_samples, n_samples: int) -> Path:
    """Write a ``.ppd`` with constant signals and sync pulses at the *sync_samples* indices."""
    on = np.full(n_samples, 5000)
    baseline = np.full(n_samples, 1000)
    return write_ppd(
        path,
        subject_id=subject_id,
        signal_on=on,
        signal_baseline=baseline,
        control_on=on // 2,
        control_baseline=baseline,
        sync_trace=pulse_trace(n_samples, sync_samples),
    )


def tsv_table(rsync_times_s, n_trials: int = 0) -> pd.DataFrame:
    """A minimal pyControl table: info rows, rsync events, and *n_trials* filler state rows."""
    rows = [(0.0, "info", "subject_id", "RT_1")]
    rows += [(t, "event", "sync", "rsync") for t in rsync_times_s]
    rows += [(float(i), "state", None, "ITI") for i in range(n_trials)]
    return pd.DataFrame(rows, columns=["time", "type", "subtype", "content"]).sort_values(
        "time", kind="stable", ignore_index=True
    )


def write_tsv(path: Path, df: pd.DataFrame) -> Path:
    """Write *df* as a tab-separated pyControl file."""
    df.to_csv(path, sep="\t", index=False)
    return path
