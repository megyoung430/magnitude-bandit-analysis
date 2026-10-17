"""Reference other recording systems (photometry, video) to pyControl time, in seconds.

Every function here is pure: it takes pulse times or tables already in memory and returns
times, with no file access. The lab convention is that all signals end up on pyControl's
clock; the sync pulses (``rsync``) that pyControl sends to the other system tie the clocks
together.
"""

import numpy as np
import pandas as pd

from src.sync.rsync import RsyncAligner

# pyControl timestamps are in seconds; the aligner wants the length of one unit in ms.
PYCONTROL_UNITS_MS = 1000


class NoSyncPulsesError(ValueError):
    """A pyControl session holds no ``rsync`` events."""


def pycontrol_sync_times(df: pd.DataFrame) -> np.ndarray:
    """Times of the rsync pulses pyControl sent, from a session's event table.

    Args:
        df (pd.DataFrame): A pyControl TSV as a table, with ``time`` (s), ``type`` and
            ``content`` columns.

    Returns:
        np.ndarray: Time of every ``rsync`` event (s), in file order.

    Raises:
        ValueError: If a required column is missing.
        NoSyncPulsesError: If the session has no ``rsync`` events.
    """
    missing = {"time", "type", "content"} - set(df.columns)
    if missing:
        raise ValueError(f"pyControl table lacks column(s) {sorted(missing)}.")
    is_sync = (df["type"] == "event") & (df["content"] == "rsync")
    if not is_sync.any():
        raise NoSyncPulsesError("pyControl session has no rsync events.")
    return df.loc[is_sync, "time"].to_numpy(dtype=float)


def align_to_pycontrol(
    pulse_times: np.ndarray, pycontrol_pulses_s: np.ndarray, units_ms: float
) -> RsyncAligner:
    """Match another system's sync pulses to pyControl's.

    Args:
        pulse_times (np.ndarray): Sync pulse times on the other system (system A).
        pycontrol_pulses_s (np.ndarray): Sync pulse times on pyControl (system B), in seconds.
        units_ms (float): Length of one unit of *pulse_times* in milliseconds: 1 for ms,
            1e-6 for ns, 1000 for s.

    Returns:
        RsyncAligner: Aligner with the other system as A and pyControl as B.

    Raises:
        RsyncError: If the two pulse trains do not match.
        ValueError: If either train is unusable (see :class:`RsyncAligner`).
    """
    return RsyncAligner(
        pulse_times, pycontrol_pulses_s, units_A=units_ms, units_B=PYCONTROL_UNITS_MS
    )


def to_pycontrol_time(times, aligner: RsyncAligner, extrapolate: bool = False) -> np.ndarray:
    """Convert times on the other system to pyControl seconds.

    Args:
        times (array-like): Times on the other system, in the units given to *aligner*.
        aligner (RsyncAligner): From :func:`align_to_pycontrol`.
        extrapolate (bool): Continue the clock beyond the first and last matched pulse. Default
            False, which leaves those times NaN.

    Returns:
        np.ndarray: pyControl time (s) of each entry of *times*; NaN outside the matched
        pulses unless extrapolating.
    """
    return aligner.A_to_B(times, extrapolate=extrapolate)


def photometry_sample_times(
    recording: dict, pycontrol_pulses_s: np.ndarray
) -> tuple[np.ndarray, RsyncAligner]:
    """pyControl time of every sample of a pyPhotometry recording.

    Args:
        recording (dict): As returned by ``import_ppd``; uses ``time`` (ms, per sample) and
            ``pulse_times_1`` (ms, rsync rising edges on digital input 1).
        pycontrol_pulses_s (np.ndarray): pyControl rsync times (s).

    Returns:
        tuple[np.ndarray, RsyncAligner]: pyControl time (s) of each sample, NaN before the
        first and after the last matched pulse, and the aligner used.
    """
    aligner = align_to_pycontrol(recording["pulse_times_1"], pycontrol_pulses_s, units_ms=1)
    return to_pycontrol_time(recording["time"], aligner), aligner


def _rising_edge_times(gpio_df: pd.DataFrame, sync_column: str) -> np.ndarray:
    """Camera timestamps (ns) of the frames where *sync_column* goes from 0 to 1."""
    for column in (sync_column, "Timestamp"):
        if column not in gpio_df.columns:
            raise ValueError(f"GPIO table has no column {column!r}; has {list(gpio_df.columns)}.")
    sync = gpio_df[sync_column].to_numpy()
    if not np.isin(sync, (0, 1)).all():
        raise ValueError(f"GPIO column {sync_column!r} must hold only 0 and 1.")
    timestamps = gpio_df["Timestamp"].to_numpy(dtype=float)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("GPIO Timestamp must be strictly increasing.")
    rising = np.flatnonzero(np.diff(sync) == 1) + 1
    if len(rising) == 0:
        raise ValueError(f"GPIO column {sync_column!r} has no rising edge.")
    return timestamps[rising]


def video_frame_times(
    gpio_df: pd.DataFrame, pycontrol_pulses_s: np.ndarray, sync_column: str = "GPIO1"
) -> tuple[np.ndarray, RsyncAligner]:
    """pyControl time of every video frame.

    Args:
        gpio_df (pd.DataFrame): The camera's ``*_GPIO_data.csv`` as a table, one row per
            frame, with the GPIO columns and ``Timestamp`` (ns, camera clock).
        pycontrol_pulses_s (np.ndarray): pyControl rsync times (s).
        sync_column (str): GPIO column carrying the rsync pulses. Default ``"GPIO1"``.

    Returns:
        tuple[np.ndarray, RsyncAligner]: pyControl time (s) of each frame, NaN before the
        first and after the last matched pulse, and the aligner used.

    Raises:
        ValueError: If the sync column or ``Timestamp`` is missing or malformed, or the
            sync column never rises.
    """
    edge_times_ns = _rising_edge_times(gpio_df, sync_column)
    aligner = align_to_pycontrol(edge_times_ns, pycontrol_pulses_s, units_ms=1e-6)
    return to_pycontrol_time(gpio_df["Timestamp"].to_numpy(dtype=float), aligner), aligner


def sync_quality(aligner: RsyncAligner) -> dict:
    """Summary of how well another system's pulses matched pyControl's.

    Args:
        aligner (RsyncAligner): From :func:`align_to_pycontrol` (pyControl is system B).

    Returns:
        dict: ``n_pulses_other`` and ``n_pulses_pycontrol`` (counts); ``n_matched`` (pulses
        of the other system with a pyControl partner); ``fraction_matched`` (``n_matched`` over
        the shorter train, the most that could match); ``max_abs_interval_discrepancy_ms`` and
        ``median_abs_interval_discrepancy_ms`` (differences between the two systems' intervals
        across matched neighbouring pulses); ``first_matched_time_s`` and
        ``last_matched_time_s`` (pyControl time of the first and last matched pulse).
    """
    n_matched = int(np.sum(~np.isnan(aligner.cor_times_B)))
    n_other, n_pycontrol = len(aligner.pulse_times_A), len(aligner.pulse_times_B)
    discrepancy = np.abs(aligner.interval_discrepancy_ms())
    return {
        "n_pulses_other": n_other,
        "n_pulses_pycontrol": n_pycontrol,
        "n_matched": n_matched,
        "fraction_matched": n_matched / min(n_other, n_pycontrol),
        "max_abs_interval_discrepancy_ms": float(discrepancy.max()),
        "median_abs_interval_discrepancy_ms": float(np.median(discrepancy)),
        "first_matched_time_s": float(aligner.first_matched_time_B),
        "last_matched_time_s": float(aligner.last_matched_time_B),
    }
