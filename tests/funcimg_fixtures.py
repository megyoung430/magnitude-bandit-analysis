"""Synthetic mini2p DAQ recordings and traces for the imaging tests."""

import numpy as np

DAQ_RATE = 5000.0
FRAME_PERIOD_S = 0.133


def square_wave(n_samples: int, rising: np.ndarray, width: int, high: float = 4.9) -> np.ndarray:
    """A noisy 0 V / *high* V signal that rises at each sample index in *rising*."""
    rng = np.random.default_rng(0)
    signal = rng.normal(-0.2, 0.01, n_samples).astype(np.float32)
    for start in rising:
        signal[start : start + width] += high
    return signal


def daq_session(rng: np.random.Generator, n_frames: int = 2000):
    """A DAQ recording with a frame clock and rsync pulses, and pyControl's view of the pulses.

    pyControl's clock runs 0.01 % fast and starts 12.3 s before the DAQ's.

    Returns:
        dict: ``frame_clock`` and ``rsync`` signals, the true frame start times on the DAQ
        (``frame_s``) and on pyControl (``frame_pycontrol_s``), and pyControl's rsync times
        (``pycontrol_rsync_s``).
    """
    frame_s = 1.0 + FRAME_PERIOD_S * np.arange(n_frames)
    duration = frame_s[-1] + 2.0
    rsync_s = np.cumsum(rng.uniform(0.6, 4.5, int(duration)))  # pyControl's rsync intervals
    rsync_s = rsync_s[rsync_s < duration - 0.5]
    n_samples = int(duration * DAQ_RATE)
    frame_clock = square_wave(n_samples, np.round(frame_s * DAQ_RATE).astype(int), 600)
    rsync = square_wave(n_samples, np.round(rsync_s * DAQ_RATE).astype(int), 250, high=3.3)

    def to_pycontrol(t):
        return 12.3 + t * 1.0001

    return {
        "frame_clock": frame_clock,
        "rsync": rsync,
        "frame_s": frame_s,
        "frame_pycontrol_s": to_pycontrol(frame_s),
        "pycontrol_rsync_s": to_pycontrol(rsync_s),
    }
