"""Frame-clock and rsync pulses from the mini2p acquisition's DAQ recording (``file_*.h5``).

The DAQ samples two analogue lines at ``samplerate`` Hz (5 kHz): ``FrameClock``, high while
each imaging frame (all channels of one timepoint) is acquired, and ``PyControlPulse``, the
rsync pulses pyControl sends. A pulse starts where a line crosses the midpoint between its
low and high levels.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME_CLOCK = "FrameClock"
PYCONTROL_PULSE = "PyControlPulse"
# A line whose high and low levels differ by less than this (V) carries no pulses.
MIN_SWING_V = 0.5


def rising_edges(signal: np.ndarray) -> np.ndarray:
    """Sample indices where *signal* goes from low to high.

    A line that is already high at the first sample has no edge there.

    Args:
        signal (np.ndarray): One DAQ line (V).

    Returns:
        np.ndarray: Index of the first high sample of each pulse.

    Raises:
        ValueError: If the line never swings by ``MIN_SWING_V``, so holds no pulses.
    """
    signal = np.asarray(signal, dtype=float)
    low, high = np.percentile(signal, [0.1, 99.9])
    if high - low < MIN_SWING_V:
        raise ValueError(f"The line swings by only {high - low:.3f} V: it holds no pulses.")
    is_high = signal > (low + high) / 2
    return np.flatnonzero(~is_high[:-1] & is_high[1:]) + 1


def edge_times(signal: np.ndarray, samplerate: float) -> np.ndarray:
    """Times (s, DAQ clock) of the rising edges of *signal*; see :func:`rising_edges`."""
    return rising_edges(signal) / float(samplerate)


@dataclass(frozen=True)
class DaqSync:
    """Pulse times from one DAQ file, in seconds on the DAQ clock."""

    path: Path
    frame_s: np.ndarray  # start of each imaging frame
    rsync_s: np.ndarray  # each rsync pulse from pyControl


def load_daq_sync(h5_path) -> DaqSync:
    """Read the frame clock and rsync pulses from a mini2p ``file_*.h5``.

    Raises:
        KeyError: If the file lacks either line or its ``samplerate`` attribute.
    """
    import h5py

    with h5py.File(h5_path, "r") as daq:
        missing = [line for line in (FRAME_CLOCK, PYCONTROL_PULSE) if line not in daq]
        if missing or "samplerate" not in daq.attrs:
            raise KeyError(f"{h5_path} lacks {missing or 'the samplerate attribute'}.")
        rate = float(daq.attrs["samplerate"])
        frame_s = edge_times(daq[FRAME_CLOCK][:], rate)
        rsync_s = edge_times(daq[PYCONTROL_PULSE][:], rate)
    return DaqSync(Path(h5_path), frame_s, rsync_s)
