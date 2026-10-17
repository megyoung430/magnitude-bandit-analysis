"""Cut event-aligned windows out of a photometry trace on pyControl time.

Every function is pure. A trace comes with the pyControl time of each sample, which is NaN
before the first and after the last matched sync pulse (see ``PhotometryRecording``). A window
is used only if it lies completely inside the span between those pulses, so no window is
padded or extrapolated.
"""

import numpy as np


def relative_time_grid(window: tuple[float, float], fs: float) -> np.ndarray:
    """Times (s) relative to an event at every sample of the window, both ends included.

    Args:
        window (tuple[float, float]): ``(start, stop)`` relative to the event (s); must span 0.
        fs (float): Sampling rate (Hz).

    Returns:
        np.ndarray: ``start, start + 1/fs, ..., stop`` (the ends are rounded to whole samples).

    Raises:
        ValueError: If the window does not span zero.
    """
    if not window[0] < 0 < window[1]:
        raise ValueError(f"the event window {window} must span zero.")
    return np.arange(round(window[0] * fs), round(window[1] * fs) + 1) / fs


def sync_coverage(time: np.ndarray) -> tuple[float, float]:
    """First and last finite sample time: the span the sync pulses cover.

    Args:
        time (np.ndarray): pyControl time (s) of each sample, NaN outside the sync pulses.

    Returns:
        tuple[float, float]: ``(first, last)`` finite time (s).

    Raises:
        ValueError: If no time is finite, or the finite times do not strictly increase.
    """
    finite = time[np.isfinite(time)]
    if len(finite) == 0:
        raise ValueError("the trace has no finite sample time.")
    if np.any(np.diff(finite) <= 0):
        raise ValueError("sample times must be strictly increasing.")
    return float(finite[0]), float(finite[-1])


def complete_window_mask(
    event_times: np.ndarray, coverage: tuple[float, float], window: tuple[float, float]
) -> np.ndarray:
    """Which events have their whole window inside *coverage*.

    Args:
        event_times (np.ndarray): Event times (s).
        coverage (tuple[float, float]): ``(first, last)`` time covered (s).
        window (tuple[float, float]): ``(start, stop)`` relative to each event (s).

    Returns:
        np.ndarray: Boolean mask, True where ``event + start >= first`` and
        ``event + stop <= last``.
    """
    event_times = np.asarray(event_times, dtype=float)
    return (event_times + window[0] >= coverage[0]) & (event_times + window[1] <= coverage[1])


def extract_event_windows(
    time: np.ndarray,
    values: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate *values* onto a common grid around each event with a complete window.

    Args:
        time (np.ndarray): pyControl time (s) of each sample, NaN outside the sync pulses.
        values (np.ndarray): The trace, same length as *time*.
        event_times (np.ndarray): Event times (s), all finite.
        window (tuple[float, float]): ``(start, stop)`` relative to each event (s).
        fs (float): Sampling rate (Hz); sets the spacing of the relative-time grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The relative-time grid (s), the windows as
        an ``(n_kept, n_grid)`` matrix, and a boolean mask over *event_times* marking the
        events kept.

    Raises:
        ValueError: If the lengths differ, an event time is not finite, the sample times do
            not increase, or a kept window contains a non-finite value.
    """
    time, values = np.asarray(time, dtype=float), np.asarray(values, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time and values must have the same length.")
    if not np.isfinite(event_times).all():
        raise ValueError("event times must be finite.")
    grid = relative_time_grid(window, fs)
    keep = complete_window_mask(event_times, sync_coverage(time), window)
    covered = np.isfinite(time)
    queries = event_times[keep, None] + grid[None, :]
    trials = np.interp(queries.ravel(), time[covered], values[covered]).reshape(queries.shape)
    if not np.isfinite(trials).all():
        raise ValueError("a kept window contains non-finite values.")
    return grid, trials, keep


def random_event_times(
    rng: np.random.Generator,
    n_events: int,
    coverage: tuple[float, float],
    window: tuple[float, float],
) -> np.ndarray:
    """Event times drawn uniformly over the span where a whole window fits in *coverage*.

    Args:
        rng (np.random.Generator): Source of randomness, so draws are reproducible.
        n_events (int): How many times to draw.
        coverage (tuple[float, float]): ``(first, last)`` time covered (s).
        window (tuple[float, float]): ``(start, stop)`` relative to each event (s).

    Returns:
        np.ndarray: ``n_events`` times (s), each with a complete window.

    Raises:
        ValueError: If the covered span is shorter than the window.
    """
    low, high = coverage[0] - window[0], coverage[1] - window[1]
    if high <= low:
        raise ValueError(f"coverage {coverage} is shorter than the window {window}.")
    return rng.uniform(low, high, size=n_events)
