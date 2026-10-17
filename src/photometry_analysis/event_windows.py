"""Cut event-aligned windows out of a photometry trace on pyControl time.

Every function is pure. A trace comes with the pyControl time of each sample, which is NaN
before the first and after the last matched sync pulse (see ``PhotometryRecording``), and may be
NaN for a stretch in between (a gap in the sync). A window is used only if it lies completely
inside the span between the first and last pulse AND does not overlap any such gap, so no window
is padded, extrapolated or interpolated across missing time. Random control times obey the same
rule.
"""

from collections.abc import Sequence

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


GAP_TOLERANCE = 1e-9  # s; absorbs float rounding when a window ends exactly on a gap edge


def gap_spans(time: np.ndarray) -> list[tuple[float, float]]:
    """Stretches of NaN sample time inside the sync coverage.

    Args:
        time (np.ndarray): pyControl time (s) of each sample, NaN outside the sync pulses.

    Returns:
        list[tuple[float, float]]: For each gap, ``(last finite time before, first finite time
        after)`` (s). Time inside such a span is unknown, so no window may overlap it. Empty
        if the finite samples are contiguous.
    """
    time = np.asarray(time, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(time))
    if len(finite_idx) == 0:
        return []
    jumps = np.flatnonzero(np.diff(finite_idx) > 1)
    return [(float(time[finite_idx[j]]), float(time[finite_idx[j + 1]])) for j in jumps]


def complete_window_mask(
    event_times: np.ndarray,
    coverage: tuple[float, float],
    window: tuple[float, float],
    gaps: Sequence[tuple[float, float]] = (),
) -> np.ndarray:
    """Which events have their whole window inside *coverage* and clear of every gap.

    Args:
        event_times (np.ndarray): Event times (s).
        coverage (tuple[float, float]): ``(first, last)`` time covered (s).
        window (tuple[float, float]): ``(start, stop)`` relative to each event (s).
        gaps (Sequence[tuple[float, float]]): NaN-time spans inside the coverage, from
            :func:`gap_spans`. Default none.

    Returns:
        np.ndarray: Boolean mask, True where ``event + start >= first``, ``event + stop <=
        last``, and the window does not overlap any gap (touching a gap's edge sample is fine).
    """
    event_times = np.asarray(event_times, dtype=float)
    start, stop = event_times + window[0], event_times + window[1]
    mask = (start >= coverage[0]) & (stop <= coverage[1])
    for gap_start, gap_stop in gaps:
        overlaps = (stop > gap_start + GAP_TOLERANCE) & (start < gap_stop - GAP_TOLERANCE)
        mask &= ~overlaps
    return mask


def extract_event_windows(
    time: np.ndarray,
    values: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate *values* onto a common grid around each event with a complete window.

    Interpolation uses only samples with a finite time, and only for windows that lie inside
    the coverage and clear of every NaN-time gap, so ``np.interp`` never clamps or bridges.

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
    keep = complete_window_mask(event_times, sync_coverage(time), window, gap_spans(time))
    covered = np.isfinite(time)
    queries = event_times[keep, None] + grid[None, :]
    trials = np.interp(queries.ravel(), time[covered], values[covered]).reshape(queries.shape)
    if not np.isfinite(trials).all():
        raise ValueError("a kept window contains non-finite values.")
    return grid, trials, keep


def _allowed_intervals(
    low: float, high: float, forbidden: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``[low, high]`` minus the *forbidden* intervals, as a sorted list of non-empty pieces."""
    pieces = [(low, high)]
    for f_low, f_high in forbidden:
        remaining = []
        for p_low, p_high in pieces:
            if f_high <= p_low or f_low >= p_high:
                remaining.append((p_low, p_high))
                continue
            if f_low > p_low:
                remaining.append((p_low, f_low))
            if f_high < p_high:
                remaining.append((f_high, p_high))
        pieces = remaining
    return [(a, b) for a, b in pieces if b > a]


def random_event_times(
    rng: np.random.Generator,
    n_events: int,
    coverage: tuple[float, float],
    window: tuple[float, float],
    gaps: Sequence[tuple[float, float]] = (),
    event_span: tuple[float, float] | None = None,
) -> np.ndarray:
    """Event times drawn uniformly where a whole window fits, clear of gaps.

    Args:
        rng (np.random.Generator): Source of randomness, so draws are reproducible.
        n_events (int): How many times to draw.
        coverage (tuple[float, float]): ``(first, last)`` time covered (s).
        window (tuple[float, float]): ``(start, stop)`` relative to each event (s).
        gaps (Sequence[tuple[float, float]]): NaN-time spans inside the coverage, from
            :func:`gap_spans`; no drawn window overlaps one. Default none.
        event_span (tuple[float, float] | None): Also confine the drawn times themselves to
            this ``(first, last)`` span (s), e.g. the task period. Default None, meaning the
            whole coverage.

    Returns:
        np.ndarray: ``n_events`` times (s), each of which passes :func:`complete_window_mask`.

    Raises:
        ValueError: If no time is left once the coverage, event span and gaps are applied.
    """
    low, high = coverage[0] - window[0], coverage[1] - window[1]
    if event_span is not None:
        low, high = max(low, event_span[0]), min(high, event_span[1])
    forbidden = [(g0 - window[1], g1 - window[0]) for g0, g1 in gaps]
    pieces = _allowed_intervals(low, high, forbidden)
    if not pieces:
        raise ValueError(
            f"no room for a window {window} in coverage {coverage} "
            f"(event span {event_span}, gaps {list(gaps)})."
        )
    lengths = np.array([b - a for a, b in pieces])
    chosen = rng.choice(len(pieces), size=n_events, p=lengths / lengths.sum())
    lows = np.array([a for a, _ in pieces])[chosen]
    return lows + rng.uniform(size=n_events) * lengths[chosen]
