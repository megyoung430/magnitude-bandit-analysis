"""Event-aligned windows of imaging traces, in whole frames (pure).

Imaging frames come every ~133 ms, so an event is assigned to its nearest frame and a window is
a run of whole frames around it: no interpolation, and the x axis can be read in frames or, with
the frame period, in seconds. Frames without a pyControl time (outside the sync pulses) are
never part of a window.
"""

import warnings
from contextlib import contextmanager

import numpy as np


def frame_period_s(frame_times_s: np.ndarray) -> float:
    """Median interval between frames that have a pyControl time (s)."""
    times = np.asarray(frame_times_s, dtype=float)
    return float(np.median(np.diff(times[np.isfinite(times)])))


def limited_frame_times(
    matched_s: np.ndarray,
    extrapolated_s: np.ndarray,
    frame_s: np.ndarray,
    daq_rsync_s: np.ndarray,
) -> np.ndarray:
    """Frame times extended past the matched pulses, but only as far as the DAQ's own pulses.

    The aligner converts times only between the first and last *matched* pulse; the last few
    pulses usually fall in an incomplete chunk and stay unmatched, which would drop the end of
    every run. Frames between the first and last rsync pulse the DAQ recorded take the
    extrapolated clock (drift is ~0.1 ms per second); frames outside them stay NaN.

    Args:
        matched_s (np.ndarray): pyControl time per frame without extrapolation (NaN outside).
        extrapolated_s (np.ndarray): pyControl time per frame with extrapolation.
        frame_s (np.ndarray): Frame start times on the DAQ clock (s).
        daq_rsync_s (np.ndarray): Rsync pulses on the DAQ clock (s).

    Returns:
        np.ndarray: pyControl time (s) per frame, NaN outside the DAQ's sync pulses.
    """
    frame_s = np.asarray(frame_s, dtype=float)
    inside = (frame_s >= np.min(daq_rsync_s)) & (frame_s <= np.max(daq_rsync_s))
    times = np.where(np.isfinite(matched_s), matched_s, extrapolated_s)
    return np.where(inside, times, np.nan)


def event_frames(
    frame_times_s: np.ndarray, event_times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The frame nearest each event, and how far the event is from that frame's start.

    A frame's time is the start of its acquisition (the frame clock's rising edge); a frame
    takes almost a whole period to acquire, so frame 0 of a window can start up to half a
    period before or after the event.

    Args:
        frame_times_s (np.ndarray): pyControl time (s) of each frame, NaN where unknown.
        event_times_s (np.ndarray): Event times (s, pyControl).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Frame index of each event, the event time
        minus that frame's time (s), and whether the event is kept: within half a frame period
        of a frame whose time is known, so neither outside the imaging nor next to a gap.
    """
    times = np.asarray(frame_times_s, dtype=float)
    events = np.asarray(event_times_s, dtype=float)
    known = np.flatnonzero(np.isfinite(times))
    if len(known) < 2:
        raise ValueError("Need at least two frames with a pyControl time to place events.")
    position = np.clip(np.searchsorted(times[known], events), 1, len(known) - 1)
    before, after = known[position - 1], known[position]
    nearest = np.where(events - times[before] <= times[after] - events, before, after)
    offset = events - times[nearest]
    kept = np.abs(offset) <= frame_period_s(times) / 2
    return nearest, offset, kept


def window_offsets(pre_frames: int, post_frames: int) -> np.ndarray:
    """Frame offsets of a window, ``-pre_frames .. post_frames`` (0 is the event's frame)."""
    return np.arange(-pre_frames, post_frames + 1)


def frame_windows(
    traces: np.ndarray,
    event_index: np.ndarray,
    pre_frames: int,
    post_frames: int,
    valid_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Windows of whole frames around each event, for every cell.

    Args:
        traces (np.ndarray): ``(n_cells, n_frames)``.
        event_index (np.ndarray): Frame of each event (see :func:`event_frames`).
        pre_frames (int): Frames before the event's frame.
        post_frames (int): Frames after it.
        valid_frames (np.ndarray): Boolean per frame; False for frames without a pyControl time.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(n_cells, n_kept, n_window)`` windows, and which events
        were kept: those whose whole window lies inside the recording on valid frames.
    """
    traces = np.asarray(traces, dtype=float)
    offsets = window_offsets(pre_frames, post_frames)
    frames = np.asarray(event_index, dtype=int)[:, None] + offsets[None, :]
    inside = (frames >= 0).all(axis=1) & (frames < traces.shape[1]).all(axis=1)
    kept = inside.copy()
    kept[inside] = np.asarray(valid_frames, bool)[frames[inside]].all(axis=1)
    return traces[:, frames[kept]], kept


def mean_by_category(
    windows: np.ndarray, categories_per_event: np.ndarray, categories: list[int]
) -> dict[int, np.ndarray]:
    """Each cell's mean window over the trials of each category, ignoring NaN frames.

    Args:
        windows (np.ndarray): ``(n_cells, n_events, n_window)``.
        categories_per_event (np.ndarray): Category (e.g. reward pulses) of each event.
        categories (list[int]): Categories to compute; one without trials is left out.

    Returns:
        dict[int, np.ndarray]: ``{category: (n_cells, n_window)}``.
    """
    labels = np.asarray(categories_per_event)
    means = {}
    for category in categories:
        trials = windows[:, labels == category]
        if trials.shape[1]:
            with _quiet_empty_means():
                means[category] = np.nanmean(trials, axis=1)
    return means


def pooled_mean_sem(per_run: list[dict[int, np.ndarray]], categories: list[int]) -> dict:
    """Mean and SEM across every cell of every run, per category.

    Each cell contributes its own trial-averaged window, so a run's weight is its number of
    cells, not its number of trials.

    Returns:
        dict: ``{category: {"mean", "sem", "n_cells", "n_runs"}}`` for categories with cells;
        ``sem`` is NaN where fewer than two cells have a value.
    """
    pooled = {}
    for category in categories:
        stacks = [run[category] for run in per_run if category in run]
        if not stacks:
            continue
        cells = np.vstack(stacks)
        cells = cells[np.isfinite(cells).any(axis=1)]  # a cell with no value adds nothing
        if not len(cells):
            continue
        n_valid = np.sum(np.isfinite(cells), axis=0)
        with _quiet_empty_means():
            mean = np.nanmean(cells, axis=0)
            sd = np.nanstd(cells, axis=0, ddof=1) if len(cells) > 1 else np.full(mean.shape, np.nan)
        sem = np.where(n_valid > 1, sd / np.sqrt(np.maximum(n_valid, 1)), np.nan)
        pooled[category] = {"mean": mean, "sem": sem, "n_cells": len(cells), "n_runs": len(stacks)}
    return pooled


def top_cells_by_max(dff: np.ndarray, n: int) -> np.ndarray:
    """The *n* cells with the largest ΔF/F0 anywhere in the trace, largest first.

    Cells whose ΔF/F0 is undefined throughout (F0 never valid) are left out.
    """
    with _quiet_empty_means():
        peaks = np.nanmax(np.asarray(dff, dtype=float), axis=1)
    order = [int(i) for i in np.argsort(-np.nan_to_num(peaks, nan=-np.inf), kind="stable")]
    return np.array([i for i in order if np.isfinite(peaks[i])][:n], dtype=int)


@contextmanager
def _quiet_empty_means():
    """All-NaN slices give NaN, as intended, without numpy's RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield
