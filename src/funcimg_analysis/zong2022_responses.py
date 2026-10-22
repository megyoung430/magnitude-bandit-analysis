"""Outcome-aligned responses and the shuffle test for responsive cells (pure).

The test follows the escape-active analysis of Zong et al. 2022 (STAR Methods, "escape-active
cells"): a score ``(A_event - A_other) / (A_event + A_other)`` per cell, compared with the scores
of the same activity circularly time-shifted against the behaviour.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.funcimg_analysis.event_frames import (
    event_frames,
    frame_period_s,
    frame_windows,
    mean_by_category,
    outcome_window_mask,
    pooled_mean_sem,
    quiet_empty_means,
    window_frame_counts,
    window_offsets,
)
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES

NULL_PERCENTILE = 95  # a cell is responsive above this percentile of its shuffled scores
CHANCE_LEVEL = 1 - NULL_PERCENTILE / 100


def valid_span(frame_times_s: np.ndarray) -> slice:
    """The frames from the first to the last with a pyControl time.

    Raises:
        ValueError: If no frame has a time, or a frame inside the span lacks one (the sync
            only leaves frames without a time outside its pulses, so a gap inside is a bug).
    """
    known = np.flatnonzero(np.isfinite(frame_times_s))
    if not len(known):
        raise ValueError("no frame has a pyControl time.")
    span = slice(int(known[0]), int(known[-1]) + 1)
    if not np.isfinite(np.asarray(frame_times_s)[span]).all():
        raise ValueError("frame times have a gap inside the synchronised span.")
    return span


@dataclass(frozen=True)
class AlignedWindows:
    """Windows of whole frames around the outcomes that have a complete window.

    Attributes:
        offsets (np.ndarray): Frame offsets of the window (0 is the outcome's frame).
        period_s (float): Frame period.
        windows (np.ndarray): ``(n_cells, n_events, n_window)``.
        pulses (np.ndarray): Reward pulses of each kept event.
        event_index (np.ndarray): Frame of each kept event.
        n_dropped (int): Outcomes of the requested phases lost to a missing or incomplete window.
    """

    offsets: np.ndarray
    period_s: float
    windows: np.ndarray
    pulses: np.ndarray
    event_index: np.ndarray
    n_dropped: int


def aligned_windows(
    traces: np.ndarray,
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
) -> AlignedWindows:
    """Cut every cell's trace into whole-frame windows around each outcome.

    Args:
        traces (np.ndarray): ``(n_cells, n_frames)``, same frames as *frame_times_s*.
        frame_times_s (np.ndarray): pyControl time of each frame, NaN where unknown.
        outcomes (pd.DataFrame): ``outcome_time``, ``num_rew_pulses`` and ``phase`` per trial.
        window_s (tuple[float, float]): Window around the outcome (s), rounded to whole frames.
        phases (tuple[str, ...]): Trial phases to include.

    Returns:
        AlignedWindows: Only outcomes within half a frame of a timed frame whose whole window
        lies on timed frames are kept.
    """
    period = frame_period_s(frame_times_s)
    pre, post = window_frame_counts(window_s, period)
    trials = outcomes[outcomes["phase"].isin(phases)]
    index, _, near_frame = event_frames(frame_times_s, trials["outcome_time"].to_numpy(float))
    windows, complete = frame_windows(
        traces, index[near_frame], pre, post, np.isfinite(frame_times_s)
    )
    kept = np.flatnonzero(near_frame)[complete]
    return AlignedWindows(
        offsets=window_offsets(pre, post),
        period_s=period,
        windows=windows,
        pulses=trials["num_rew_pulses"].to_numpy(int)[kept],
        event_index=index[kept],
        n_dropped=int(len(trials) - len(kept)),
    )


def pooled_responses(
    aligned: AlignedWindows, cell_mask: np.ndarray, categories: list[int] = PULSE_CATEGORIES
) -> dict:
    """Mean ± SEM across the selected cells of each cell's trial-averaged window, per pulse count.

    Returns:
        dict: ``{pulses: {"mean", "sem", "n_cells", "n_runs"}}`` (see ``pooled_mean_sem``);
        empty when no cell is selected.
    """
    if not np.asarray(cell_mask).any():
        return {}
    means = mean_by_category(aligned.windows[cell_mask], aligned.pulses, categories)
    return pooled_mean_sem([means], categories)


# ---- responsive cells --------------------------------------------------------------------------


def response_frames(response_s: float, period_s: float) -> int:
    """Frames whose start lies in a response window of *response_s* seconds (at least one)."""
    return max(1, int(round(response_s / period_s)))


def reward_frame_mask(
    n_frames: int,
    event_index: np.ndarray,
    pulses: np.ndarray,
    n_response_frames: int,
    rewarded: bool,
) -> np.ndarray:
    """True on the response window after each rewarded outcome (``rewarded=False``: unrewarded).

    The window is the outcome's frame and the ``n_response_frames - 1`` after it.
    """
    pulses = np.asarray(pulses)
    chosen = np.asarray(event_index)[pulses > 0 if rewarded else pulses == 0]
    return outcome_window_mask(n_frames, chosen, 0, n_response_frames - 1)


def _check_mask(mask: np.ndarray) -> None:
    if not mask.any() or mask.all():
        raise ValueError(
            "The response window must cover some frames but not all: "
            f"{int(mask.sum())} of {len(mask)} frames are inside it."
        )


def distinction_scores(traces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(A_in - A_out) / (A_in + A_out)`` per cell, A being the mean activity in/outside *mask*.

    NaN where the denominator is zero (a cell with no activity) or the trace is NaN.
    """
    _check_mask(mask)
    inside, outside = traces[:, mask].mean(axis=1), traces[:, ~mask].mean(axis=1)
    return _contrast(inside, outside)


def _contrast(inside: np.ndarray, outside: np.ndarray) -> np.ndarray:
    total = inside + outside
    return np.divide(inside - outside, total, out=np.full(total.shape, np.nan), where=total != 0)


def shifted_distinction_scores(
    traces: np.ndarray, mask: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    """Scores of each cell's activity circularly shifted by each of *shifts* frames.

    Equivalent to ``np.roll(trace, s)`` followed by ``distinction_scores``, but all shifts come
    from one FFT cross-correlation of the mask with the trace.

    Returns:
        np.ndarray: ``(n_cells, n_shifts)``.
    """
    _check_mask(mask)
    n_frames = traces.shape[1]
    # Σ_i mask[i] * trace[i - s] is the cross-correlation of the mask with the trace at lag s
    correlation = np.fft.irfft(
        np.fft.rfft(mask.astype(float))[None] * np.conj(np.fft.rfft(traces, axis=1)),
        n=n_frames,
        axis=1,
    )
    n_in = int(mask.sum())
    inside = correlation[:, np.asarray(shifts, dtype=int)] / n_in
    outside = (traces.sum(axis=1, keepdims=True) - correlation[:, shifts]) / (n_frames - n_in)
    return _contrast(inside, outside)


def circular_shifts(
    n_frames: int, period_s: float, n_shifts: int, min_shift_s: float, rng: np.random.Generator
) -> np.ndarray:
    """Random shifts (frames) at least *min_shift_s* from either end of the recording.

    Raises:
        ValueError: If the recording is shorter than twice the minimum shift.
    """
    low = int(np.ceil(min_shift_s / period_s))
    high = n_frames - low
    if high < low:
        raise ValueError(
            f"The {n_frames}-frame recording is shorter than twice the {min_shift_s} s "
            "minimum shift."
        )
    return rng.integers(low, high + 1, n_shifts)


def responsive_cells(
    traces: np.ndarray, mask: np.ndarray, shifts: np.ndarray, percentile: float = NULL_PERCENTILE
) -> dict:
    """Shuffle test: a cell is responsive if its score exceeds the shuffled scores' percentile.

    Args:
        traces (np.ndarray): ``(n_cells, n_frames)`` activity, e.g. ``ΔF/F_clear``.
        mask (np.ndarray): Frames inside the response windows.
        shifts (np.ndarray): Circular shifts (frames) forming the null.
        percentile (float): Percentile of the null a cell must exceed. Default 95.

    Returns:
        dict: ``score``, ``null_95th`` (the *percentile* of each cell's null), ``p_value``
        (share of shuffles at least as large, counting the real one) and ``responsive``, each
        ``(n_cells,)``. A cell with NaN score is never responsive.
    """
    score = distinction_scores(traces, mask)
    null = shifted_distinction_scores(traces, mask, shifts)
    with quiet_empty_means(), np.errstate(invalid="ignore"):
        threshold = np.nanpercentile(null, percentile, axis=1)
        at_least = (null >= score[:, None]).sum(axis=1)
        responsive = score > threshold
    p_value = np.where(np.isfinite(score), (1 + at_least) / (1 + len(shifts)), np.nan)
    return {"score": score, "null_95th": threshold, "p_value": p_value, "responsive": responsive}


def chance_summary(responsive: np.ndarray, chance: float = CHANCE_LEVEL) -> dict:
    """Responsive cells against the share expected when nothing responds.

    Returns:
        dict: ``n_cells``, ``n_responsive``, ``fraction``, ``expected_by_chance`` and the
        two-sided binomial ``p_binomial`` of the count under the chance level. The cells are
        not independent, so the p value is a rough guide.
    """
    flags = np.asarray(responsive, dtype=bool)
    n, k = len(flags), int(flags.sum())
    return {
        "n_cells": n,
        "n_responsive": k,
        "fraction": k / n if n else np.nan,
        "expected_by_chance": chance * n,
        "p_binomial": float(binomtest(k, n, chance).pvalue) if n else np.nan,
    }
