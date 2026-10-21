"""Which cells respond to the reward, ranked with cross-validation against selection bias (pure).

A cell's *response index* summarises how much its trials change from the baseline before the
outcome to the response after it. Ranking cells by the index of the same trials that are then
plotted selects noise: the cells at the top of a ranking of noise already look responsive.
The trials are therefore split into two halves, the cells are ranked on half A and shown on half
B, and a circular-shift null says how many cells would exceed their own chance level without any
alignment between the cell and the outcomes.

The index of one trial is the mean over the response range minus the mean over the baseline range
(whole frames starting inside the range; the frame the outcome fell in is left out of the
baseline of a trial where the outcome came before that frame's start). A cell's index is then:

- ``"rewarded"``: the mean over the rewarded trials (1-4 pulses) divided by their standard
  deviation, an effect size that does not grow with the number of trials;
- ``"4_vs_0"``: Welch's t of the 4-pulse trials against the 0-pulse trials, which does grow
  with the number of trials.
"""

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.funcimg_analysis.event_frames import (
    event_frames,
    frame_period_s,
    frame_windows,
    mean_by_category,
    quiet_empty_means,
    robust_z,
    sem_by_category,
    window_frame_counts,
    window_offsets,
)
from src.funcimg_analysis.frame_times import RunMismatchError
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES
from src.funcimg_analysis.shared_signal_null import NULL_PERCENTILE, null_percentile, shift_offsets

CONTRASTS = ("rewarded", "4_vs_0")
RANK_ON = ("split_half", "all")
CHANCE_FRACTION = 1 - NULL_PERCENTILE / 100  # exceeding a 95th percentile happens 5 % of the time
MIN_TRIALS_FOR_SD = 2  # a standard deviation needs two trials
DEFAULT_MIN_TRIALS_PER_GROUP = 3  # per group and half; fewer make the index too noisy to rank
FRAME_TOLERANCE = 1e-9  # s; a frame starting on a range edge belongs to the range


class TooFewTrialsError(Exception):
    """A run has too few trials in a half (or in total) to compute the response index."""


def _in_range(offsets: np.ndarray, period_s: float, range_s: tuple[float, float]) -> np.ndarray:
    """Offsets whose frames start within ``[range_s[0], range_s[1])`` of the outcome's frame."""
    start = np.asarray(offsets) * period_s
    inside = (start >= range_s[0] - FRAME_TOLERANCE) & (start < range_s[1] - FRAME_TOLERANCE)
    if not inside.any():
        raise ValueError(f"no frames start within {range_s} s of the outcome.")
    return np.asarray(offsets)[inside]


def _window_means(
    traces: np.ndarray,
    event_index: np.ndarray,
    offsets: np.ndarray,
    wrap: bool,
    skip_last_pre_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Mean over the frames at *offsets* around each event: ``(n_cells, n_events)``, NaN-safe.

    With *wrap* the frames wrap around the ends of the trace (a circular shift). Where
    *skip_last_pre_frame* is True the frame at offset -1 is left out of that event's mean.
    """
    frames = np.asarray(event_index)[:, None] + np.asarray(offsets)[None, :]
    n_frames = traces.shape[1]
    values = traces[:, frames % n_frames if wrap else frames]
    if skip_last_pre_frame is not None:
        skipped = np.asarray(skip_last_pre_frame, dtype=bool)[:, None] & (offsets == -1)[None, :]
        values = np.where(skipped[None], np.nan, values)
    with quiet_empty_means():
        return np.nanmean(values, axis=2)


def trial_responses(
    traces: np.ndarray,
    event_index: np.ndarray,
    offsets: np.ndarray,
    period_s: float,
    baseline_s: tuple[float, float],
    response_s: tuple[float, float],
    baseline_spanned: np.ndarray | None = None,
    wrap: bool = False,
) -> np.ndarray:
    """Per cell and trial: the mean over the response range minus the mean over the baseline range.

    Args:
        traces (np.ndarray): ``(n_cells, n_frames)``.
        event_index (np.ndarray): Frame of each trial's outcome.
        offsets (np.ndarray): Frame offsets available around each outcome (0 is its frame).
        period_s (float): Frame period (s).
        baseline_s (tuple[float, float]): Baseline range (s), frames starting in ``[start, stop)``.
        response_s (tuple[float, float]): Response range (s), the same convention.
        baseline_spanned (np.ndarray | None): Per trial, True if the outcome came before its
            frame's start, so the frame at offset -1 was being acquired and is left out of the
            baseline. Default None, meaning none.
        wrap (bool): Wrap frames around the ends of the trace, for circularly shifted outcomes.

    Returns:
        np.ndarray: ``(n_cells, n_trials)``; NaN where a range has no finite frame.

    Raises:
        ValueError: If a range contains no frame of *offsets*.
    """
    base = _in_range(offsets, period_s, baseline_s)
    response = _in_range(offsets, period_s, response_s)
    return _window_means(traces, event_index, response, wrap) - _window_means(
        traces, event_index, base, wrap, baseline_spanned
    )


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: mean, sample variance (ddof 1) and number of finite values."""
    finite = np.isfinite(values)
    n = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite, values, 0.0).sum(axis=1) / n
        squares = np.where(finite, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
        return mean, squares / (n - 1), n


def response_index(
    per_trial: np.ndarray, pulses: np.ndarray, contrast: str, trial_mask: np.ndarray
) -> np.ndarray:
    """Each cell's response index over the trials in *trial_mask*.

    Args:
        per_trial (np.ndarray): ``(n_cells, n_trials)`` from :func:`trial_responses`.
        pulses (np.ndarray): Reward pulses of each trial.
        contrast (str): ``"rewarded"`` (mean / SD over trials with 1-4 pulses) or ``"4_vs_0"``
            (Welch's t of the 4-pulse against the 0-pulse trials).
        trial_mask (np.ndarray): Boolean per trial; the trials to use.

    Returns:
        np.ndarray: ``(n_cells,)``; NaN where fewer than two trials have a value, or there is no
        spread, since the index would be infinite or a guess.

    Raises:
        ValueError: If *contrast* is not one of ``CONTRASTS``.
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, not {contrast!r}.")
    pulses, use = np.asarray(pulses), np.asarray(trial_mask, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        if contrast == "rewarded":
            mean, variance, n = _moments(per_trial[:, use & (pulses >= 1)])
            index = mean / np.sqrt(variance)
            defined = (n >= MIN_TRIALS_FOR_SD) & (variance > 0)
        else:
            mean_4, var_4, n_4 = _moments(per_trial[:, use & (pulses == 4)])
            mean_0, var_0, n_0 = _moments(per_trial[:, use & (pulses == 0)])
            spread = np.sqrt(var_4 / n_4 + var_0 / n_0)
            index = (mean_4 - mean_0) / spread
            defined = (n_4 >= MIN_TRIALS_FOR_SD) & (n_0 >= MIN_TRIALS_FOR_SD) & (spread > 0)
    return np.where(defined, index, np.nan)


def check_enough_trials(
    pulses: np.ndarray, masks: dict[str, np.ndarray], contrast: str, minimum: int
) -> None:
    """Raise unless every mask has at least *minimum* trials in each group the contrast compares.

    Raises:
        TooFewTrialsError: Naming the mask and the group that is short, and what to do about it.
    """
    pulses = np.asarray(pulses)
    groups = {
        "rewarded": {"rewarded": pulses >= 1},
        "4_vs_0": {"4-pulse": pulses == 4, "0-pulse": pulses == 0},
    }
    for name, mask in masks.items():
        for group, in_group in groups[contrast].items():
            n = int((mask & in_group).sum())
            if n < minimum:
                advice = (
                    'use the "rewarded" contrast (it pools the 1-4 pulse trials) or collect more '
                    "sessions"
                    if contrast == "4_vs_0"
                    else "collect more sessions or include more phases"
                )
                raise TooFewTrialsError(
                    f"{n} {group} trial(s) in {name} ({contrast}); the response index needs at "
                    f"least {minimum} per group and half to be rankable: {advice}."
                )


def split_halves(pulses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Which trials are in half A: they alternate A, B, A, ... in time within each pulse count.

    Alternating (rather than a random split) pairs each trial with its neighbour in time, so slow
    changes over the run are shared by both halves, and each half gets (nearly) the same number
    of trials of every pulse count. A pulse count with an odd number of trials gives its extra
    trial to the half it starts with; the starting half alternates from one pulse count to the
    next, so the extras balance across the halves. Which half the first pulse count starts with
    is drawn from *rng*.

    Args:
        pulses (np.ndarray): Reward pulses of each trial, in time order.
        rng (np.random.Generator): Chooses the starting half of the first pulse count.

    Returns:
        np.ndarray: Boolean per trial, True for half A.
    """
    pulses = np.asarray(pulses)
    half_a = np.zeros(len(pulses), dtype=bool)
    first = int(rng.integers(2))
    for k, category in enumerate(np.unique(pulses)):
        trials = np.flatnonzero(pulses == category)
        half_a[trials] = (np.arange(len(trials)) + first + k) % 2 == 0
    return half_a


def exceeds_null(index: np.ndarray, null: np.ndarray, q: float = NULL_PERCENTILE) -> np.ndarray:
    """Whether each cell's index is above the *q*-th percentile of its own null.

    Args:
        index (np.ndarray): ``(n_cells,)``.
        null (np.ndarray): ``(n_shifts, n_cells)`` indices of the circularly shifted outcomes.
        q (float): Percentile.

    Returns:
        np.ndarray: Boolean per cell; False where the index or the null is undefined.
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(index) > null_percentile(null, q)


def _rank_correlation(index_a: np.ndarray, index_b: np.ndarray) -> float:
    """Spearman correlation of the two halves' indices over the cells defined in both."""
    ok = np.isfinite(index_a) & np.isfinite(index_b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(index_a[ok], index_b[ok]).statistic)


def _null_summary(indices: dict, nulls: dict) -> dict:
    """Counts of cells above their null on the ranking trials, and on the held-out trials."""
    above = {name: exceeds_null(indices[name], nulls[name]) for name in indices}
    first = next(iter(indices))
    n_cells = int(np.isfinite(indices[first]).sum())
    summary = {"n_cells": n_cells, f"n_exceed_{first}": int(above[first].sum())}
    summary[f"expected_{first}"] = CHANCE_FRACTION * n_cells
    if "b" in above:
        both = above["a"] & above["b"]
        summary["n_exceed_b"] = int(above["b"].sum())
        summary["n_exceed_b_of_a"] = int(both.sum())
        # among cells chosen without regard to the held-out half, 5 % are above its null by chance
        summary["expected_b_of_a"] = CHANCE_FRACTION * summary["n_exceed_a"]
    return summary


class _Trials(NamedTuple):
    """The trials of a run that have a complete window of imaged frames."""

    event_index: np.ndarray  # frame of each outcome
    spanned: np.ndarray  # outcome came before its frame's start (frame -1 was being acquired)
    pulses: np.ndarray
    masks: dict[str, np.ndarray]  # {"a": half A, "b": half B} or {"all": every trial}


class _Ranges(NamedTuple):
    """How a trial's response is measured, and the frame offsets available around an outcome."""

    offsets: np.ndarray
    period_s: float
    baseline_s: tuple[float, float]
    response_s: tuple[float, float]
    contrast: str


def _null_indices(z: np.ndarray, trials: _Trials, ranges: _Ranges, shifts: np.ndarray) -> dict:
    """Index per cell with the outcomes circularly shifted by each shift: ``{mask: (S, cells)}``."""
    nulls = {name: np.empty((len(shifts), len(z))) for name in trials.masks}
    for i, shift in enumerate(shifts):
        shifted = trial_responses(
            z,
            trials.event_index + shift,
            ranges.offsets,
            ranges.period_s,
            ranges.baseline_s,
            ranges.response_s,
            trials.spanned,
            wrap=True,
        )
        for name, mask in trials.masks.items():
            nulls[name][i] = response_index(shifted, trials.pulses, ranges.contrast, mask)
    return nulls


def _plotted_responses(
    z: np.ndarray,
    trials: _Trials,
    ranges: _Ranges,
    plot_frames: tuple[int, int],
    plot_mask: np.ndarray,
) -> dict:
    """Trial-averaged windows of the plotted trials, each trial minus its own baseline mean."""
    all_frames = np.ones(z.shape[1], dtype=bool)  # the trials already have complete windows
    windows, _ = frame_windows(z, trials.event_index, *plot_frames, all_frames)
    baseline_offsets = _in_range(ranges.offsets, ranges.period_s, ranges.baseline_s)
    baseline = _window_means(z, trials.event_index, baseline_offsets, False, trials.spanned)
    windows = (windows - baseline[:, :, None])[:, plot_mask]
    pulses = trials.pulses[plot_mask]
    with quiet_empty_means():
        rewarded_mean = np.nanmean(windows[:, pulses >= 1], axis=1)
    means = mean_by_category(windows, pulses, PULSE_CATEGORIES)
    ranked_mean = rewarded_mean if ranges.contrast == "rewarded" else means[4] - means[0]
    return {
        "means": means,
        "sems": sem_by_category(windows, pulses, PULSE_CATEGORIES),
        "rewarded_mean": rewarded_mean,
        "ranked_mean": ranked_mean,  # the response the index measures: what the heatmap shows
    }


def _analyse_measure(
    trace: np.ndarray,
    trials: _Trials,
    ranges: _Ranges,
    shifts: np.ndarray,
    plot_frames: tuple[int, int],
    plot_mask: np.ndarray,
) -> dict:
    """Indices, null and plotted responses of one measure (see :func:`analyse_reward_cells`)."""
    z, flat = robust_z(trace)
    per_trial = trial_responses(
        z,
        trials.event_index,
        ranges.offsets,
        ranges.period_s,
        ranges.baseline_s,
        ranges.response_s,
        trials.spanned,
    )
    indices = {
        name: response_index(per_trial, trials.pulses, ranges.contrast, mask)
        for name, mask in trials.masks.items()
    }
    rank_index = indices.get("a", indices.get("all"))
    ranked = np.flatnonzero(np.isfinite(rank_index))
    rank_correlation = (
        _rank_correlation(indices["a"], indices["b"]) if "a" in indices else float("nan")
    )
    return {
        **{f"index_{name}": index for name, index in indices.items()},
        "rank_index": rank_index,
        "order": ranked[np.argsort(-rank_index[ranked], kind="stable")],
        "rank_correlation": rank_correlation,
        "null": _null_summary(indices, _null_indices(z, trials, ranges, shifts)),
        "flat_cells": np.flatnonzero(flat).tolist(),
        **_plotted_responses(z, trials, ranges, plot_frames, plot_mask),
    }


def _trial_counts(pulses: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    return {c: int(np.sum(pulses[mask] == c)) for c in PULSE_CATEGORIES}


def analyse_reward_cells(
    traces: dict[str, np.ndarray],
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    *,
    phases: tuple[str, ...],
    plot_window_s: tuple[float, float],
    baseline_s: tuple[float, float],
    response_s: tuple[float, float],
    contrast: str,
    rank_on: str,
    n_shifts: int,
    min_shift_s: float,
    rng: np.random.Generator,
    min_trials_per_group: int = DEFAULT_MIN_TRIALS_PER_GROUP,
) -> dict:
    """Rank the cells of a run by their reward response, cross-validated, with a shifted null.

    Args:
        traces (dict[str, np.ndarray]): ``{measure: (n_cells, n_frames)}``, e.g. ΔF/F0 of each
            kind; each is robust-z-scored per cell here (the index does not depend on scale).
        frame_times_s (np.ndarray): pyControl time of each frame (NaN where unknown).
        outcomes (pd.DataFrame): ``outcome_time``, ``num_rew_pulses`` and ``phase`` per trial.
        phases (tuple[str, ...]): Trial phases to include.
        plot_window_s (tuple[float, float]): Window of the plotted responses (s).
        baseline_s (tuple[float, float]): Baseline range of a trial (s).
        response_s (tuple[float, float]): Response range of a trial (s).
        contrast (str): ``"rewarded"`` or ``"4_vs_0"`` (see :func:`response_index`).
        rank_on (str): ``"split_half"``: rank on half A, plot half B. ``"all"``: rank and plot
            all trials, which is biased towards cells that happen to look responsive.
        n_shifts (int): Circular shifts of the outcomes for the null.
        min_shift_s (float): Smallest circular shift (s), in both directions.
        rng (np.random.Generator): Draws the split and the shifts.
        min_trials_per_group (int): Fewest trials of each group the contrast compares, in each
            half (or in all trials), below which the run is refused. Default 3.

    Returns:
        dict: ``period_s``, ``offsets`` (frames of the plotted window), ``pulses``, ``halves``
        (True for half A), ``rank_mask`` and ``plot_mask`` (the trials ranked and plotted),
        ``rank_on``, ``cross_validated``, ``contrast``, ``n_trials``, ``n_trials_plot`` and
        ``n_trials_by_half`` (``{"a": ..., "b": ...}`` or ``{"all": ...}``) per pulse count,
        ``n_events_dropped``, and per measure under ``measures``: ``index_a`` and ``index_b``
        (or ``index_all``), ``rank_index``, ``order`` (cells with a rank index, highest first),
        ``rank_correlation`` of the two halves, ``null`` (counts above the shifted null, with
        the counts expected by chance), ``means``, ``sems``, ``rewarded_mean`` and
        ``ranked_mean`` (the response the index measures: the rewarded mean, or 4-pulse minus
        0-pulse) of the plotted trials, and ``flat_cells`` (no scale).

    Raises:
        ValueError: If *contrast* or *rank_on* is unknown, a range has no frames, or the run is
            too short for the circular shifts.
        RunMismatchError: If a trace's length differs from the frame times'.
        TooFewTrialsError: If a half has fewer than *min_trials_per_group* trials of a group the
            contrast needs.
    """
    if rank_on not in RANK_ON:
        raise ValueError(f"rank_on must be one of {RANK_ON}, not {rank_on!r}.")
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, not {contrast!r}.")
    n_frames = len(frame_times_s)
    for name, trace in traces.items():
        if trace.shape[1] != n_frames:
            raise RunMismatchError(
                f"{name} has {trace.shape[1]} frames but there are {n_frames} frame times."
            )
    period = frame_period_s(frame_times_s)
    needed = [plot_window_s, baseline_s, response_s]
    hull = (min(w[0] for w in needed), max(w[1] for w in needed))
    pre, post = window_frame_counts(hull, period)
    trials = outcomes[outcomes["phase"].isin(phases)]
    index, offset_s, near = event_frames(frame_times_s, trials["outcome_time"].to_numpy(float))
    probe = next(iter(traces.values()))[:1]
    _, complete = frame_windows(probe, index[near], pre, post, np.isfinite(frame_times_s))
    kept = np.flatnonzero(near)[complete]
    pulses = trials["num_rew_pulses"].to_numpy(int)[kept]
    halves = split_halves(pulses, rng)
    everything = np.ones(len(pulses), dtype=bool)
    split = rank_on == "split_half"
    masks = {"a": halves, "b": ~halves} if split else {"all": everything}
    check_enough_trials(pulses, masks, contrast, min_trials_per_group)
    shifts = shift_offsets(n_frames, math.ceil(min_shift_s / period), n_shifts, rng)
    run_trials = _Trials(index[kept], offset_s[kept] < 0, pulses, masks)
    ranges = _Ranges(window_offsets(pre, post), period, baseline_s, response_s, contrast)
    plot_frames = window_frame_counts(plot_window_s, period)
    rank_mask, plot_mask = (halves, ~halves) if split else (everything, everything)
    return {
        "period_s": period,
        "offsets": window_offsets(*plot_frames),
        "pulses": pulses,
        "halves": halves,
        "rank_mask": rank_mask,
        "plot_mask": plot_mask,
        "rank_on": rank_on,
        "cross_validated": split,
        "contrast": contrast,
        "n_trials": _trial_counts(pulses, everything),
        "n_trials_plot": _trial_counts(pulses, plot_mask),
        "n_trials_by_half": {name: _trial_counts(pulses, mask) for name, mask in masks.items()},
        "n_events_dropped": int(len(trials) - len(kept)),
        "measures": {
            name: _analyse_measure(trace, run_trials, ranges, shifts, plot_frames, plot_mask)
            for name, trace in traces.items()
        },
    }
