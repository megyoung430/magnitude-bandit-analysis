"""Shared-signal statistics and residual reward responses of one imaging run (pure).

Given the cells' traces and the background's, this detrends and z-scores them, computes the
statistics of :mod:`src.funcimg_analysis.run_statistics`, and aligns two sets of traces to the
outcomes through ``pulse_responses``: the detrended cells, and the cells with the background's
fitted contribution subtracted. Only the background differs between the two, so any change in
the response is due to removing it.
"""

import math

import numpy as np
import pandas as pd

from src.funcimg_analysis.event_frames import (
    MAD_TO_SD,
    event_frames,
    frame_period_s,
    outcome_window_mask,
    quiet_empty_means,
    window_frame_counts,
)
from src.funcimg_analysis.frame_times import RunMismatchError
from src.funcimg_analysis.run_responses import check_measures, pulse_responses
from src.funcimg_analysis.run_statistics import (
    DEFAULT_NULL_KINDS,
    check_null_kinds,
    shared_signal_stats,
)
from src.funcimg_analysis.shared_signal import (
    detrended_z,
    remove_background,
    window_frames_from_s,
)
from src.funcimg_analysis.shared_signal_null import shift_offsets

Z_MEASURES = ("F z", "dF/F0 z")  # the analysis is in z units, so only the z measures
MIN_FIT_FRACTION = 0.1  # the slope needs frames outside the outcome windows to be fitted on


class SharedSignalUnavailableError(Exception):
    """A run cannot be analysed for shared signal; the message says why."""


class TooShortForSharedSignalError(SharedSignalUnavailableError):
    """A run is shorter than the detrend window or twice the minimum circular shift."""


class TooFewCellsError(SharedSignalUnavailableError):
    """A run has fewer than two ROIs, so there is no leave-one-out average."""


class NoFitFramesError(SharedSignalUnavailableError):
    """Too few frames lie outside the outcome windows to fit the background's slope on."""


def _get(source: dict, key: tuple, what: str):
    if key not in source:
        raise ValueError(f"no {what} for {key}; the run needs one for every channel analysed.")
    return source[key]


def _z_rows(traces: np.ndarray, window_frames: int, what: str) -> np.ndarray:
    """Detrended z of rows that must have a scale (a constant trace is an error)."""
    z, flat = detrended_z(traces, window_frames)
    if flat.any():
        raise ValueError(f"the {what} is constant after detrending; it has no scale.")
    return z


def outcome_fit_mask(
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    phases: tuple[str, ...],
    window_s: tuple[float, float],
    period_s: float,
) -> np.ndarray:
    """True for the frames outside every outcome window (of the outcomes in *phases*)."""
    times = outcomes[outcomes["phase"].isin(phases)]["outcome_time"].to_numpy(float)
    index, _, kept = event_frames(frame_times_s, times)
    pre, post = window_frame_counts(window_s, period_s)
    return ~outcome_window_mask(len(frame_times_s), index[kept], pre, post)


def fit_frames(
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    phases: tuple[str, ...],
    window_s: tuple[float, float],
    period_s: float,
) -> np.ndarray:
    """The frames to fit the background's slope on: those outside every outcome window.

    Raises:
        NoFitFramesError: If less than 10 % of the frames are left.
    """
    mask = outcome_fit_mask(frame_times_s, outcomes, phases, window_s, period_s)
    if mask.mean() < MIN_FIT_FRACTION:
        raise NoFitFramesError(
            f"the outcome windows cover {100 * (1 - mask.mean()):.0f}% of the frames, "
            "leaving too few to fit the background's slope on; shorten fit_exclude_window_s "
            "or set fit_exclude_outcomes=False."
        )
    return mask


def _robust_scale(traces: np.ndarray) -> np.ndarray:
    """Per-row ``1.4826 * MAD`` (NaN-safe), the scale ``robust_z`` divides by."""
    with quiet_empty_means():
        median = np.nanmedian(traces, axis=1, keepdims=True)
        return MAD_TO_SD * np.nanmedian(np.abs(traces - median), axis=1)


def _in_units_of(result: dict, scales: dict) -> dict:
    """A copy of *result* with each cell's z response multiplied by the scale of its raw view.

    ``pulse_responses`` z-scores a trace by its own scale. After removing the background that
    scale is smaller than the detrended cell's, which would inflate a surviving response, so the
    responses are multiplied back to the units of the detrended cell's z (1 = its own noise).
    """
    rescaled = {"means": {}, "sems": {}}
    for kind in rescaled:
        for view, per_pulse in result[kind].items():
            factor = scales[(view[0], view[1].removesuffix(" z"))][:, None]
            rescaled[kind][view] = {pulses: window * factor for pulses, window in per_pulse.items()}
    return {**result, **rescaled}


def analyse_shared_signal(
    views: dict,
    background: dict,
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    *,
    neuropil: dict,
    background_halves: dict,
    rng: np.random.Generator,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    measures: tuple[str, ...] = Z_MEASURES,
    detrend_window_s: float,
    min_shift_s: float,
    n_shifts: int,
    null_kinds: tuple[str, ...] = DEFAULT_NULL_KINDS,
    fit_exclude_outcomes: bool = True,
    fit_exclude_window_s: tuple[float, float] | None = None,
) -> dict:
    """Shared-signal statistics and original vs background-removed reward responses of a run.

    Args:
        views (dict): ``{(channel, "F" or "dF/F0"): (n_cells, n_frames)}`` for the cells in use.
        background (dict): ``{(channel, "F"): (n_frames,)}`` for every channel in *views*.
        frame_times_s (np.ndarray): pyControl time of each frame (NaN where unknown).
        outcomes (pd.DataFrame): ``outcome_time``, ``num_rew_pulses`` and ``phase`` per trial.
        neuropil (dict): ``{(channel, "F"): (n_cells, n_frames)}`` each cell's own neuropil trace
            (``Fneu``), the second control of the partial correlation.
        background_halves (dict): ``{(channel, "F"): (2, n_frames)}`` the F of two disjoint sets
            of background pixels, for the background's split-half reliability.
        rng (np.random.Generator): Draws the circular shifts and the reliability splits.
        window_s (tuple[float, float]): Window around the outcome (s).
        phases (tuple[str, ...]): Trial phases to include.
        measures (tuple[str, ...]): ``"F z"`` and/or ``"dF/F0 z"``: which views to analyse and
            align.
        detrend_window_s (float): Length of the running-percentile baseline (s).
        min_shift_s (float): Smallest circular shift (s), in both directions.
        n_shifts (int): Number of shifts per chance level.
        null_kinds (tuple[str, ...]): Which chance levels to compute (see ``run_statistics``).
        fit_exclude_outcomes (bool): Fit each cell's slope on the background only on frames
            outside the outcome windows, then apply it to every frame. Default True: a reward
            response the background happens to share would otherwise pull the slope towards
            itself and bias the residual response.
        fit_exclude_window_s (tuple[float, float] | None): The window left out around each
            outcome (of the outcomes in *phases*, all pulse counts). Default None, meaning
            *window_s*.

    Returns:
        dict: ``stats`` (``{(channel, measure): shared_signal_stats}``), ``original`` and
        ``residual`` (``pulse_responses`` of the detrended cells and of the cells minus the
        background's fit, the latter in units of the detrended cell's own robust scale),
        ``fit_fraction`` (share of frames the slope was fitted on), ``n_cells`` and ``period_s``.

    Raises:
        ValueError: If *measures* is empty or has a name other than ``F z`` and ``dF/F0 z``, if
            *null_kinds* has an unknown name, if the neuropil or background halves of a channel
            are missing, or a background or neuropil is constant.
        RunMismatchError: If a trace's length differs from the frame times'.
        TooFewCellsError: If there are fewer than two ROIs.
        TooShortForSharedSignalError: If the run is shorter than the detrend window or twice
            the minimum shift.
        NoFitFramesError: If less than 10 % of the frames lie outside the outcome windows.
    """
    check_measures(measures, Z_MEASURES)
    check_null_kinds(null_kinds)
    n_frames = len(frame_times_s)
    for kind, source in (
        ("cells", views),
        ("background", background),
        ("neuropil", neuropil),
        ("background halves", background_halves),
    ):
        for name, traces in source.items():
            if traces.shape[-1] != n_frames:
                raise RunMismatchError(
                    f"{kind} {name} has {traces.shape[-1]} frames but there are {n_frames} "
                    "frame times."
                )
    n_cells = next(iter(views.values())).shape[0]
    if n_cells < 2:
        raise TooFewCellsError(f"{n_cells} ROI(s): the leave-one-out average needs at least two.")
    period = frame_period_s(frame_times_s)
    window_frames = window_frames_from_s(detrend_window_s, period)
    min_shift_frames = math.ceil(min_shift_s / period)
    if window_frames > n_frames:
        raise TooShortForSharedSignalError(
            f"{n_frames} frames are fewer than the detrend window of {window_frames} frames."
        )
    if n_frames < 2 * min_shift_frames:
        raise TooShortForSharedSignalError(
            f"{n_frames} frames are too few for circular shifts of at least {min_shift_s} s."
        )
    fit_mask = None
    if fit_exclude_outcomes:
        fit_mask = fit_frames(
            frame_times_s, outcomes, phases, fit_exclude_window_s or window_s, period
        )
    offsets = shift_offsets(n_frames, min_shift_frames, n_shifts, rng)
    raw_measures = {measure.removesuffix(" z") for measure in measures}
    wanted = [key for key in views if key[1] in raw_measures]
    stats, original, residual, residual_scale = {}, {}, {}, {}
    for channel in dict.fromkeys(channel for channel, _ in wanted):
        key = (channel, "F")
        background_z = _z_rows(
            _get(background, key, "background")[None, :], window_frames, "background"
        )[0]
        halves_z = _z_rows(
            _get(background_halves, key, "background halves"), window_frames, "background half"
        )
        neuropil_z = _z_rows(_get(neuropil, key, "neuropil"), window_frames, "neuropil")
        for view in (view for view in wanted if view[0] == channel):
            z, _ = detrended_z(views[view], window_frames)
            stats[view] = shared_signal_stats(
                z,
                background_z,
                offsets,
                rng=rng,
                neuropil_z=neuropil_z,
                background_halves_z=halves_z,
                null_kinds=null_kinds,
            )
            original[view] = z
            residual[view] = remove_background(z, background_z, fit_mask)
            residual_scale[view] = _robust_scale(residual[view])
    align = dict(window_s=window_s, phases=phases, measures=measures)
    return {
        "stats": stats,
        "original": pulse_responses(original, frame_times_s, outcomes, **align),
        "residual": _in_units_of(
            pulse_responses(residual, frame_times_s, outcomes, **align), residual_scale
        ),
        "fit_fraction": 1.0 if fit_mask is None else float(fit_mask.mean()),
        "n_cells": n_cells,
        "period_s": period,
    }
