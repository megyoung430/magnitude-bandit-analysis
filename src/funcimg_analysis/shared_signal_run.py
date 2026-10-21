"""Shared-signal statistics and residual reward responses of one imaging run (pure).

Given the cells' traces and the background's, this detrends and z-scores them, computes the
statistics of :mod:`src.funcimg_analysis.shared_signal`, and aligns two sets of traces to the
outcomes through ``pulse_responses``: the detrended cells, and the cells with the background's
fitted contribution subtracted. Only the background differs between the two, so any change in
the response is due to removing it.
"""

import math

import numpy as np
import pandas as pd

from src.funcimg_analysis.event_frames import MAD_TO_SD, frame_period_s, quiet_empty_means
from src.funcimg_analysis.frame_times import RunMismatchError
from src.funcimg_analysis.run_responses import check_measures, pulse_responses
from src.funcimg_analysis.shared_signal import (
    detrended_z,
    remove_background,
    shared_signal_stats,
    shift_offsets,
    window_frames_from_s,
)

Z_MEASURES = ("F z", "dF/F0 z")  # the analysis is in z units, so only the z measures


class TooShortForSharedSignalError(Exception):
    """A run is shorter than the detrend window or twice the minimum circular shift."""


def _background_z(background: dict, channel: str, window_frames: int) -> np.ndarray:
    z, flat = detrended_z(background[(channel, "F")][None, :], window_frames)
    if flat[0]:
        raise ValueError(f"the {channel} background is constant after detrending; it has no scale.")
    return z[0]


def _robust_scale(traces: np.ndarray) -> np.ndarray:
    """Per-row ``1.4826 * MAD`` (NaN-safe), the scale ``robust_z`` divides by."""
    with quiet_empty_means():
        median = np.nanmedian(traces, axis=1, keepdims=True)
        return MAD_TO_SD * np.nanmedian(np.abs(traces - median), axis=1)


def _in_units_of(result: dict, scales: dict) -> dict:
    """Multiply each cell's z response by *scales* of its raw view, undoing the re-z-scoring.

    ``pulse_responses`` z-scores a trace by its own scale. After removing the background that
    scale is smaller than the detrended cell's, which would inflate a surviving response, so the
    responses are multiplied back to the units of the detrended cell's z (1 = its own noise).
    """
    for view in result["means"]:
        factor = scales[(view[0], view[1].removesuffix(" z"))][:, None]
        for table in (result["means"][view], result["sems"][view]):
            for pulses in table:
                table[pulses] = table[pulses] * factor
    return result


def analyse_shared_signal(
    views: dict,
    background: dict,
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    *,
    rng: np.random.Generator,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    measures: tuple[str, ...] = Z_MEASURES,
    detrend_window_s: float,
    min_shift_s: float,
    n_shifts: int,
) -> dict:
    """Shared-signal statistics and original vs background-removed reward responses of a run.

    Args:
        views (dict): ``{(channel, "F" or "dF/F0"): (n_cells, n_frames)}`` for the cells in use.
        background (dict): ``{(channel, "F"): (n_frames,)}`` for every channel in *views*.
        frame_times_s (np.ndarray): pyControl time of each frame (NaN where unknown).
        outcomes (pd.DataFrame): ``outcome_time``, ``num_rew_pulses`` and ``phase`` per trial.
        rng (np.random.Generator): Draws the circular shifts of the chance levels.
        window_s (tuple[float, float]): Window around the outcome (s).
        phases (tuple[str, ...]): Trial phases to include.
        measures (tuple[str, ...]): ``"F z"`` and/or ``"dF/F0 z"``: which views to analyse and
            align.
        detrend_window_s (float): Length of the running-percentile baseline (s).
        min_shift_s (float): Smallest circular shift (s), in both directions.
        n_shifts (int): Number of shifts per chance level.

    Returns:
        dict: ``stats`` (``{(channel, measure): shared_signal_stats}``), ``original`` and
        ``residual`` (``pulse_responses`` of the detrended cells and of the cells minus the
        background's fit, the latter in units of the detrended cell's own robust scale),
        ``n_cells`` and ``period_s``.

    Raises:
        ValueError: If *measures* is empty or has a name other than ``F z`` and ``dF/F0 z``,
            or a background is constant.
        RunMismatchError: If a trace's length differs from the frame times'.
        TooShortForSharedSignalError: If the run is shorter than the detrend window or twice
            the minimum shift.
    """
    check_measures(measures, Z_MEASURES)
    n_frames = len(frame_times_s)
    for name, traces in {**views, **background}.items():
        if traces.shape[-1] != n_frames:
            raise RunMismatchError(
                f"{name} has {traces.shape[-1]} frames but there are {n_frames} frame times."
            )
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
    offsets = shift_offsets(n_frames, min_shift_frames, n_shifts, rng)
    raw_measures = {measure.removesuffix(" z") for measure in measures}
    wanted = [key for key in views if key[1] in raw_measures]
    stats, original, residual, residual_scale = {}, {}, {}, {}
    for channel in dict.fromkeys(channel for channel, _ in wanted):
        background_z = _background_z(background, channel, window_frames)
        for key in (key for key in wanted if key[0] == channel):
            z, _ = detrended_z(views[key], window_frames)
            stats[key] = shared_signal_stats(z, background_z, offsets)
            original[key], residual[key] = z, remove_background(z, background_z)
            residual_scale[key] = _robust_scale(residual[key])
    align = dict(window_s=window_s, phases=phases, measures=measures)
    return {
        "stats": stats,
        "original": pulse_responses(original, frame_times_s, outcomes, **align),
        "residual": _in_units_of(
            pulse_responses(residual, frame_times_s, outcomes, **align), residual_scale
        ),
        "n_cells": next(iter(views.values())).shape[0],
        "period_s": period,
    }
