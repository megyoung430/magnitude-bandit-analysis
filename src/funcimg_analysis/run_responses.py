"""Reward-aligned responses of every cell of one imaging run, by number of reward pulses.

Outcome times come from ``src.behavior_analysis.outcome_events`` (reward onset for rewarded
trials, no-reward cue onset for unrewarded ones; ``num_rew_pulses`` 0-4), the same definitions as
the photometry analysis. Four views of each cell are aligned: raw F and ΔF/F0 of the green
(functional) channel and of the red (structural) channel. ΔF/F0 is computed by
``mini2p_preprocessing`` with the fluorescence settings recorded at extraction, so it matches the
preprocessing QC.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.funcimg_analysis.event_frames import (
    baseline_subtract,
    event_frames,
    frame_period_s,
    frame_windows,
    mean_by_category,
    robust_z,
    sem_by_category,
    trials_without_baseline,
    window_frame_counts,
    window_offsets,
)
from src.funcimg_analysis.frame_times import RunMismatchError

RAW_VIEWS = (("green", "F"), ("green", "dF/F0"), ("red", "F"), ("red", "dF/F0"))
# Robust z over each cell's whole trace, then each trial's pre-event mean subtracted.
Z_VIEWS = tuple((channel, f"{measure} z") for channel, measure in RAW_VIEWS)
VIEWS = RAW_VIEWS + Z_VIEWS
MEASURES = ("F", "dF/F0", "F z", "dF/F0 z")
BACKGROUND_MEASURES = ("F", "F z")  # a background has no neuropil or F0, so no ΔF/F0
MIN_PRE_FRAMES_FOR_Z = 2  # the frame an event falls in can be left out, leaving one fewer
PULSE_CATEGORIES = [0, 1, 2, 3, 4]
EXTRACTION_RECORD = "cellpose_extraction.json"


def check_measures(measures: tuple[str, ...], valid: tuple[str, ...] = MEASURES) -> None:
    """Raise unless *measures* is a non-empty collection of names in *valid*.

    Raises:
        TypeError: If *measures* is a single string rather than a tuple of names.
        ValueError: If it is empty or has a name not in *valid* (the valid names are listed).
    """
    if isinstance(measures, str):
        raise TypeError(f"measures must be a tuple of names, e.g. ({measures!r},), not a string.")
    if not measures:
        raise ValueError(f"Give at least one measure; the valid measures are {list(valid)}.")
    unknown = [measure for measure in measures if measure not in valid]
    if unknown:
        raise ValueError(f"unknown measure(s) {unknown}; the valid measures are {list(valid)}.")


def pulse_responses(
    views: dict,
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    categories: list[int] = PULSE_CATEGORIES,
    measures: tuple[str, ...] = MEASURES,
) -> dict:
    """Each cell's mean window around the outcome, per pulse count, for every view.

    Args:
        views (dict): ``{(channel, measure): (n_cells, n_frames)}`` raw views (``RAW_VIEWS``);
            the matching ``Z_VIEWS`` are added here.
        frame_times_s (np.ndarray): pyControl time of each frame (NaN where unknown).
        outcomes (pd.DataFrame): ``outcome_time``, ``num_rew_pulses`` and ``phase`` per trial.
        window_s (tuple[float, float]): Window around the outcome (s), rounded to whole frames.
        phases (tuple[str, ...]): Trial phases to include, e.g. ``("task",)``.
        categories (list[int]): Pulse counts to report.
        measures (tuple[str, ...]): Which of ``F``, ``dF/F0``, ``F z``, ``dF/F0 z`` to compute,
            for each channel in *views*. The z measures need frames before the outcome.

    Returns:
        dict: ``offsets`` (frames), ``period_s``, ``means`` (``{view: {pulses: (n_cells,
        n_window)}}`` for the requested measures), ``sems`` (the same, SEM across trials),
        ``n_trials`` per pulse count, ``n_cells``, ``n_events_dropped`` (outcomes without a
        complete window of imaged frames), ``n_trials_no_baseline`` (trials left out because
        no pre-event frame remained for a z baseline; not in ``n_trials``) and ``flat_cells``
        (``{z view: ROI indices}`` with no defined scale, whose z is NaN).

    Raises:
        RunMismatchError: If a view's frame count differs from the frame times'.
        ValueError: If *measures* is empty or has an unknown name, or a z measure is requested
            but the window has fewer than 2 pre-event frames.
    """
    check_measures(measures)
    n_frames = len(frame_times_s)
    for view, traces in views.items():
        if traces.shape[1] != n_frames:
            raise RunMismatchError(
                f"{view} has {traces.shape[1]} frames but there are {n_frames} frame times; "
                "the traces and the frame clock are not from the same acquisition."
            )
    period = frame_period_s(frame_times_s)
    pre, post = window_frame_counts(window_s, period)
    wants_z = any(measure.endswith(" z") for measure in measures)
    if wants_z and pre < MIN_PRE_FRAMES_FOR_Z:
        raise ValueError(
            f"Window {window_s} s has {pre} pre-event frame(s); the z measures need at least "
            f"{MIN_PRE_FRAMES_FOR_Z} pre-event frames for their baseline (the frame an event "
            "falls in is left out of it); start the window earlier or ask for raw measures only."
        )
    views, flat_cells = _requested_views(views, measures)
    trials = outcomes[outcomes["phase"].isin(phases)]
    index, offset_s, near_frame = event_frames(
        frame_times_s, trials["outcome_time"].to_numpy(float)
    )
    valid = np.isfinite(frame_times_s)
    first = next(iter(views.values()))
    _, complete = frame_windows(first[:1], index[near_frame], pre, post, valid)
    kept = np.flatnonzero(near_frame)[complete]
    event_before_frame = offset_s[kept] < 0  # the frame before it was being acquired
    # a trial with no baseline frame would be NaN in the z views yet counted in n_trials
    no_baseline = (
        trials_without_baseline(pre, event_before_frame)
        if wants_z
        else np.zeros(len(kept), dtype=bool)
    )
    n_without_baseline = int(no_baseline.sum())
    kept, event_before_frame = kept[~no_baseline], event_before_frame[~no_baseline]
    pulses = trials["num_rew_pulses"].to_numpy(int)[kept]
    means, sems = {}, {}
    for view, traces in views.items():
        windows, _ = frame_windows(traces, index[kept], pre, post, valid)
        if view in Z_VIEWS:
            windows = baseline_subtract(windows, pre, event_before_frame)
        means[view] = mean_by_category(windows, pulses, categories)
        sems[view] = sem_by_category(windows, pulses, categories)
    counts = {c: int(np.sum(pulses == c)) for c in categories if np.any(pulses == c)}
    return {
        "offsets": window_offsets(pre, post),
        "period_s": period,
        "means": means,
        "sems": sems,
        "n_trials": counts,
        "n_cells": first.shape[0],
        "n_events_dropped": int(len(trials) - len(kept) - n_without_baseline),
        "n_trials_no_baseline": n_without_baseline,
        "flat_cells": flat_cells,
    }


def _requested_views(views: dict, measures: tuple[str, ...]) -> tuple[dict, dict]:
    """The requested raw and robust-z views of *views*, and the cells with no scale per z view."""
    out, flat_cells = {}, {}
    for channel, measure in RAW_VIEWS:
        if (channel, measure) not in views:
            continue
        if measure in measures:
            out[(channel, measure)] = views[(channel, measure)]
        if f"{measure} z" in measures:
            z_view = (channel, f"{measure} z")
            out[z_view], flat = robust_z(views[(channel, measure)])
            flat_cells[z_view] = np.flatnonzero(flat).tolist()
    return out, flat_cells


def plane_views(plane_dir) -> tuple[dict, object]:
    """Raw F and ΔF/F0 of both channels for every ROI of a plane, and the plane's traces.

    Raises:
        FileNotFoundError: If the plane has no Cellpose extraction record or no red traces.
    """
    from mini2p_preprocessing.config import FluorescenceSettings
    from mini2p_preprocessing.fluorescence import compute_dff
    from mini2p_preprocessing.plane import load_plane_traces

    plane_dir = Path(plane_dir)
    record_path = plane_dir / EXTRACTION_RECORD
    if not record_path.is_file():
        raise FileNotFoundError(f"{plane_dir} has no {EXTRACTION_RECORD}; extract it first.")
    settings = FluorescenceSettings(**json.loads(record_path.read_text())["fluorescence_settings"])
    traces = load_plane_traces(plane_dir)
    if traces.f_chan2 is None:
        raise FileNotFoundError(f"{plane_dir} has no red-channel traces (F_chan2.npy).")
    green = compute_dff(traces.f, traces.fneu, traces.fs, settings)
    red = compute_dff(traces.f_chan2, traces.fneu_chan2, traces.fs, settings)
    views = {
        ("green", "F"): traces.f,
        ("green", "dF/F0"): green.dff,
        ("red", "F"): traces.f_chan2,
        ("red", "dF/F0"): red.dff,
    }
    return views, traces
