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
    event_frames,
    frame_period_s,
    frame_windows,
    mean_by_category,
    window_offsets,
)
from src.funcimg_analysis.frame_times import RunMismatchError

VIEWS = (("green", "F"), ("green", "dF/F0"), ("red", "F"), ("red", "dF/F0"))
PULSE_CATEGORIES = [0, 1, 2, 3, 4]
EXTRACTION_RECORD = "cellpose_extraction.json"


def pulse_responses(
    views: dict,
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    categories: list[int] = PULSE_CATEGORIES,
) -> dict:
    """Each cell's mean window around the outcome, per pulse count, for every view.

    Args:
        views (dict): ``{(channel, measure): (n_cells, n_frames)}``.
        frame_times_s (np.ndarray): pyControl time of each frame (NaN where unknown).
        outcomes (pd.DataFrame): ``outcome_time``, ``num_rew_pulses`` and ``phase`` per trial.
        window_s (tuple[float, float]): Window around the outcome (s), rounded to whole frames.
        phases (tuple[str, ...]): Trial phases to include, e.g. ``("task",)``.
        categories (list[int]): Pulse counts to report.

    Returns:
        dict: ``offsets`` (frames), ``period_s``, ``means`` (``{view: {pulses: (n_cells,
        n_window)}}``), ``n_trials`` per pulse count, ``n_cells`` and ``n_events_dropped``
        (outcomes without a complete window of imaged frames).

    Raises:
        RunMismatchError: If a view's frame count differs from the frame times'.
    """
    n_frames = len(frame_times_s)
    for view, traces in views.items():
        if traces.shape[1] != n_frames:
            raise RunMismatchError(
                f"{view} has {traces.shape[1]} frames but there are {n_frames} frame times; "
                "the traces and the frame clock are not from the same acquisition."
            )
    period = frame_period_s(frame_times_s)
    pre, post = int(round(-window_s[0] / period)), int(round(window_s[1] / period))
    trials = outcomes[outcomes["phase"].isin(phases)]
    index, _, near_frame = event_frames(frame_times_s, trials["outcome_time"].to_numpy(float))
    valid = np.isfinite(frame_times_s)
    first = next(iter(views.values()))
    _, complete = frame_windows(first[:1], index[near_frame], pre, post, valid)
    kept = np.flatnonzero(near_frame)[complete]
    pulses = trials["num_rew_pulses"].to_numpy(int)[kept]
    means = {}
    for view, traces in views.items():
        windows, _ = frame_windows(traces, index[kept], pre, post, valid)
        means[view] = mean_by_category(windows, pulses, categories)
    counts = {c: int(np.sum(pulses == c)) for c in categories if np.any(pulses == c)}
    return {
        "offsets": window_offsets(pre, post),
        "period_s": period,
        "means": means,
        "n_trials": counts,
        "n_cells": first.shape[0],
        "n_events_dropped": int(len(trials) - len(kept)),
    }


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
