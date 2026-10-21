"""Synthetic imaging runs for the reward-cell tests: noise cells with a planted reward response."""

import numpy as np
import pandas as pd

PERIOD = 0.133
N_FRAMES = 6000
N_EVENTS = 140
SETTINGS = {
    "phases": ("task",),
    "plot_window_s": (-2.0, 5.0),
    "baseline_s": (-2.0, 0.0),
    "response_s": (0.0, 2.0),
    "contrast": "rewarded",
    "rank_on": "split_half",
    "n_shifts": 100,
    "min_shift_s": 30.0,
}


def frame_times():
    return 20.0 + PERIOD * np.arange(N_FRAMES)


def event_frames(rng):
    """Outcome frames with irregular spacing, so that no circular shift realigns them."""
    gaps = rng.integers(30, 50, N_EVENTS)
    return 60 + np.cumsum(gaps)


def outcomes(rng, frames=None):
    frames = event_frames(rng) if frames is None else frames
    pulses = np.arange(len(frames)) % 5
    return (
        pd.DataFrame(
            {
                "outcome_time": frame_times()[frames] + 0.01,
                "num_rew_pulses": pulses,
                "phase": "task",
            }
        ),
        frames,
        pulses,
    )


def smooth_noise(rng, n_cells, width=5):
    raw = rng.normal(size=(n_cells, N_FRAMES + width))
    kernel = np.ones(width) / np.sqrt(width)
    out = np.array([np.convolve(row, kernel, mode="valid")[:N_FRAMES] for row in raw])
    return out / out.std(axis=1, keepdims=True)


def make_cells(rng, frames, pulses, n_cells=100, n_planted=20, amplitude=1.0):
    """Noise traces; the first *n_planted* cells respond for ~1.5 s after rewarded outcomes."""
    traces = smooth_noise(rng, n_cells)
    for frame, count in zip(frames, pulses, strict=True):
        if count > 0:
            traces[:n_planted, frame : frame + 10] += amplitude * count / 2
    return 1.0 + 0.1 * traces  # a ΔF/F0-like trace around 1 with a 10 % noise scale


def small_run(seed, n_events, n_cells=200):
    """Pure-noise cells and *n_events* shuffled outcomes (a few trials per pulse count).

    The run is as long as its events need (about 27 s per event), so the shifted null always has
    room; returns ``(traces, frame_times, outcomes)``.
    """
    rng = np.random.default_rng(seed)
    n_frames = 40 + 200 * n_events + 400
    times = 20.0 + PERIOD * np.arange(n_frames)
    frames = 60 + np.cumsum(rng.integers(100, 200, n_events))
    pulses = rng.permutation(np.arange(n_events) % 5)
    outs = pd.DataFrame(
        {"outcome_time": times[frames] + 0.01, "num_rew_pulses": pulses, "phase": "task"}
    )
    raw = rng.normal(size=(n_cells, n_frames + 5))
    noise = np.array(
        [np.convolve(row, np.ones(5) / np.sqrt(5), mode="valid")[:n_frames] for row in raw]
    )
    return 1.0 + 0.1 * noise / noise.std(axis=1, keepdims=True), times, outs
