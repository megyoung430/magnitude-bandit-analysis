"""Synthetic runs for the shared-signal tests: a field signal, private noise and a reward
response that can sit in the background, in the cells, or leak a little into both."""

import numpy as np
import pandas as pd

from src.funcimg_analysis.event_frames import pooled_mean_sem
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES
from src.funcimg_analysis.shared_signal import mean_response
from src.funcimg_analysis.shared_signal_run import analyse_shared_signal

PERIOD = 0.133
N_FRAMES = 900
N_CELLS = 4  # the plane fixture's cell count
EVENT_FRAMES = np.arange(40, N_FRAMES - 30, 20)
PULSES = np.arange(len(EVENT_FRAMES)) % 5
SETTINGS = {
    "detrend_window_s": 15.0,
    "min_shift_s": 10.0,
    "n_shifts": 20,
    "fit_exclude_window_s": (-0.3, 1.6),  # the ~1 s response, so a quarter of the frames remain
}
WINDOW_S = (-1.5, 3.0)
MEASURES = ("F z", "dF/F0 z")


def smooth(rng, n=N_FRAMES, width=9):
    raw = rng.normal(size=n + width)
    out = np.convolve(raw, np.ones(width) / np.sqrt(width), mode="valid")[:n]
    return out / out.std()


def reward_bump():
    """A response after every outcome, growing with the number of reward pulses."""
    bump = np.zeros(N_FRAMES)
    for frame, pulses in zip(EVENT_FRAMES, PULSES, strict=True):
        bump[frame : frame + 8] += pulses  # ~1 s
    return bump


def make_traces(rng, reward_in: str, n_cells=N_CELLS):
    """Background and cell traces: shared field + private noise, with the reward placed.

    ``reward_in``: ``"none"``; ``"background"`` (the cells see the same signal as the
    background); ``"cells"`` (private to the cells); ``"spill"`` (the cells' own response plus a
    small leak of it into the background).
    """
    field, bump = smooth(rng), reward_bump()
    bg_weight = {"background": 60, "spill": 15}.get(reward_in, 0)
    shared = reward_in == "background"
    own = {"cells": 80, "spill": 80}.get(reward_in, 0)
    background = 1000 + 30 * field + bg_weight * bump
    weights = rng.uniform(0.8, 1.2, n_cells)
    cells = np.array(
        [
            2000 + 40 * w * field + 30 * smooth(rng) + 80 * w * bump * shared + own * bump
            for w in weights
        ]
    )
    return background, cells


def neuropil_and_halves(rng, background, cells):
    """Each cell's neuropil (carries part of the field) and two halves of the background."""
    n = len(cells)
    neuropil = np.array([500 + 8 * smooth(rng) + 0.2 * (background - 1000) for _ in range(n)])
    halves = np.array([background + 2 * smooth(rng), background + 2 * smooth(rng)])
    return neuropil, halves


def frame_times():
    return 20.0 + PERIOD * np.arange(N_FRAMES)


def outcomes():
    return pd.DataFrame(
        {
            "outcome_time": frame_times()[EVENT_FRAMES] + 0.01,
            "num_rew_pulses": PULSES,
            "phase": "task",
        }
    )


def run_inputs(rng, reward_in, channels=("green",), reward_in_red="none"):
    """``views, background, neuropil, background_halves`` of the requested channels."""
    views, background, neuropil, halves = {}, {}, {}, {}
    for channel in channels:
        bg, cells = make_traces(rng, reward_in if channel == "green" else reward_in_red)
        np_, half = neuropil_and_halves(rng, bg, cells)
        views[(channel, "F")], views[(channel, "dF/F0")] = cells, cells / 2000 - 1
        background[(channel, "F")], neuropil[(channel, "F")] = bg, np_
        halves[(channel, "F")] = half
    return views, background, neuropil, halves


def analyse(inputs, **overrides):
    views, background, neuropil, halves = inputs
    settings = {**SETTINGS, "window_s": WINDOW_S, "phases": ("task",), "measures": MEASURES}
    settings.update(overrides)
    return analyse_shared_signal(
        views,
        background,
        frame_times(),
        outcomes(),
        neuropil=neuropil,
        background_halves=halves,
        rng=np.random.default_rng(0),
        **settings,
    )


def response_2s(result, which, pulses=4, view=("green", "F z")):
    pooled = {view: pooled_mean_sem([result[which]["means"][view]], PULSE_CATEGORIES)}
    return mean_response(pooled, result[which]["offsets"], PERIOD)[view][pulses]
