"""Synthetic fluorescence with a known ground truth for the Zong et al. 2022 pipeline tests.

A cell is a slowly drifting baseline F0 plus calcium transients (instant rise, exponential
decay, as a spike train convolved with the indicator kernel) and Gaussian noise, so the tests
can check every step against what was planted.
"""

from dataclasses import dataclass

import numpy as np

PERIOD_S = 0.133
FS_HZ = 1 / PERIOD_S
N_FRAMES = 6000  # about 800 s
F0_LEVEL = 100.0
NOISE_SD = 1.0  # 1 % of F0
DECAY_S = 1.0


@dataclass(frozen=True)
class PlantedCell:
    fcorr: np.ndarray
    f0_true: np.ndarray
    spike_frames: np.ndarray  # the long, large transients that must be found
    short_frames: np.ndarray  # a 3-frame bump (0.4 s): too short to count
    weak_frames: np.ndarray  # a long bump below twice the noise: too small to count


def kernel(tau_s: float, n_frames: int = 200) -> np.ndarray:
    return np.exp(-np.arange(n_frames) * PERIOD_S / tau_s)


def spike_train(n_frames: int, frames, amplitudes) -> np.ndarray:
    train = np.zeros(n_frames)
    train[np.asarray(frames, dtype=int)] = amplitudes
    return train


def drifting_baseline(n_frames: int = N_FRAMES, amplitude: float = 5.0, period_s: float = 400.0):
    t = np.arange(n_frames) * PERIOD_S
    return F0_LEVEL + amplitude * np.sin(2 * np.pi * t / period_s)


def planted_cell(rng: np.random.Generator, n_frames: int = N_FRAMES) -> PlantedCell:
    """One cell: sparse large transients on a slow drift, plus two decoys that must be ignored."""
    f0_true = drifting_baseline(n_frames)
    spike_frames = np.arange(300, n_frames - 300, 450)  # one every ~60 s
    short_frames = np.array([n_frames // 2 + 100])
    weak_frames = np.array([n_frames // 2 + 1500])
    dff = np.convolve(spike_train(n_frames, spike_frames, 1.0), kernel(DECAY_S))[:n_frames]
    short = np.zeros(n_frames)
    short[short_frames[0] : short_frames[0] + 3] = 0.5  # 3 frames = 0.4 s
    weak = np.zeros(n_frames)
    weak[weak_frames[0] : weak_frames[0] + 40] = 0.012  # 5 s, but only 1.2 noise SDs
    fcorr = f0_true * (1 + dff + short + weak) + rng.normal(0, NOISE_SD, n_frames)
    return PlantedCell(fcorr, f0_true, spike_frames, short_frames, weak_frames)


def planted_cells(rng: np.random.Generator, n_cells: int = 3, n_frames: int = N_FRAMES):
    cells = [planted_cell(rng, n_frames) for _ in range(n_cells)]
    return np.vstack([c.fcorr for c in cells]), cells
