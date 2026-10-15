"""Trial-based companion figures for the session-based reversal plots."""

import math

import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis.reversals_by_trial import reversal_trial_series, trailing_reversal_rate
from src.behavior_visualization.plot_helpers import save_or_show
from src.behavior_visualization.plot_style import BAD_COLOR, GOOD_COLOR, TOTAL_COLOR


def plot_reversals_by_trial(subjects_trials, *, window=None, threshold=None, save_path=None):
    series = reversal_trial_series(subjects_trials)
    colors = {"good": GOOD_COLOR, "bad": BAD_COLOR, "total": TOTAL_COLOR}
    if window is not None:
        series = {
            m: {k: trailing_reversal_rate(v, window) for k, v in d.items()}
            for m, d in series.items()
        }
    offset = 1 if window is not None else 0
    ylabel = "Reversals per 100 trials" if window is not None else "Cumulative reversals"
    title = (
        f"Trailing {window}-trial reversal rate"
        if window is not None
        else "Cumulative reversals by trial"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    kinds = sorted({k for d in series.values() for k in d})
    for kind in kinds:
        arrays = [d[kind] for d in series.values() if kind in d]
        padded = np.full((len(arrays), max(map(len, arrays))), np.nan)
        for i, values in enumerate(arrays):
            padded[i, : len(values)] = values
            ax.plot(np.arange(len(values)) + offset, values, color=colors[kind], alpha=0.15, lw=1)
        mean = np.nanmean(padded, axis=0)
        ax.plot(
            np.arange(len(mean)) + offset,
            mean,
            color=colors[kind],
            lw=2.5,
            label=f"{kind.title()} (mean of observed mice)",
        )
    if window is None and threshold is not None:
        ax.axhline(threshold, color="gray", ls="--", label="Threshold")
    ax.set(xlabel="Number of trials completed", ylabel=ylabel, title=title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    mice = sorted(series)
    fig2, axes = plt.subplots(
        math.ceil(len(mice) / 3), 3, figsize=(15, 4 * math.ceil(len(mice) / 3)), squeeze=False
    )
    for ax, mouse in zip(axes.flat, mice, strict=False):
        for kind, values in series[mouse].items():
            ax.plot(np.arange(len(values)) + offset, values, color=colors[kind], label=kind.title())
        if window is None and threshold is not None:
            ax.axhline(threshold, color="gray", ls="--")
        ax.set(title=mouse, xlabel="Number of trials completed", ylabel=ylabel)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(mice) :]:
        ax.set_visible(False)
    fig2.suptitle(title)
    fig2.tight_layout()
    save_or_show(
        {" Across Mice": fig, " By Mouse": fig2},
        save_path or None,  # an empty path means "show"
        formats=("png", "pdf"),
        dpi={"png": 180, "pdf": 180},
    )
