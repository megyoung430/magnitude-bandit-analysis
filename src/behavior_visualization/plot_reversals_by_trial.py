"""Trial-based companion figures for the session-based reversal plots."""

import math
from pathlib import Path  # noqa: F401  # public to star-importing notebooks

import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis.reversals_by_trial import reversal_trial_series, trailing_reversal_rate
from src.behavior_visualization.plot_helpers import Title, save_or_show, set_title, stats_line
from src.behavior_visualization.plot_style import BAD_COLOR, GOOD_COLOR, TOTAL_COLOR


def plot_reversals_by_trial(subjects_trials, *, window=None, threshold=None, save_path=None):
    """Plot reversals against the number of trials completed, across mice and per mouse.

    Saves two figures, ``<save_path> Across Mice`` (faint per-mouse lines plus the mean of the
    mice observed at each trial) and ``<save_path> By Mouse`` (one panel per mouse, three
    panels per row), each in every figure format with a caption file. Without *window* the y
    axis is the cumulative reversal count; with it, a trailing reversal rate per 100 trials.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session_dict}}``.
        window (int | None): Trailing window in trials for the reversal rate (a positive
            integer); None plots cumulative counts instead. Default None.
        threshold (float | None): Dashed reference line at this cumulative count; ignored when
            *window* is given. Default None.
        save_path (str | Path | None): Base path without extension; None or empty shows the
            figures instead. Default None.

    Returns:
        None

    Raises:
        ValueError: If no mouse has trials, a mouse mixes counter and blocks-only sessions, a
            mouse's reversal counters are not aligned and cumulative, or *window* is not a
            positive integer.
    """
    series = reversal_trial_series(subjects_trials)
    colors = {"good": GOOD_COLOR, "bad": BAD_COLOR, "total": TOTAL_COLOR}
    if window is not None:
        series = {
            m: {k: trailing_reversal_rate(v, window) for k, v in d.items()}
            for m, d in series.items()
        }
    offset = 1 if window is not None else 0
    ylabel = "Reversals per 100 trials" if window is not None else "Cumulative reversals"
    heading = (
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
    ax.set(xlabel="Number of trials completed", ylabel=ylabel)
    caption = ("Faint lines: each mouse; bold: the mean of the mice observed at each trial.",)
    stats = stats_line("mean", "mice", [(len(series), "mouse", "mice")])
    set_title(ax, Title(heading, stats, caption))
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
    fig2.suptitle(heading)
    fig2.tight_layout()
    save_or_show(
        {" Across Mice": fig, " By Mouse": fig2},
        save_path or None,  # an empty path means "show"
        dpi={"png": 180, "pdf": 180},
    )
