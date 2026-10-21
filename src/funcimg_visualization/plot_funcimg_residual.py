"""Figures for the background-corrected ΔF/F0: pooled responses and example traces."""

import matplotlib.pyplot as plt
import numpy as np

from src.funcimg_analysis.event_frames import robust_z
from src.funcimg_visualization.plot_funcimg_responses import (
    CHANNEL_TITLES,
    middle_frames,
    plot_pulse_responses,
    window_tag,
)

CHANNELS = ("green", "red")
MEASURE = "dF/F0 z"
SUITE2P_COLOR, CORRECTED_COLOR = "0.55", "#D55E00"


def dff_comparison_png_name(
    file_tag: str, variant: str, x_units: str, window_s: tuple[float, float]
) -> str:
    """File name of the Suite2p vs background-corrected ΔF/F0 response figure."""
    return f"bg_corrected_dff_{file_tag}_{variant}_{x_units}_{window_tag(window_s)}.png"


def plot_dff_comparison(
    suite2p: dict,
    corrected: dict,
    offsets: np.ndarray,
    period_s: float,
    x_units: str = "s",
    axes=None,
    title: str = "",
):
    """Pooled reward responses of Suite2p's ΔF/F0 next to the background-corrected ΔF/F0.

    Columns are green Suite2p, green background-corrected, red Suite2p, red background-corrected,
    all on the same y limits so the sizes can be compared directly.

    Args:
        suite2p (dict): ``pool_runs`` output for ``"dF/F0 z"`` with Suite2p's ΔF/F0.
        corrected (dict): The same for the background-corrected ΔF/F0.
        offsets (np.ndarray): Window offsets in frames.
        period_s (float): Frame period (s).
        x_units (str): ``"s"`` or ``"frames"``.
        axes: A ``(1, 4)`` array of axes, or None for a new figure.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If *axes* is not ``(1, 4)``.
    """
    if axes is None:
        _, axes = plt.subplots(
            1, 4, figsize=(18, 4.2), sharex=True, constrained_layout=True, squeeze=False
        )
    axes = np.asarray(axes)
    if axes.shape != (1, 4):
        raise ValueError(f"Need a 1 x 4 array of axes, got {axes.shape}.")
    for pooled, columns, name in (
        (suite2p, [0, 2], "Suite2p"),
        (corrected, [1, 3], "bg-corrected"),
    ):
        present = {key: value for key, value in pooled.items()}
        for channel in CHANNELS:
            present.setdefault((channel, MEASURE), {})  # drawn empty, then labelled below
        plot_pulse_responses(
            present, offsets, period_s, x_units, axes[:, columns], measures=(MEASURE,)
        )
        for column, channel in zip(columns, CHANNELS, strict=True):
            ax = axes[0, column]
            ax.set_title(
                f"{name} dF/F0, {CHANNEL_TITLES[channel]}\nrobust z, baseline-subtracted",
                fontsize=8,
            )
            ax.set_ylabel("ΔF/F0 (robust z, baseline-subtracted)")
            if not pooled.get((channel, MEASURE)):
                ax.text(
                    0.5, 0.5, "no usable cells", ha="center", va="center", transform=ax.transAxes
                )
    low = min(ax.get_ylim()[0] for ax in axes[0])
    high = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(low, high)
    axes[0, 0].figure.suptitle(
        title or "Reward responses of Suite2p's and the background-corrected ΔF/F0"
    )
    return axes[0, 0].figure


def plot_example_traces(
    example: dict,
    time_s: np.ndarray,
    n_frames_shown: int | None = 2000,
    axes=None,
    title: str = "",
):
    """Suite2p's and the background-corrected ΔF/F0 of a few cells over the middle frames.

    The two ΔF/F0 have different scales and baselines (the corrected F0 includes the cell's
    neuropil, which Suite2p subtracts), so each trace is drawn as a robust z over the whole run
    and the two overlaid on one axis per cell show the shapes, not the magnitudes.

    Args:
        example (dict): ``cells`` (ROI indices), ``suite2p`` and ``corrected`` (both
            ``(n_cells, n_frames)``).
        time_s (np.ndarray): pyControl time (s) of each frame.
        n_frames_shown (int | None): Frames drawn, centred in the run; None draws every frame.
        axes: One axis per cell, or None for a new figure.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If there is no example cell, the two ΔF/F0 arrays differ in shape or do not
            match the frame times, or *axes* has the wrong number of axes.
    """
    suite2p, corrected = np.asarray(example["suite2p"]), np.asarray(example["corrected"])
    n = len(example["cells"])
    if n == 0:
        raise ValueError("no example cell: every ROI of the channel is excluded.")
    if suite2p.shape != corrected.shape or suite2p.shape[1] != len(time_s):
        raise ValueError(
            f"Suite2p ({suite2p.shape}) and corrected ({corrected.shape}) ΔF/F0 must have the "
            f"same shape, with {len(time_s)} frames."
        )
    if axes is None:
        _, axes = plt.subplots(
            n, 1, figsize=(12, 2.2 * n + 1), sharex=True, constrained_layout=True, squeeze=False
        )
    axes = np.asarray(axes).reshape(-1)
    if len(axes) != n:
        raise ValueError(f"Need {n} axes, one per example cell, got {len(axes)}.")
    frames = middle_frames(len(time_s), n_frames_shown)
    before, after = robust_z(suite2p)[0], robust_z(corrected)[0]
    for ax, roi, trace_a, trace_b in zip(axes, example["cells"], before, after, strict=True):
        ax.plot(
            time_s[frames],
            trace_a[frames],
            color=SUITE2P_COLOR,
            linewidth=0.8,
            label="Suite2p dF/F0",
        )
        ax.plot(
            time_s[frames],
            trace_b[frames],
            color=CORRECTED_COLOR,
            linewidth=0.8,
            label="bg-corrected dF/F0",
        )
        ax.set_ylabel(f"ROI {roi}\nrobust z", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, ncols=2, loc="upper right")
    axes[-1].set_xlabel("pyControl time (s)")
    figure = axes[0].figure
    figure.suptitle(title or "Example cells: Suite2p vs background-corrected ΔF/F0")
    figure.text(
        0.01, 0.005,
        "Each ΔF/F0 is shown as a robust z. The corrected ΔF/F0's F0 includes the cell's "
        "neuropil (Suite2p subtracts it), so the two baselines differ.",
        fontsize=7, ha="left", va="bottom",
    )  # fmt: skip
    return figure
