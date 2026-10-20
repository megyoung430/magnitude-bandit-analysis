"""Figures for reward-aligned imaging responses and a run's most responsive cells."""

import matplotlib.pyplot as plt
import numpy as np

# 0 pulses (no reward) in red; 1-4 pulses from teal to dark blue, as in the photometry figures.
PULSE_COLORS = {0: "#C44E52", 1: "#41B6C4", 2: "#1D91C0", 3: "#225EA8", 4: "#0C2C84"}
CHANNEL_TITLES = {"green": "Green (functional)", "red": "Red (structural)"}


def _x(offsets: np.ndarray, period_s: float, x_units: str) -> tuple[np.ndarray, str]:
    if x_units == "frames":
        return offsets, "Frames from outcome"
    if x_units == "s":
        return offsets * period_s, "Time from outcome (s)"
    raise ValueError(f"x_units must be 's' or 'frames', not {x_units!r}.")


def plot_pulse_responses(
    pooled: dict,
    offsets: np.ndarray,
    period_s: float,
    x_units: str = "s",
    axes=None,
    title: str = "",
):
    """Mean ± SEM across cells of each view, one line per number of reward pulses.

    Args:
        pooled (dict): ``{(channel, measure): {pulses: {"mean", "sem", "n_cells", "n_runs"}}}``.
        offsets (np.ndarray): Window offsets in frames (0 = outcome frame).
        period_s (float): Frame period, for an x axis in seconds.
        x_units (str): ``"s"`` or ``"frames"``.
        axes: A 2 x 2 array of axes (rows F, ΔF/F0; columns green, red), or None for a new figure.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    fig = np.asarray(axes).flat[0].figure
    x, xlabel = _x(np.asarray(offsets), period_s, x_units)
    for col, channel in enumerate(("green", "red")):
        for row, measure in enumerate(("F", "dF/F0")):
            ax = np.asarray(axes)[row, col]
            for pulses, stats in sorted(pooled[(channel, measure)].items()):
                label = f"{pulses} pulses ({stats['n_cells']} cells, {stats['n_runs']} runs)"
                ax.plot(x, stats["mean"], color=PULSE_COLORS[pulses], label=label)
                ax.fill_between(
                    x,
                    stats["mean"] - stats["sem"],
                    stats["mean"] + stats["sem"],
                    color=PULSE_COLORS[pulses],
                    alpha=0.2,
                    linewidth=0,
                )
            ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--")
            ax.set_title(f"{CHANNEL_TITLES[channel]}: {'raw F' if measure == 'F' else 'ΔF/F0'}")
            ax.set_ylabel("F (a.u.)" if measure == "F" else "ΔF/F0")
            if row == 1:
                ax.set_xlabel(xlabel)
    np.asarray(axes)[0, 1].legend(fontsize=8, frameon=False)
    fig.suptitle(title or "Outcome-aligned response, mean ± SEM across cells")
    return fig


def _outline(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edge & (labels > 0)


def plot_top_cells(top: dict, x_units: str = "s", title: str = ""):
    """Stacked ΔF/F0 traces of a run's most responsive cells, and where they are.

    Outcomes are marked above the traces, coloured by number of reward pulses.

    Args:
        top (dict): From ``src.funcimg_analysis.collect.top_cells``.
        x_units (str): ``"s"`` (pyControl time) or ``"frames"`` (frame index).
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, (ax_traces, ax_fov) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True
    )
    if x_units == "s":
        x, xlabel = top["time_s"], "pyControl time (s)"
        events = top["outcomes"]["outcome_time"].to_numpy(float)
    elif x_units == "frames":
        x, xlabel = np.arange(top["dff"].shape[1]), "Frame"
        known = np.isfinite(top["time_s"])
        events = np.interp(
            top["outcomes"]["outcome_time"].to_numpy(float),
            top["time_s"][known],
            np.flatnonzero(known),
        )
    else:
        raise ValueError(f"x_units must be 's' or 'frames', not {x_units!r}.")
    spacing = max(float(np.nanpercentile(top["dff"], 99)), 0.5) if top["dff"].size else 1.0
    for rank, (cell, trace) in enumerate(zip(top["cells"], top["dff"], strict=True)):
        base = -rank * spacing
        ax_traces.plot(x, base + trace, color="k", linewidth=0.5)
        ax_traces.text(np.nanmin(x), base, f"ROI {cell} ", ha="right", va="bottom", fontsize=7)
    top_y = spacing
    for time, pulses in zip(events, top["outcomes"]["num_rew_pulses"], strict=True):
        ax_traces.plot([time, time], [top_y, top_y + 0.4 * spacing], color=PULSE_COLORS[pulses])
    for pulses in sorted(set(top["outcomes"]["num_rew_pulses"])):
        ax_traces.plot([], [], color=PULSE_COLORS[pulses], label=f"{pulses} pulses")
    ax_traces.legend(loc="upper right", fontsize=7, frameon=False, ncols=5)
    ax_traces.set_xlabel(xlabel)
    ax_traces.set_yticks([])
    ax_traces.set_title(f"Green ΔF/F0 of the {len(top['cells'])} cells with the largest max ΔF/F0")
    image = top["mean_image"]
    low, high = np.nanpercentile(image, [1, 99.5])
    ax_fov.imshow(image, cmap="gray", vmin=low, vmax=high)
    chosen = np.where(np.isin(top["labels"], top["cells"] + 1), top["labels"], 0)
    overlay = np.zeros((*image.shape, 4))
    overlay[_outline(chosen)] = (1.0, 0.3, 0.1, 1.0)
    ax_fov.imshow(overlay)
    for cell in top["cells"]:
        ys, xs = np.nonzero(top["labels"] == cell + 1)
        if len(ys):
            ax_fov.text(xs.mean(), ys.mean(), str(cell), color="yellow", fontsize=6, ha="center")
    ax_fov.set_axis_off()
    ax_fov.set_title("Mean image; top cells outlined")
    fig.suptitle(title)
    return fig
