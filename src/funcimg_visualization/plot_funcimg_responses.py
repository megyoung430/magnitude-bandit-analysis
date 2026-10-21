"""Figures for reward-aligned imaging responses and a run's most responsive cells."""

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# 0 pulses (no reward) in red; 1-4 pulses from teal to dark blue, as in the photometry figures.
PULSE_COLORS = {0: "#C44E52", 1: "#41B6C4", 2: "#1D91C0", 3: "#225EA8", 4: "#0C2C84"}
CHANNEL_TITLES = {"green": "Green (functional)", "red": "Red (structural)"}


def _x(offsets: np.ndarray, period_s: float, x_units: str) -> tuple[np.ndarray, str]:
    if x_units == "frames":
        return offsets, "Frames from the outcome's frame"
    if x_units == "s":
        return offsets * period_s, "Time from outcome to frame start (s)"
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


def roi_colors(n: int) -> list[str]:
    """Hex colours for *n* ROIs, cycling matplotlib's default ``tab10`` order."""
    cycle = plt.get_cmap("tab10").colors
    return [matplotlib.colors.to_hex(cycle[i % len(cycle)]) for i in range(n)]


def trace_spacing(dff: np.ndarray) -> float:
    """Vertical distance between stacked traces.

    The 99.5th percentile of the per-trace range (max - min), plus a margin, so adjacent
    traces clear each other without a single huge transient setting the scale when there
    are many traces.

    Args:
        dff (np.ndarray): ``(n_cells, n_frames)`` ΔF/F0.
    """
    ranges = np.nanmax(dff, axis=1) - np.nanmin(dff, axis=1)
    return float(1.1 * np.nanpercentile(ranges, 99.5))


def trace_baselines(dff: np.ndarray) -> np.ndarray:
    """Baseline (y of ΔF/F0 = 0) of each stacked trace: row 0 at 0, each next one lower."""
    return -trace_spacing(dff) * np.arange(len(dff))


def _x_axis(top: dict, x_units: str) -> tuple[np.ndarray, str, np.ndarray]:
    """Frame positions, axis label and outcome positions for ``x_units``."""
    times = top["outcomes"]["outcome_time"].to_numpy(float)
    if x_units == "s":
        return top["time_s"], "pyControl time (s)", times
    if x_units == "frames":
        known = np.isfinite(top["time_s"])
        imaged = (times >= top["time_s"][known][0]) & (times <= top["time_s"][known][-1])
        # outside the imaged span np.interp would pin the mark to the first or last frame
        events = np.where(
            imaged, np.interp(times, top["time_s"][known], np.flatnonzero(known)), np.nan
        )
        return np.arange(top["dff"].shape[1]), "Frames", events
    raise ValueError(f"x_units must be 's' or 'frames', not {x_units!r}.")


def _draw_traces(ax, top: dict, x_units: str, colors: list[str]) -> None:
    """Stacked traces, one colour per ROI, with ``ROI <n>`` ticks and outcome markers above."""
    x, xlabel, events = _x_axis(top, x_units)
    dff = top["dff"]
    baselines = trace_baselines(dff)
    spacing = trace_spacing(dff)
    for base, trace, color in zip(baselines, dff, colors, strict=True):
        ax.plot(x, base + trace, color=color, linewidth=0.8)
    marker_bottom = np.nanmax(dff[0]) + 0.1 * spacing
    for time, pulses in zip(events, top["outcomes"]["num_rew_pulses"], strict=True):
        ax.plot(
            [time, time],
            [marker_bottom, marker_bottom + 0.4 * spacing],
            color=PULSE_COLORS[pulses],
        )
    for pulses in sorted(set(top["outcomes"]["num_rew_pulses"])):
        ax.plot([], [], color=PULSE_COLORS[pulses], label=f"{pulses} pulses")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncols=5)
    ax.set_yticks(baselines, [f"ROI {cell}" for cell in top["cells"]])
    ax.set_ylim(baselines[-1] + np.nanmin(dff[-1]) - 0.1 * spacing, marker_bottom + 0.6 * spacing)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROI")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_fov(ax, top: dict, colors: list[str]) -> None:
    """Mean image with each top ROI outlined and numbered in its trace colour."""
    image = top["mean_image"]
    low, high = np.nanpercentile(image, [1, 99.5])
    ax.imshow(image, cmap="gray", vmin=low, vmax=high)
    labels = top["labels"]
    chosen = np.where(np.isin(labels, top["cells"] + 1), labels, 0)
    lut = np.zeros((labels.max() + 1, 4))
    lut[top["cells"] + 1] = [matplotlib.colors.to_rgba(c) for c in colors]
    overlay = np.zeros((*image.shape, 4))
    edge = _outline(chosen)
    overlay[edge] = lut[chosen[edge]]
    ax.imshow(overlay)
    for cell, color in zip(top["cells"], colors, strict=True):
        ys, xs = np.nonzero(labels == cell + 1)
        if len(ys):
            ax.text(xs.mean(), ys.mean(), str(cell), color=color, fontsize=6, ha="center")
    ax.set_axis_off()
    ax.set_title("Mean image; top cells outlined")


def plot_top_cells(top: dict, x_units: str = "s", title: str = "", variant: str = ""):
    """Stacked ΔF/F0 traces of a run's most responsive cells, and where they are.

    Each ROI has one colour, used for its trace and for its outline and number in the field of
    view. The best ROI is on top; ``ROI <n>`` tick labels mark each trace's baseline. Outcomes
    are marked above the traces, coloured by number of reward pulses.

    Args:
        top (dict): From ``src.funcimg_analysis.collect.top_cells``.
        x_units (str): ``"s"`` (pyControl time) or ``"frames"`` (frame index).
        title (str): Figure title, e.g. the session and run.
        variant (str): Destriping variant, named in the traces' title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, (ax_traces, ax_fov) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True
    )
    colors = roi_colors(len(top["cells"]))
    _draw_traces(ax_traces, top, x_units, colors)
    n = len(top["cells"])
    ax_traces.set_title(f"{variant} green: top {n} ROIs by maximum dF/F".strip())
    _draw_fov(ax_fov, top, colors)
    fig.suptitle(title)
    return fig
