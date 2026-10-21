"""Figures for reward-aligned imaging responses and a run's most responsive cells."""

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# 0 pulses (no reward) in red; 1-4 pulses from teal to dark blue, as in the photometry figures.
PULSE_COLORS = {0: "#C44E52", 1: "#41B6C4", 2: "#1D91C0", 3: "#225EA8", 4: "#0C2C84"}
CHANNEL_TITLES = {"green": "Green (functional)", "red": "Red (structural)"}
RANKING_TITLES = {"max_dff": "maximum dF/F", "p_cell": "p(cell)"}
MEASURE_LABELS = {
    "F": ("raw F (not neuropil-corrected)", "F (a.u.)"),
    "dF/F0": ("ΔF/F0", "ΔF/F0"),
    "F z": (
        "raw F (not neuropil-corrected), robust z, baseline-subtracted",
        "F (robust z, baseline-subtracted)",
    ),
    "dF/F0 z": ("ΔF/F0, robust z, baseline-subtracted", "ΔF/F0 (robust z, baseline-subtracted)"),
}


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
    measures: tuple[str, str] = ("F z", "dF/F0 z"),
):
    """Mean ± SEM across cells of each view, one line per number of reward pulses.

    Args:
        pooled (dict): ``{(channel, measure): {pulses: {"mean", "sem", "n_cells", "n_runs"}}}``.
        offsets (np.ndarray): Window offsets in frames (0 = outcome frame).
        period_s (float): Frame period, for an x axis in seconds.
        x_units (str): ``"s"`` or ``"frames"``.
        axes: A 2 x 2 array of axes (rows *measures*; columns green, red), or None for a new
            figure.
        title (str): Figure title.
        measures (tuple[str, str]): The two rows: ``"F"``, ``"dF/F0"`` or their robust-z,
            baseline-subtracted versions ``"F z"``, ``"dF/F0 z"`` (the default).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    fig = np.asarray(axes).flat[0].figure
    x, xlabel = _x(np.asarray(offsets), period_s, x_units)
    for col, channel in enumerate(("green", "red")):
        for row, measure in enumerate(measures):
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
            name, ylabel = MEASURE_LABELS[measure]
            ax.set_title(f"{CHANNEL_TITLES[channel]}: {name}")
            ax.set_ylabel(ylabel)
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


SPACING_MARGIN = 0.1  # fraction of the spacing left between neighbouring rows
SPACING_FLOOR = 0.05  # ΔF/F0 units; keeps rows apart when every trace is flat


def clip_traces(dff: np.ndarray, clip_percentiles: tuple[float, float] | None) -> np.ndarray:
    """Each trace limited to its own percentile range (NaN-safe), for display only.

    Args:
        dff (np.ndarray): ``(n_cells, n_frames)`` ΔF/F0.
        clip_percentiles (tuple[float, float] | None): Low and high percentile, e.g. ``(1, 99)``;
            None returns *dff* unchanged.
    """
    if clip_percentiles is None:
        return dff
    low, high = np.nanpercentile(dff, clip_percentiles, axis=1, keepdims=True)
    return np.clip(dff, low, high)


def trace_spacing(traces: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> float:
    """Vertical distance between stacked traces, constant over the rows.

    The largest, over adjacent rows (in plotted order), of the lower trace's high percentile
    minus the upper trace's low percentile, so the bulk of one row never reaches the next; a
    brief transient beyond the percentiles may. A trace's own range counts too. A margin
    is added and a floor keeps all-flat traces apart.

    Args:
        traces (np.ndarray): ``(n_cells, n_frames)`` ΔF/F0, best cell first.
        percentiles (tuple[float, float]): Low and high percentile of a trace's extent. Use
            ``(0, 100)`` for traces already clipped to their percentile range.
    """
    low, high = np.nanpercentile(traces, percentiles, axis=1)
    needed = np.concatenate([high - low, high[1:] - low[:-1]])
    return float(max((1 + SPACING_MARGIN) * np.nanmax(needed), SPACING_FLOOR))


def trace_baselines(
    traces: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)
) -> np.ndarray:
    """Baseline (y of ΔF/F0 = 0) of each stacked trace: row 0 at 0, each next one lower."""
    return -trace_spacing(traces, percentiles) * np.arange(len(traces))


def middle_frames(n_frames: int, n_shown: int | None) -> slice:
    """The centred stretch of *n_shown* frames out of *n_frames* (all of them if fewer).

    Args:
        n_frames (int): Frames in the run.
        n_shown (int | None): Frames to keep, e.g. ``8000``; None keeps every frame.

    Raises:
        ValueError: If *n_shown* is not positive.
    """
    if n_shown is None or n_shown >= n_frames:
        return slice(0, n_frames)
    if n_shown < 1:
        raise ValueError(f"middle_frames_shown must be positive, not {n_shown}.")
    start = (n_frames - n_shown) // 2
    return slice(start, start + n_shown)


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


def _draw_traces(
    ax,
    top: dict,
    x_units: str,
    colors: list[str],
    clip_percentiles: tuple[float, float] | None,
    frames: slice,
) -> None:
    """Stacked traces, one colour per ROI, with ``ROI <n>`` ticks and outcome markers above."""
    if len(top["cells"]) == 0:
        ax.text(0.5, 0.5, "no ROIs to show", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x, xlabel, events = _x_axis(top, x_units)
    x = x[frames]
    # an outcome outside the frames shown would stretch the axis, so it is not marked
    events = np.where((events >= np.nanmin(x)) & (events <= np.nanmax(x)), events, np.nan)
    shown = clip_traces(top["dff"][:, frames], clip_percentiles)
    # clipped traces end exactly at their percentiles, so the spacing is measured at the ends
    spacing_percentiles = (0, 100) if clip_percentiles else (0.5, 99.5)
    spacing = trace_spacing(shown, spacing_percentiles)
    baselines = -spacing * np.arange(len(shown))
    for base, trace, color in zip(baselines, shown, colors, strict=True):
        ax.plot(x, base + trace, color=color, linewidth=0.8)
    marker_bottom = np.nanmax(shown[0]) + 0.1 * spacing
    for time, pulses in zip(events, top["outcomes"]["num_rew_pulses"], strict=True):
        ax.plot(
            [time, time],
            [marker_bottom, marker_bottom + 0.4 * spacing],
            color=PULSE_COLORS[pulses],
        )
    for pulses in sorted(set(top["outcomes"]["num_rew_pulses"])):
        ax.plot([], [], color=PULSE_COLORS[pulses], label=f"{pulses} pulses")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncols=5)
    ax.set_yticks(
        baselines,
        [f"ROI {cell} (p={p:.2f})" for cell, p in zip(top["cells"], top["p_cell"], strict=True)],
    )
    ax.set_ylim(baselines[-1] + np.nanmin(shown[-1]) - 0.1 * spacing, marker_bottom + 0.6 * spacing)
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
    # sized by the ROI numbers too: a top ROI fully hidden by overlaps has no label pixels
    n_labels = max(int(labels.max()), int(top["cells"].max()) + 1 if len(top["cells"]) else 0)
    lut = np.zeros((n_labels + 1, 4))
    if len(top["cells"]):
        lut[top["cells"] + 1] = [matplotlib.colors.to_rgba(c) for c in colors]
    overlay = np.zeros((*image.shape, 4))
    edge = _outline(chosen)
    overlay[edge] = lut[chosen[edge]]
    ax.imshow(overlay)
    hidden = []
    for cell, color in zip(top["cells"], colors, strict=True):
        ys, xs = np.nonzero(labels == cell + 1)
        if len(ys):
            ax.text(xs.mean(), ys.mean(), str(cell), color=color, fontsize=6, ha="center")
        else:
            hidden.append(f"ROI {cell}")
    ax.set_axis_off()
    note = f"\nnot visible (hidden by overlaps): {', '.join(hidden)}" if hidden else ""
    ax.set_title("Mean image; top cells outlined" + note)


def top_cells_pdf_name(n: int, rank_by: str, tag: str, variant: str, x_units: str) -> str:
    """File name of the PDF of one ranking's top cells, e.g. ``top30_cells_by_p_cell_<tag>_...``."""
    return f"top{n}_cells_by_{rank_by}_{tag}_{variant}_{x_units}.pdf"


def _ordinal(value: float) -> str:
    number = int(value) if float(value).is_integer() else value
    suffix = (
        "th"
        if 10 <= number % 100 < 20 or number % 10 not in (1, 2, 3)
        else ("st", "nd", "rd")[number % 10 - 1]
    )
    return f"{number}{suffix}"


def plot_top_cells(
    top: dict,
    x_units: str = "s",
    title: str = "",
    variant: str = "",
    clip_percentiles: tuple[float, float] | None = (1, 99),
    middle_frames_shown: int | None = None,
):
    """Stacked ΔF/F0 traces of a run's most responsive cells, and where they are.

    Each ROI has one colour, used for its trace and for its outline and number in the field of
    view. The best ROI is on top; ``ROI <n> (p=<p(cell)>)`` tick labels mark each trace's
    baseline. Outcomes are marked above the traces, coloured by number of reward pulses.
    Each trace is drawn clipped to its own percentile range, so a brief transient does not set
    the row spacing; the ranking behind *top* uses the unclipped values.

    Args:
        top (dict): From ``src.funcimg_analysis.collect.top_cells``.
        x_units (str): ``"s"`` (pyControl time) or ``"frames"`` (frame index).
        title (str): Figure title, e.g. the session and run.
        variant (str): Destriping variant, named in the traces' title.
        clip_percentiles (tuple[float, float] | None): Per-trace display range in percentiles,
            e.g. ``(1, 99)``; None draws the traces unclipped. The percentiles are taken over the
            frames shown.
        middle_frames_shown (int | None): Draw only this many frames from the middle of the
            run, e.g. ``8000``; outcomes outside them are not marked. Default None, every frame.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, (ax_traces, ax_fov) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True
    )
    colors = roi_colors(len(top["cells"]))
    n_frames = top["dff"].shape[1]
    frames = middle_frames(n_frames, middle_frames_shown)
    _draw_traces(ax_traces, top, x_units, colors, clip_percentiles, frames)
    n = len(top["cells"])
    ranking = RANKING_TITLES[top["rank_by"]]
    heading = f"{variant} green: top {n} ROIs by {ranking}".strip()
    if clip_percentiles:
        low, high = clip_percentiles
        heading += f"\nclipped to {_ordinal(low)}–{_ordinal(high)} percentile per ROI"
    if frames.stop - frames.start < n_frames:
        heading += f"; middle {frames.stop - frames.start} of {n_frames} frames"
    ax_traces.set_title(heading)
    _draw_fov(ax_fov, top, colors)
    fig.suptitle(title)
    return fig
